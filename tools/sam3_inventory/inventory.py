#!/usr/bin/env python3
"""Trace one SAM 3 forward pass and rank the aten ops.

The official weights are not used. The model is built with random initialization
on CPU, at the default 1008px input, and one image forward plus one tracker step
are recorded. Parameters are cast to bfloat16 because the ViT MLP's fused
kernel forces that dtype; see ``TRACE_DTYPE``.

FLOP totals for matmul, convolution, and scaled-dot-product attention come from
``torch.utils.flop_counter.FlopCounterMode`` (multiply-adds count as 2 flops).
That counter does not have formulas for layer norm, GELU, softmax, or
elementwise ops. Those get a separate, labeled estimate from call shapes.
Bytes are the aten-visible inputs and outputs, with aliased storages counted
once. They are not a measured bandwidth, and a fused attention kernel's
score matrix is not a separate tensor on this path.

This machine has no Metal GPU. Nothing here is an MPS measurement.
"""

from __future__ import annotations

import argparse
import collections
import csv
import gc
import json
import os
import re
import subprocess
import time
import traceback
import warnings
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import (
    FlopCounterMode,
    bmm_flop,
    conv_flop_count,
    mm_flop,
    sdpa_flop_count,
)

SAM3_SHA = "31cb7e624e4fe95301b1be1d4e809111593d2e45"
IMAGE_SIZE = 1008
# The ViT MLP calls perflib.fused.addmm_act, which casts that linear+activation
# to bfloat16. The following linear stays fp32 and then raises. Official CUDA
# inference runs the whole forward under bfloat16 autocast, so the trace casts
# the model and the normalized image to bfloat16 instead.
TRACE_DTYPE = torch.bfloat16

# Ops that only inspect a tensor. Counted, but not treated as compute.
_META_OPS = {
    "aten.size",
    "aten.sym_size",
    "aten.stride",
    "aten.sym_stride",
    "aten.numel",
    "aten.sym_numel",
    "aten.dim",
    "aten.storage_offset",
    "aten.sym_storage_offset",
    "aten.is_contiguous",
    "aten.sym_is_contiguous",
    "aten.is_strides_like_format",
    "aten.is_non_overlapping_and_dense",
    "aten.device",
    "prim.device",
    "prim.layout",
    "aten.dtype",
}


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "docs" / "ROADMAP.md").is_file():
            return candidate
    raise SystemExit("Could not find the metal-kernels repo root from this file.")


def rss_mib() -> int:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return -1


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} rss={rss_mib()}MiB] {message}", flush=True)


class CudaFactoryRedirect:
    """Send ``device='cuda'`` factory calls to CPU while CUDA is unavailable.

    ``PositionEmbeddingSine`` and the detector's relative-position cache
    allocate their precomputed tables with ``device='cuda'`` at construction.
    The values do not depend on the device. The redirect is only active around
    construction, and every redirected call is recorded.
    """

    _FACTORIES = (
        "zeros",
        "ones",
        "empty",
        "full",
        "arange",
        "tensor",
        "eye",
        "rand",
        "randn",
    )

    def __init__(self) -> None:
        self.redirects: list[str] = []
        self._saved: dict[str, object] = {}

    def __enter__(self) -> "CudaFactoryRedirect":
        for name in self._FACTORIES:
            original = getattr(torch, name)
            self._saved[name] = original

            def wrapped(*args, _orig=original, _name=name, _self=self, **kwargs):
                device = kwargs.get("device", None)
                if device is not None and not torch.cuda.is_available():
                    text = str(device)
                    if text == "cuda" or text.startswith("cuda:"):
                        kwargs = dict(kwargs)
                        kwargs["device"] = "cpu"
                        _self.redirects.append(f"torch.{_name} device={text}")
                return _orig(*args, **kwargs)

            setattr(torch, name, wrapped)
        return self

    def __exit__(self, *exc) -> None:
        for name, original in self._saved.items():
            setattr(torch, name, original)
        self._saved.clear()


# geometry_encoders.py and the tracker call Tensor.pin_memory() before a
# host-to-device copy. On a CPU build with no accelerator that raises.
# The patch returns the same CPU tensor. The call count is recorded.
PIN_MEMORY_CALLS = 0


class IdentityPinMemory:
    def __enter__(self) -> "IdentityPinMemory":
        global PIN_MEMORY_CALLS
        self._orig = torch.Tensor.pin_memory

        def _stay(tensor, *args, **kwargs):
            global PIN_MEMORY_CALLS
            PIN_MEMORY_CALLS += 1
            return tensor

        torch.Tensor.pin_memory = _stay  # type: ignore[method-assign]
        return self

    def __exit__(self, *exc) -> None:
        torch.Tensor.pin_memory = self._orig  # type: ignore[method-assign]


class IdentityCuda:
    """Make ``Tensor.cuda()`` return the same CPU tensor.

    Used only for the tracker's second frame, after the unpatched call has
    already been recorded as a failure. The official code moves memory
    features with ``.cuda()``; on this build that raises. The compute that
    follows is unchanged if the tensor is already on CPU.
    """

    def __enter__(self) -> "IdentityCuda":
        self._orig = torch.Tensor.cuda

        def _stay(tensor, *args, **kwargs):
            return tensor

        torch.Tensor.cuda = _stay  # type: ignore[method-assign]
        return self

    def __exit__(self, *exc) -> None:
        torch.Tensor.cuda = self._orig  # type: ignore[method-assign]


def _shapes_of(obj, found: list[tuple[tuple[int, ...], str]]) -> None:
    if isinstance(obj, torch.Tensor):
        found.append((tuple(int(s) for s in obj.shape), str(obj.dtype).replace("torch.", "")))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _shapes_of(item, found)
    elif isinstance(obj, dict):
        for item in obj.values():
            _shapes_of(item, found)


def _touched_bytes(obj, seen: set[int]) -> int:
    total = 0
    if isinstance(obj, torch.Tensor):
        try:
            ptr = obj.untyped_storage().data_ptr()
        except Exception:
            ptr = 0
        nbytes = obj.numel() * obj.element_size()
        if ptr and ptr not in seen:
            seen.add(ptr)
            total += nbytes
        elif ptr == 0:
            total += nbytes
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            total += _touched_bytes(item, seen)
    elif isinstance(obj, dict):
        for item in obj.values():
            total += _touched_bytes(item, seen)
    return total


class CallTracer(TorchDispatchMode):
    """Aggregate aten ops by stage, module, shapes, and dtype."""

    def __init__(self) -> None:
        super().__init__()
        self.stage = ""
        self.stack: list[str] = []
        self.rows: dict[tuple, dict] = {}
        self.meta_counts: collections.Counter[str] = collections.Counter()
        self._busy = False

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if self._busy:
            return func(*args, **kwargs)
        out = func(*args, **kwargs)
        self._busy = True
        try:
            self._record(func, args, kwargs, out)
        finally:
            self._busy = False
        return out

    def _record(self, func, args, kwargs, out) -> None:
        name = _op_name(func)
        if name in _META_OPS or name.startswith("aten.sym_"):
            self.meta_counts[name] += 1
            return
        # Shape and byte queries must not re-enter dispatch.
        with torch._C._DisableTorchDispatch():
            inputs: list[tuple[tuple[int, ...], str]] = []
            outputs: list[tuple[tuple[int, ...], str]] = []
            _shapes_of(args, inputs)
            _shapes_of(kwargs, inputs)
            _shapes_of(out, outputs)
            seen: set[int] = set()
            nbytes = _touched_bytes(args, seen) + _touched_bytes(kwargs, seen)
            nbytes += _touched_bytes(out, seen)
        transposed = False
        if name == "aten.convolution" and len(args) >= 7:
            transposed = bool(args[6])
        use_gelu = False
        if name == "aten._addmm_activation":
            use_gelu = bool(kwargs.get("use_gelu", False))
        in_shapes = tuple(shape for shape, _ in inputs[:6])
        out_shapes = tuple(shape for shape, _ in outputs[:4])
        dtypes = tuple(sorted({dtype for _, dtype in inputs[:6] + outputs[:4]}))
        module = self.stack[-1] if self.stack else ""
        key = (self.stage, module, name, dtypes, in_shapes, out_shapes, transposed, use_gelu)
        row = self.rows.get(key)
        if row is None:
            row = {"count": 0, "bytes": 0}
            self.rows[key] = row
        row["count"] += 1
        row["bytes"] += nbytes

    def records(self) -> list[dict]:
        found = []
        for key, row in self.rows.items():
            stage, module, name, dtypes, in_shapes, out_shapes, transposed, use_gelu = key
            found.append(
                {
                    "stage": stage,
                    "module": module,
                    "op": name,
                    "dtypes": list(dtypes),
                    "in_shapes": [list(s) for s in in_shapes],
                    "out_shapes": [list(s) for s in out_shapes],
                    "transposed": transposed,
                    "use_gelu": use_gelu,
                    "count": row["count"],
                    "bytes": row["bytes"],
                }
            )
        return found


def _op_name(func) -> str:
    text = str(func)
    parts = text.split(".")
    if len(parts) >= 3 and parts[0] in {"aten", "prim"}:
        return ".".join(parts[:2])
    return text


def install_module_stack(model: torch.nn.Module, tracer: CallTracer):
    hooks = []
    for name, module in model.named_modules():

        def pre(_module, _inputs, name=name):
            tracer.stack.append(name)
            if re.search(r"blocks\.\d+$", name):
                log(f"enter {name}")

        def post(_module, _inputs, _output, name=name):
            if tracer.stack and tracer.stack[-1] == name:
                tracer.stack.pop()
            elif name in tracer.stack:
                tracer.stack.remove(name)

        hooks.append(module.register_forward_pre_hook(pre))
        hooks.append(module.register_forward_hook(post))
    return hooks


def remove_hooks(hooks) -> None:
    for hook in hooks:
        hook.remove()


_MATMUL_OPS = {
    "aten.mm",
    "aten.addmm",
    "aten.bmm",
    "aten.matmul",
    "aten.linear",
    "aten.baddbmm",
    "aten.addbmm",
    "aten._addmm_activation",
    "aten._scaled_mm",
}

# FlopCounterMode has no formula for this op, and on this CPU build it does not
# decompose, so the counter total omits it. The matmul formula below fills that gap.
_COUNTER_MISS = {"aten._addmm_activation"}


def family_of(op: str) -> str:
    if "scaled_dot_product" in op or "flash_attention" in op or "efficient_attention" in op:
        return "attention"
    # Check upsample before matmul: bilinear contains the substring "linear".
    if any(token in op for token in ("upsample", "interpolate", "grid_sampler")):
        return "upsample"
    if op in _MATMUL_OPS:
        return "matmul"
    if "conv" in op:
        return "conv"
    if any(token in op for token in ("layer_norm", "group_norm", "batch_norm", "rms_norm")):
        return "norm"
    if "softmax" in op or "log_softmax" in op:
        return "softmax"
    if any(token in op for token in ("gelu", "silu", "relu", "sigmoid", "glu", "tanh")):
        return "activation"
    if "pool" in op:
        return "pool"
    if "embedding" in op:
        return "embedding"
    if any(token in op for token in ("index", "gather", "scatter", "nonzero")):
        return "index"
    if any(
        token in op
        for token in (
            "view",
            "reshape",
            "permute",
            "transpose",
            "slice",
            "select",
            "unsqueeze",
            "squeeze",
            "expand",
            "as_strided",
            "alias",
        )
    ):
        return "view"
    if any(token in op for token in ("clone", "_to_copy", "copy_", "cat", "stack")):
        return "memory"
    if any(
        token in op
        for token in (
            "add",
            "mul",
            "sub",
            "div",
            "where",
            "clamp",
            "exp",
            "log",
            "sin",
            "cos",
            "pow",
            "rsqrt",
            "sqrt",
            "neg",
            "abs",
            "reciprocal",
        )
    ):
        return "elementwise"
    return "other"


def region_of(module: str, stage: str) -> str:
    """Map a module name onto a SAM 3 region.

    FlopCounterMode names modules by class (``ViT.blocks.0``). The call tracer
    uses the registered path (``backbone.vision_backbone.trunk.blocks.0``).
    Both forms are recognized.
    """
    name = module.lower()
    if stage.startswith("tracker"):
        if any(
            token in name
            for token in ("maskdecoder", "sam_mask_decoder", "prompt_encoder", "sam_prompt")
        ):
            return "tracker_mask_decoder"
        if any(
            token in name
            for token in ("simplemaskencoder", "maskmem", "mask_down", "fuser", "cxblock")
        ):
            return "tracker_memory_encoder"
        if "transformer" in name:
            return "tracker_memory_attention"
        return "tracker_other"
    if any(token in name for token in ("language_backbone", "text_encoder", "vetextencoder")):
        return "text_encoder"
    if name == "vit" or name.startswith("vit.") or "trunk" in name or "patch_embed" in name:
        return "vision_encoder"
    if any(
        token in name
        for token in (
            "universalsegmentationhead",
            "segmentation_head",
            "pixel_decoder",
            "mask_predictor",
        )
    ):
        return "mask_head"
    if "geometry_encoder" in name or name.startswith("geometry"):
        return "geometry_encoder"
    if "dot_prod" in name or "dotprod" in name:
        return "scoring"
    if "transformerencoderfusion" in name or ".encoder" in name:
        return "detector_encoder"
    if "transformerdecoder" in name or ".decoder" in name or name.endswith("decoder"):
        return "detector_decoder"
    if stage == "image_backbone" and (name == "sequential" or name.startswith("sequential.")):
        return "vision_neck"
    if any(token in name for token in ("convs", "neck", "position_encoding")):
        return "vision_neck"
    return "other"


def _numel(shape: list[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def formula_flops(record: dict) -> int | None:
    """Apply the FlopCounterMode formulas to one aggregated call record.

    Returns None when the formula does not apply. Multiply-adds count as 2.
    """
    op = record["op"]
    inputs = record["in_shapes"]
    outputs = record["out_shapes"]
    count = record["count"]
    try:
        if op in {"aten.mm", "aten.matmul"} and len(inputs) >= 2 and len(inputs[0]) == 2:
            return count * mm_flop(inputs[0], inputs[1])
        if op in {"aten.addmm", "aten._addmm_activation"} and len(inputs) >= 3:
            return count * mm_flop(inputs[1], inputs[2])
        if op in {"aten.bmm", "aten.baddbmm"} and len(inputs) >= 2:
            mats = inputs[1:3] if op == "aten.baddbmm" and len(inputs) >= 3 else inputs[:2]
            return count * bmm_flop(mats[0], mats[1])
        if op == "aten.linear" and len(inputs) >= 2 and len(inputs[1]) == 2:
            batch = inputs[0][:-1]
            in_features = inputs[0][-1]
            out_features, weight_in = inputs[1]
            if in_features != weight_in:
                return None
            rows = _numel(list(batch)) if batch else 1
            return count * rows * out_features * 2 * in_features
        if "scaled_dot_product" in op and len(inputs) >= 3 and all(len(s) == 4 for s in inputs[:3]):
            return count * sdpa_flop_count(inputs[0], inputs[1], inputs[2])
        if op == "aten.convolution" and len(inputs) >= 2 and outputs:
            return count * conv_flop_count(
                inputs[0],
                inputs[1],
                outputs[0],
                transposed=bool(record.get("transposed")),
            )
    except Exception:
        return None
    return None


def extra_flops(record: dict) -> int | None:
    """Rough FLOPs for ops FlopCounterMode does not count.

    The formulas are documented estimates, not measurements:
    layer/group norm 5 per element, GELU/SiLU 8, softmax 5, pointwise 1,
    nearest upsample 0, bilinear upsample 4 per output element.
    """
    op = record["op"]
    outputs = record["out_shapes"]
    if not outputs:
        return None
    elements = record["count"] * _numel(outputs[0])
    if op == "aten._addmm_activation":
        # GELU or ReLU is inside the fused kernel, so it is not a separate aten op.
        return (8 if record.get("use_gelu") else 1) * elements
    if any(token in op for token in ("layer_norm", "group_norm", "batch_norm", "rms_norm")):
        return 5 * elements
    if "gelu" in op or "silu" in op:
        return 8 * elements
    if "softmax" in op:
        return 5 * elements
    if "relu" in op or "sigmoid" in op:
        return elements
    if op in {"aten.add", "aten.mul", "aten.sub", "aten.div"} or op.endswith(
        (".add", ".mul", ".sub", ".div")
    ):
        return elements
    if "upsample" in op or "interpolate" in op:
        mode_is_linear = False
        return 4 * elements if mode_is_linear else 0
    return None


def flop_counts_to_json(counts: dict) -> dict:
    exported = {}
    for module, ops in counts.items():
        exported[str(module)] = {str(op): int(value) for op, value in ops.items()}
    return exported


def leaf_modules(module_flops: dict[str, int]) -> dict[str, int]:
    names = [name for name in module_flops if name != "Global"]
    leaves = {}
    for name in names:
        if any(other != name and other.startswith(name + ".") for other in names):
            continue
        leaves[name] = module_flops[name]
    return leaves


def scan_cuda_only(sam3_root: Path) -> dict:
    """Static scan of the checkout. This is source evidence, not a runtime trace."""
    patterns = {
        "triton_import": re.compile(r"^\s*(import triton|from triton)\b", re.M),
        "flash_attn": re.compile(r"flash_attn"),
        "float8": re.compile(r"float8"),
        "tensor_cuda": re.compile(r"\.cuda\("),
        "device_cuda": re.compile(r"device\s*=\s*[\"']cuda"),
        "autocast_cuda": re.compile(r"autocast\(\s*device_type\s*=\s*[\"']cuda"),
        "torch_compile": re.compile(r"torch\.compile\("),
        "bfloat16": re.compile(r"bfloat16"),
        "cc_torch": re.compile(r"cc_torch"),
    }
    hits: dict[str, list[str]] = {key: [] for key in patterns}
    root = sam3_root / "sam3"
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(errors="replace")
        rel = str(path.relative_to(sam3_root))
        for key, pattern in patterns.items():
            count = len(pattern.findall(text))
            if count:
                hits[key].append(f"{rel}:{count}")
    # Snippets the inventory treats as blockers. The scan confirms they exist.
    expected = {
        "position_encoding_cuda_alloc": (
            "sam3/model/position_encoding.py",
            'device="cuda"',
        ),
        "decoder_coord_cache_cuda": ("sam3/model/decoder.py", 'device="cuda"'),
        "decoder_ffn_cuda_autocast": (
            "sam3/model/decoder.py",
            'torch.amp.autocast(device_type="cuda"',
        ),
        "tracker_maskmem_cuda": ("sam3/model/sam3_tracker_base.py", ".cuda("),
        "geometry_encoder_pin_memory": ("sam3/model/geometry_encoders.py", ".pin_memory("),
        "tracker_relpos_pin_memory": ("sam3/model/sam3_tracker_base.py", ".pin_memory("),
        "tracker_predictor_bf16_autocast": (
            "sam3/model/sam3_tracking_predictor.py",
            'torch.autocast(device_type="cuda", dtype=torch.bfloat16)',
        ),
        "video_inference_bf16_autocast": (
            "sam3/model/sam3_video_inference.py",
            'device_type="cuda", dtype=torch.bfloat16',
        ),
        "edt_triton_import": ("sam3/model/edt.py", "import triton"),
        "edt_imported_by_tracker_utils": (
            "sam3/model/sam3_tracker_utils.py",
            "from sam3.model.edt import edt_triton",
        ),
        "fa3_float8": ("sam3/perflib/fa3.py", "float8_e4m3fn"),
        "fused_bf16_addmm": ("sam3/perflib/fused.py", "torch.bfloat16"),
        "triton_nms": ("sam3/perflib/triton/nms.py", "import triton"),
        "triton_connected_components": (
            "sam3/perflib/triton/connected_components.py",
            "import triton",
        ),
        "training_triton_focal_loss": (
            "sam3/train/loss/sigmoid_focal_loss.py",
            "import triton",
        ),
    }
    confirmed = {}
    missing = []
    for key, (rel, snippet) in expected.items():
        path = sam3_root / rel
        text = path.read_text(errors="replace") if path.is_file() else ""
        ok = snippet in text
        confirmed[key] = {"path": rel, "snippet": snippet, "present": ok}
        if not ok:
            missing.append(key)
    return {"pattern_hits": hits, "confirmed_snippets": confirmed, "missing_expected": missing}


def trace_stage(tracer: CallTracer, stage: str, fn):
    tracer.stage = stage
    tracer.stack.clear()
    counter = FlopCounterMode(display=False)
    started = time.perf_counter()
    # Parameters stay fp32. CPU bfloat16 autocast matches the official CUDA
    # autocast closely enough for the fused bf16 MLP kernel and fp32 geometric
    # prompts to meet. See metadata.dtype_reason.
    with (
        tracer,
        counter,
        torch.autocast(device_type="cpu", dtype=torch.bfloat16),
        IdentityPinMemory(),
    ):
        result = fn()
    elapsed = time.perf_counter() - started
    log(f"stage {stage} done in {elapsed:.1f}s  counter_flops={counter.get_total_flops()}")
    return (
        result,
        flop_counts_to_json(counter.get_flop_counts()),
        dict(counter.get_unsupported_ops()),
        elapsed,
    )


def build_image_model(redirect: CudaFactoryRedirect):
    from sam3.model_builder import build_sam3_image_model

    with redirect:
        model = build_sam3_image_model(
            device="cpu",
            eval_mode=True,
            load_from_HF=False,
            compile=False,
            enable_inst_interactivity=False,
        )
    model.eval()
    return model


def build_tracker(redirect: CudaFactoryRedirect):
    from sam3.model_builder import build_tracker

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with redirect:
            tracker = build_tracker(apply_temporal_disambiguation=False, with_backbone=False)
    tracker.eval()
    notes = [f"{item.category.__name__}: {item.message}" for item in caught]
    return tracker, notes


def run_image(model, tracer: CallTracer, prompt: str):
    import numpy as np
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor

    processor = Sam3Processor(model, resolution=IMAGE_SIZE, device="cpu", confidence_threshold=0.5)
    image = Image.fromarray(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8))
    # Same resize and normalize as Sam3Processor.set_image. The tensor stays
    # float32; the stage runs under CPU bfloat16 autocast.
    from torchvision.transforms import v2

    normalized = processor.transform(v2.functional.to_image(image)).unsqueeze(0)

    def backbone_forward():
        state = {
            "original_height": IMAGE_SIZE,
            "original_width": IMAGE_SIZE,
            "backbone_out": model.backbone.forward_image(normalized),
        }
        return state

    state, flops_image, unsupported_image, seconds_image = trace_stage(
        tracer, "image_backbone", backbone_forward
    )
    # Split the official set_text_prompt body so the text encoder and the
    # detector are separate stages. The calls are the same ones set_text_prompt makes.
    unsupported = {}
    seconds = {"image_backbone": seconds_image}

    def run_text():
        outputs = model.backbone.forward_text([prompt], device="cpu")
        state["backbone_out"].update(outputs)
        state["geometric_prompt"] = model._get_dummy_prompt()
        return outputs

    _, text_flops, unsupported_text, seconds_text = trace_stage(tracer, "image_text", run_text)
    _, detector_flops, unsupported_det, seconds_det = trace_stage(
        tracer, "image_detector", lambda: processor._forward_grounding(state)
    )
    seconds["image_text"] = seconds_text
    seconds["image_detector"] = seconds_det
    unsupported["image_backbone"] = unsupported_image
    unsupported["image_text"] = unsupported_text
    unsupported["image_detector"] = unsupported_det
    return {
        "state": state,
        "flops": {
            "image_backbone": flops_image,
            "image_text": text_flops,
            "image_detector": detector_flops,
        },
        "unsupported": unsupported,
        "seconds": seconds,
    }


def fpn_for_tracker(state) -> dict:
    backbone = state["backbone_out"]
    features = backbone["backbone_fpn"]
    positions = backbone["vision_pos_enc"]
    levels = min(3, len(features))
    chosen_f = features[-levels:]
    chosen_p = positions[-levels:]
    feat_sizes = [(int(t.shape[-2]), int(t.shape[-1])) for t in chosen_p]
    vision_feats = [t.flatten(2).permute(2, 0, 1).contiguous() for t in chosen_f]
    vision_pos = [t.flatten(2).permute(2, 0, 1).contiguous() for t in chosen_p]
    return {
        "vision_feats": vision_feats,
        "vision_pos": vision_pos,
        "feat_sizes": feat_sizes,
        "level_shapes_nchw": [list(t.shape) for t in chosen_f],
    }


def run_tracker(tracker, feats: dict, tracer: CallTracer) -> dict:
    point_inputs = {
        "point_coords": torch.tensor([[[IMAGE_SIZE / 2, IMAGE_SIZE / 2]]], dtype=torch.float32),
        "point_labels": torch.tensor([[1]], dtype=torch.int32),
    }
    image = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    common = dict(
        current_vision_feats=feats["vision_feats"],
        current_vision_pos_embeds=feats["vision_pos"],
        feat_sizes=feats["feat_sizes"],
        image=image,
        point_inputs=point_inputs,
        mask_inputs=None,
        output_dict=output_dict,
        num_frames=2,
        run_mem_encoder=True,
    )

    def project_high_res():
        # Sam3TrackerBase.forward_image runs these 1x1 convs before tracking.
        # The raw FPN is 256 channels; conv_s0/conv_s1 match the mask decoder's
        # upscaling residuals (transformer_dim//8 and //4).
        if feats.get("high_res_projected"):
            return
        projected = []
        convs = (
            tracker.sam_mask_decoder.conv_s0,
            tracker.sam_mask_decoder.conv_s1,
        )
        for index, (feat, size) in enumerate(zip(feats["vision_feats"], feats["feat_sizes"])):
            if index >= len(convs):
                projected.append(feat)
                continue
            batch, channels = feat.shape[1], feat.shape[2]
            height, width = size
            nchw = feat.permute(1, 2, 0).reshape(batch, channels, height, width)
            nchw = convs[index](nchw)
            projected.append(nchw.flatten(2).permute(2, 0, 1).contiguous())
        feats["vision_feats"] = projected
        feats["high_res_projected"] = True

    def cloned_common():
        return dict(
            common,
            current_vision_feats=[t.clone() for t in feats["vision_feats"]],
            current_vision_pos_embeds=[t.clone() for t in feats["vision_pos"]],
        )

    init_out, flops_init, unsupported_init, seconds_init = trace_stage(
        tracer,
        "tracker_init",
        lambda: (
            project_high_res(),
            tracker.track_step(frame_idx=0, is_init_cond_frame=True, **cloned_common()),
        )[1],
    )
    output_dict["cond_frame_outputs"][0] = init_out
    unpatched = None
    try:
        # pin_memory is patched so this attempt reaches the hardcoded .cuda() call.
        with IdentityPinMemory():
            tracker.track_step(frame_idx=1, is_init_cond_frame=False, **cloned_common())
        unpatched = {"raised": False}
    except Exception as exc:
        unpatched = {
            "raised": True,
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(limit=12),
        }
        log(f"tracker frame 2 unpatched failed: {type(exc).__name__}: {exc}")
    # The failure happens before memory attention. Re-run with .cuda() as identity
    # so the compute ops after that device move are recorded.
    with IdentityCuda():
        _frame2, flops_mem, unsupported_mem, seconds_mem = trace_stage(
            tracer,
            "tracker_memory",
            lambda: tracker.track_step(frame_idx=1, is_init_cond_frame=False, **cloned_common()),
        )
    return {
        "flops": {"tracker_init": flops_init, "tracker_memory": flops_mem},
        "unsupported": {"tracker_init": unsupported_init, "tracker_memory": unsupported_mem},
        "seconds": {"tracker_init": seconds_init, "tracker_memory": seconds_mem},
        "unpatched_frame2": unpatched,
        "feat_sizes": feats["feat_sizes"],
        "level_shapes_nchw": feats["level_shapes_nchw"],
    }


def global_flops(stage_flops: dict) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    """Sum Global rows across stages. Module rows are not summed (they repeat Global)."""
    by_op: collections.Counter[str] = collections.Counter()
    by_module: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for stage, counts in stage_flops.items():
        for module, ops in counts.items():
            for op, value in ops.items():
                by_module[f"{stage}:{module}"][op] += int(value)
                if module == "Global":
                    by_op[op] += int(value)
    return dict(by_op), {key: dict(value) for key, value in by_module.items()}


def summarize(records: list[dict], stage_flops: dict) -> dict:
    counter_by_op, counter_by_module = global_flops(stage_flops)
    counter_total = sum(counter_by_op.values())

    module_totals: collections.Counter[str] = collections.Counter()
    for key, ops in counter_by_module.items():
        module = key.split(":", 1)[1]
        if module == "Global":
            continue
        module_totals[key] += sum(ops.values())
    # Leaf selection is per stage, because the same relative name can recur.
    per_stage_modules: dict[str, dict[str, int]] = collections.defaultdict(dict)
    for key, total in module_totals.items():
        stage, module = key.split(":", 1)
        per_stage_modules[stage][module] = total
    region_counter = collections.Counter()
    leaf_rows = []
    for stage, modules in per_stage_modules.items():
        for module, total in leaf_modules(modules).items():
            region = region_of(module, stage)
            region_counter[f"{stage}:{region}"] += total
            leaf_rows.append(
                {"stage": stage, "module": module, "region": region, "counter_flops": total}
            )

    families = collections.defaultdict(
        lambda: {
            "calls": 0,
            "bytes": 0,
            "formula_flops": 0,
            "extra_flops": 0,
            "ops": collections.Counter(),
        }
    )
    regions_bytes = collections.Counter()
    regions_calls = collections.Counter()
    region_formula: collections.Counter[str] = collections.Counter()
    shape_rows = []
    for record in records:
        family = family_of(record["op"])
        region = region_of(record["module"], record["stage"])
        bucket = families[family]
        bucket["calls"] += record["count"]
        bucket["bytes"] += record["bytes"]
        bucket["ops"][record["op"]] += record["count"]
        formula = formula_flops(record)
        extra = extra_flops(record)
        if formula:
            bucket["formula_flops"] += formula
        if extra:
            bucket["extra_flops"] += extra
        if formula:
            region_formula[f"{record['stage']}:{region}"] += formula
        regions_bytes[f"{record['stage']}:{region}"] += record["bytes"]
        regions_calls[f"{record['stage']}:{region}"] += record["count"]
        shape_flops = formula or 0
        if shape_flops or record["bytes"] > 0:
            shape_rows.append(
                {
                    "stage": record["stage"],
                    "region": region,
                    "family": family,
                    "op": record["op"],
                    "dtypes": ",".join(record["dtypes"]),
                    "in_shapes": record["in_shapes"],
                    "out_shapes": record["out_shapes"],
                    "count": record["count"],
                    "formula_flops": formula or 0,
                    "extra_flops": extra or 0,
                    "bytes": record["bytes"],
                    "module": record["module"],
                }
            )

    counter_by_family: collections.Counter[str] = collections.Counter()
    for op, value in counter_by_op.items():
        counter_by_family[family_of(str(op))] += int(value)
    missed_by_family: collections.Counter[str] = collections.Counter()
    for record in records:
        if record["op"] not in _COUNTER_MISS:
            continue
        missed = formula_flops(record) or 0
        missed_by_family[family_of(record["op"])] += missed

    family_rows = []
    for name, bucket in families.items():
        counter_flops = int(counter_by_family[name])
        missed_flops = int(missed_by_family[name])
        family_rows.append(
            {
                "family": name,
                "calls": bucket["calls"],
                "bytes": bucket["bytes"],
                "counter_flops": counter_flops,
                "missed_by_counter_flops": missed_flops,
                "rank_flops": counter_flops + missed_flops,
                "formula_flops": bucket["formula_flops"],
                "extra_flops": bucket["extra_flops"],
                "top_ops": bucket["ops"].most_common(8),
            }
        )
    family_rows.sort(
        key=lambda row: (row["rank_flops"], row["extra_flops"], row["bytes"]), reverse=True
    )

    bytes_total = sum(row["bytes"] for row in family_rows)
    rank_total = sum(row["rank_flops"] for row in family_rows)
    for row in family_rows:
        row["bytes_share"] = (row["bytes"] / bytes_total) if bytes_total else 0.0
        row["rank_share"] = (row["rank_flops"] / rank_total) if rank_total else 0.0

    grouped: dict[tuple, dict] = {}
    for row in shape_rows:
        key = (
            row["stage"],
            row["region"],
            row["family"],
            row["op"],
            row["dtypes"],
            json.dumps(row["in_shapes"]),
            json.dumps(row["out_shapes"]),
        )
        slot = grouped.get(key)
        if slot is None:
            slot = dict(row)
            slot["modules"] = 1
            grouped[key] = slot
        else:
            slot["count"] += row["count"]
            slot["formula_flops"] += row["formula_flops"]
            slot["extra_flops"] += row["extra_flops"]
            slot["bytes"] += row["bytes"]
            slot["modules"] += 1
    grouped_rows = sorted(
        grouped.values(), key=lambda row: (row["formula_flops"], row["bytes"]), reverse=True
    )
    shape_rows.sort(key=lambda row: (row["formula_flops"], row["bytes"]), reverse=True)
    leaf_rows.sort(key=lambda row: row["counter_flops"], reverse=True)
    return {
        "counter_total_flops": counter_total,
        "counter_omits_addmm_activation": True,
        "rank_total_flops": rank_total,
        "counter_by_op": counter_by_op,
        "bytes_total": bytes_total,
        "families": family_rows,
        "regions_rank_flops": [
            {"stage_region": key, "rank_flops": value}
            for key, value in region_formula.most_common()
        ],
        "regions_bytes": [
            {"stage_region": key, "bytes": value, "calls": regions_calls[key]}
            for key, value in regions_bytes.most_common()
        ],
        "top_shapes": shape_rows[:40],
        "grouped_shapes": grouped_rows[:40],
        "top_leaf_modules": leaf_rows[:40],
        "formula_flops_total": sum(row["formula_flops"] for row in family_rows),
        "extra_flops_total": sum(row["extra_flops"] for row in family_rows),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            for key, value in list(flat.items()):
                if isinstance(value, (list, dict)):
                    flat[key] = json.dumps(value)
            writer.writerow(flat)


def git_sha(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def self_check() -> None:
    tracer = CallTracer()

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(8, 8)
            self.norm = torch.nn.LayerNorm(8)
            self.conv = torch.nn.Conv2d(1, 2, 3, padding=1)

        def forward(self, x, image):
            y = torch.nn.functional.gelu(self.norm(self.lin(x)))
            q = y.view(1, 2, 2, 4)
            attended = torch.nn.functional.scaled_dot_product_attention(q, q, q)
            return self.conv(image) + attended.sum()

    module = Tiny().eval()
    hooks = install_module_stack(module, tracer)
    x = torch.randn(2, 8)
    image = torch.randn(1, 1, 4, 4)
    with torch.inference_mode():
        _out, flops, _unsupported, _seconds = trace_stage(
            tracer, "self_check", lambda: module(x, image)
        )
    remove_hooks(hooks)
    summary = summarize(tracer.records(), {"self_check": flops})
    families = {row["family"] for row in summary["families"]}
    missing = {"matmul", "attention", "conv", "norm", "activation"} - families
    if missing:
        raise SystemExit(f"self-check missing families: {sorted(missing)}")
    if summary["counter_total_flops"] <= 0:
        raise SystemExit("self-check counter recorded no FLOPs")
    ratio = summary["formula_flops_total"] / summary["counter_total_flops"]
    if not 0.5 <= ratio <= 1.5:
        raise SystemExit(
            f"self-check formula/counter ratio {ratio:.3f} is outside 0.5..1.5 "
            f"(formula={summary['formula_flops_total']} counter={summary['counter_total_flops']})"
        )
    log(f"self-check ok families={sorted(families)} ratio={ratio:.3f}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam3-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--prompt", default="a red cube")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--skip-tracker", action="store_true")
    parser.add_argument(
        "--resummarize",
        action="store_true",
        help="Rebuild summary.csv/json from an existing calls.json and flops_by_stage.json",
    )
    return parser.parse_args()


NOTES = [
    "rank_flops is FlopCounterMode plus the matmul formula for aten._addmm_activation. That op has no counter formula and does not decompose on this CPU build, so the counter total omits the ViT MLP's fused linear+GELU.",
    "Multiply-adds count as 2. The counter also omits layer norm, GELU, softmax, elementwise ops, and the softmax inside SDPA. extra_flops estimates those (5 per norm element, 8 per GELU, 5 per softmax, 1 per pointwise op). The fused GELU inside _addmm_activation is in extra_flops, not in rank_flops.",
    "formula_flops applies the counter's matmul, convolution, and SDPA formulas to the aten ops recorded by the tracer.",
    "bytes count distinct input and output storages at the aten boundary. Views are included, so the view family's byte share is not traffic. aten._to_copy is mostly the bfloat16 autocast casting float32 weights. The SDPA score matrix is not a separate tensor on this CPU kernel.",
    "Compute dtype is bfloat16 under CPU autocast. Parameters stay float32. See metadata.dtype_reason.",
    "Wall times include the Python dispatcher and are not a model benchmark.",
    "No MPS or MLX kernel was executed.",
]


def write_reports(out_dir: Path, summary: dict) -> None:
    write_csv(
        out_dir / "families.csv",
        summary["families"],
        [
            "family",
            "calls",
            "rank_flops",
            "rank_share",
            "counter_flops",
            "missed_by_counter_flops",
            "formula_flops",
            "extra_flops",
            "bytes",
            "bytes_share",
            "top_ops",
        ],
    )
    write_csv(
        out_dir / "shapes.csv",
        summary["grouped_shapes"],
        [
            "stage",
            "region",
            "family",
            "op",
            "dtypes",
            "in_shapes",
            "out_shapes",
            "count",
            "modules",
            "formula_flops",
            "extra_flops",
            "bytes",
        ],
    )
    write_csv(
        out_dir / "regions.csv",
        summary["regions_rank_flops"],
        ["stage_region", "rank_flops"],
    )


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", message="FlopCounterMode does not have")
    warnings.filterwarnings("ignore", message="torch.jit.script")
    warnings.filterwarnings("ignore", message="CUDA is not available")
    torch.set_num_threads(int(os.environ.get("SAM3_THREADS", "4")))
    torch.manual_seed(0)
    if args.self_check:
        self_check()
        return

    out_dir = args.output or (repo_root() / "tools" / "sam3_inventory" / "output")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.resummarize:
        payload = json.loads((out_dir / "summary.json").read_text())
        records = json.loads((out_dir / "calls.json").read_text())
        stage_flops = json.loads((out_dir / "flops_by_stage.json").read_text())
        summary = summarize(records, stage_flops)
        payload["summary"] = summary
        payload["metadata"]["notes"] = NOTES
        (out_dir / "summary.json").write_text(json.dumps(payload, indent=2))
        write_reports(out_dir, summary)
        log(
            f"resummarized {out_dir} rank_flops={summary['rank_total_flops']} "
            f"counter_flops={summary['counter_total_flops']}"
        )
        return

    import sam3

    sam3_root = args.sam3_root or Path(sam3.__file__).resolve().parents[1]
    sha = git_sha(sam3_root)
    if sha != SAM3_SHA:
        log(f"warning: sam3 checkout {sha} != pinned {SAM3_SHA}")
    scan = scan_cuda_only(sam3_root)
    (out_dir / "cuda_scan.json").write_text(json.dumps(scan, indent=2))
    if scan["missing_expected"]:
        log(f"warning: expected CUDA snippets missing: {scan['missing_expected']}")

    redirect = CudaFactoryRedirect()
    log("building image model")
    model = build_image_model(redirect)
    n_params = sum(p.numel() for p in model.parameters())
    log(
        f"image model params={n_params} compute_dtype={TRACE_DTYPE} "
        f"cuda_factory_redirects={len(redirect.redirects)}"
    )
    if args.build_only:
        names = [name for name, _ in model.named_modules()]
        print("module sample:")
        for name in names[:40]:
            print(" ", name)
        print("redirects", collections.Counter(redirect.redirects))
        return

    tracer = CallTracer()
    hooks = install_module_stack(model, tracer)
    log(f"tracing image prompt={args.prompt!r}")
    with torch.inference_mode():
        image_result = run_image(model, tracer, args.prompt)
    feats = fpn_for_tracker(image_result["state"])
    log(f"fpn levels {feats['level_shapes_nchw']}")
    remove_hooks(hooks)
    del image_result["state"]
    del model
    gc.collect()

    tracker_result = None
    tracker_notes: list[str] = []
    if not args.skip_tracker:
        log("building tracker without a second vision backbone")
        tracker, tracker_notes = build_tracker(redirect)
        hooks = install_module_stack(tracker, tracer)
        with torch.inference_mode():
            tracker_result = run_tracker(tracker, feats, tracer)
        remove_hooks(hooks)
        del tracker
        gc.collect()

    stage_flops = dict(image_result["flops"])
    seconds = dict(image_result["seconds"])
    unsupported = dict(image_result["unsupported"])
    if tracker_result:
        stage_flops.update(tracker_result["flops"])
        seconds.update(tracker_result["seconds"])
        unsupported.update(tracker_result["unsupported"])

    records = tracer.records()
    summary = summarize(records, stage_flops)
    metadata = {
        "sam3_sha": sha,
        "sam3_sha_pinned": SAM3_SHA,
        "torch": torch.__version__,
        "image_size": IMAGE_SIZE,
        "parameter_dtype": "float32",
        "compute_dtype": str(TRACE_DTYPE).replace("torch.", ""),
        "dtype_reason": (
            "Parameters stay float32. Each traced stage runs under "
            "torch.autocast(device_type='cpu', dtype=torch.bfloat16). "
            "sam3/model/vitdet.py Mlp.forward calls sam3/perflib/fused.py addmm_act, "
            "which casts that linear and its activation to bfloat16 regardless. "
            "Without autocast the next float32 linear raises "
            "'mat1 and mat2 must have the same dtype, but got BFloat16 and Float'. "
            "Casting every parameter to bfloat16 instead makes the CPU layer_norm "
            "kernel raise 'expect parameter to have scalar type of Float'. "
            "Official inference uses a CUDA bfloat16 autocast over float32 weights. "
            "This is the CPU equivalent, not a Metal run. Complex RoPE buffers stay complex."
        ),
        "prompt": args.prompt,
        "weights": "random initialization, load_from_HF=False",
        "device": "cpu",
        "cuda_available": torch.cuda.is_available(),
        "params_image": n_params,
        "cuda_factory_redirects": collections.Counter(redirect.redirects),
        "pin_memory_identity_calls": PIN_MEMORY_CALLS,
        "pin_memory_reason": (
            "geometry_encoders.py and sam3_tracker_base.py call Tensor.pin_memory() "
            "before .to(device). With no accelerator that raises "
            "'Cannot access accelerator device when none is available'. "
            "The trace returns the same CPU tensor from pin_memory so roi_align "
            "and the tracker still run. The call count is pin_memory_identity_calls."
        ),
        "tracker_build_warnings": tracker_notes,
        "tracker_frame2_unpatched": None
        if not tracker_result
        else tracker_result["unpatched_frame2"],
        "tracker_frame2_patch": (
            "Tensor.cuda patched to identity so the memory-conditioned step could run on CPU. "
            "The unpatched call is recorded separately and raises."
        ),
        "fpn_level_shapes_nchw": feats["level_shapes_nchw"],
        "seconds_including_dispatcher": seconds,
        "unsupported_ops": unsupported,
        "meta_op_counts": dict(tracer.meta_counts),
        "notes": NOTES,
    }
    payload = {"metadata": metadata, "summary": summary}
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    (out_dir / "calls.json").write_text(json.dumps(records))
    (out_dir / "flops_by_stage.json").write_text(json.dumps(stage_flops))
    write_reports(out_dir, summary)
    log(
        f"wrote {out_dir} counter_flops={summary['counter_total_flops']} "
        f"rank_flops={summary['rank_total_flops']} records={len(records)}"
    )


if __name__ == "__main__":
    main()
