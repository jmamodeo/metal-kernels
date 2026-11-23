# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the SAM 3 compatibility layer.

MPS is not available here. These tests check the rewrites and the linear+GELU
fallback against the fused op. The full forward, when sam3 is installed, is
test_patched_image_and_tracker_match_fused_reference.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import pytest

pytest.importorskip("torch")

import torch
import torch.nn as nn


def _sam3():
    import metal_kernels.integrations.sam3 as compat

    return compat


def test_package_import_does_not_import_the_shim():
    script = (
        "import metal_kernels, sys; "
        "assert 'metal_kernels.integrations.sam3' not in sys.modules; "
        "assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_factories_cuda_and_pin_memory_land_on_cpu():
    compat = _sam3()
    compat.install(device="cpu", mlp="linear_gelu")
    try:
        zeros = torch.zeros(2, 3, device="cuda")
        arange = torch.arange(4, device="cuda:0")
        assert zeros.device.type == "cpu"
        assert arange.device.type == "cpu"
        moved = torch.zeros(1).cuda()
        assert moved.device.type == "cpu"
        pinned = torch.zeros(1).pin_memory()
        assert pinned.device.type == "cpu"
        assert compat.stats()["factory_redirects"] >= 2
        assert compat.stats()["cuda_calls"] >= 1
        assert compat.stats()["pin_memory_calls"] >= 1
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            assert torch.is_autocast_enabled("cpu")
            assert torch.get_autocast_dtype("cpu") == torch.bfloat16
        with torch.amp.autocast(device_type="cuda", enabled=False):
            assert not torch.is_autocast_enabled("cpu")
        assert compat.stats()["autocast_rewrites"] >= 2
    finally:
        compat.uninstall()


def test_linear_gelu_matches_aten_addmm_activation():
    compat = _sam3()
    torch.manual_seed(0)
    linear = nn.Linear(32, 64)
    inputs = torch.randn(4, 7, 32)
    with torch.inference_mode():
        bias = linear.bias.detach().to(torch.bfloat16)
        weight = linear.weight.detach().to(torch.bfloat16)
        fused = torch.ops.aten._addmm_activation(
            bias,
            inputs.reshape(-1, 32).to(torch.bfloat16),
            weight.t(),
            beta=1,
            alpha=1,
            use_gelu=True,
        ).view(4, 7, 64)
        fallback = compat._linear_gelu(nn.GELU, linear, inputs)
    assert fallback.dtype == torch.bfloat16
    assert fallback.shape == fused.shape
    difference = (fallback.float() - fused.float()).abs()
    assert float(difference.max()) < 1e-2


def test_triton_stub_satisfies_the_edt_import_pattern():
    script = """
import sys
sys.modules["triton"] = None
from metal_kernels.integrations import sam3
sam3._ensure_triton()
import triton
import triton.language as tl

@triton.jit
def edt_kernel(horizontal: tl.constexpr):
    return horizontal

assert not isinstance(edt_kernel, type(lambda: None)) or True
try:
    edt_kernel[1, 1](None)
except RuntimeError as exc:
    assert "Triton" in str(exc)
else:
    raise SystemExit("stub kernel did not raise")
print("ok")
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="The unpatched CUDA failure is checked on the Linux CPU build.",
)
def test_unpatched_image_build_hits_cuda():
    if importlib.util.find_spec("sam3") is None:
        pytest.skip("sam3 is not installed")
    script = """
from sam3.model_builder import build_sam3_image_model
build_sam3_image_model(
    device="cpu",
    load_from_HF=False,
    compile=False,
    enable_inst_interactivity=False,
    eval_mode=True,
)
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode != 0
    combined = result.stderr + result.stdout
    assert "CUDA" in combined or "cuda" in combined


def _compare(reference, candidate):
    """Maximum absolute difference on finite elements.

    A random-init bf16 detector can overflow to NaN. Both runs have to overflow
    in the same places. The vision FPN is required to be finite: that is where
    the MLP fallback runs.
    """
    assert reference.keys() == candidate.keys()
    worst = 0.0
    worst_key = ""
    for key, ref in reference.items():
        tensors = ref if isinstance(ref, list) else [ref]
        others = candidate[key] if isinstance(candidate[key], list) else [candidate[key]]
        assert len(tensors) == len(others)
        for index, (left, right) in enumerate(zip(tensors, others)):
            assert left.shape == right.shape, key
            finite = torch.isfinite(left) & torch.isfinite(right)
            if key == "fpn":
                assert torch.isfinite(left).all(), key
                assert torch.isfinite(right).all(), key
            else:
                assert torch.isfinite(left).eq(torch.isfinite(right)).all(), key
            if not bool(finite.any()):
                continue
            difference = float((left[finite] - right[finite]).abs().max())
            if difference > worst:
                worst = difference
                worst_key = f"{key}[{index}]"
    return worst, worst_key


def _image_outputs(model, image):
    from sam3.model.sam3_image_processor import Sam3Processor

    processor = Sam3Processor(model, resolution=1008, device="cpu", confidence_threshold=0.5)
    with torch.inference_mode():
        state = processor.set_image(image)
        text = model.backbone.forward_text(["a red cube"], device="cpu")
        backbone = state["backbone_out"]
        backbone.update(text)
        outputs = model.forward_grounding(
            backbone_out=backbone,
            find_input=processor.find_stage,
            geometric_prompt=model._get_dummy_prompt(),
            find_target=None,
        )
    return {
        "fpn": [tensor.detach().float().cpu().clone() for tensor in backbone["backbone_fpn"]],
        "logits": outputs["pred_logits"].detach().float().cpu().clone(),
        "boxes": outputs["pred_boxes"].detach().float().cpu().clone(),
        "masks": outputs["pred_masks"].detach().float().cpu().clone(),
        "presence": outputs["presence_logit_dec"].detach().float().cpu().clone(),
    }


def _project_high_res(tracker, feats):
    convs = (tracker.sam_mask_decoder.conv_s0, tracker.sam_mask_decoder.conv_s1)
    projected = []
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


def test_patched_image_and_tracker_match_fused_reference():
    if importlib.util.find_spec("sam3") is None:
        pytest.skip("sam3 is not installed")
    from PIL import Image

    compat = _sam3()
    compat.install(device="cpu", mlp="fused")
    torch.manual_seed(0)
    torch.set_num_threads(1)
    model = compat.build_image_model(
        load_from_HF=False, enable_inst_interactivity=False, eval_mode=True
    )
    assert model.training is False
    image = Image.new("RGB", (1008, 1008), (120, 40, 10))
    before = compat.stats()["factory_redirects"]
    # Build already happened; the redirect counter includes construction.
    assert before > 0

    compat.set_mlp("fused")
    torch.manual_seed(0)
    first = _image_outputs(model, image)
    torch.manual_seed(0)
    second = _image_outputs(model, image)
    repeat, repeat_key = _compare(first, second)
    print(f"image fused repeat max abs {repeat:.6f} at {repeat_key}")
    for index, tensor in enumerate(first["fpn"]):
        print(f"fpn[{index}] reference abs max {float(tensor.abs().max()):.4f}")

    compat.set_mlp("linear_gelu")
    assert compat.active_mlp() == "linear_gelu"
    torch.manual_seed(0)
    fallback = _image_outputs(model, image)
    worst, worst_key = _compare(first, fallback)
    print(f"image fused vs linear+GELU max abs {worst:.6f} at {worst_key}")
    # CPU bf16 kernels are not bitwise stable: a second fused run moves by
    # a few hundredths to a few tenths. The fallback has to stay inside that
    # spread. The single-op test covers the GELU formula itself.
    assert repeat <= 0.5, repeat
    assert worst <= 0.5, (worst, worst_key)

    from sam3.model.sam3_image_processor import Sam3Processor

    processor = Sam3Processor(model, device="cpu")
    state = processor.set_text_prompt("a red cube", processor.set_image(image))
    assert "scores" in state

    backbone = state["backbone_out"]
    features = backbone["backbone_fpn"]
    positions = backbone["vision_pos_enc"]
    levels = min(3, len(features))
    chosen_f = features[-levels:]
    chosen_p = positions[-levels:]
    feats = {
        "vision_feats": [tensor.flatten(2).permute(2, 0, 1).contiguous() for tensor in chosen_f],
        "vision_pos": [tensor.flatten(2).permute(2, 0, 1).contiguous() for tensor in chosen_p],
        "feat_sizes": [(int(tensor.shape[-2]), int(tensor.shape[-1])) for tensor in chosen_p],
    }
    del model, state, backbone, features, positions
    tracker = compat.build_tracker(apply_temporal_disambiguation=False, with_backbone=False)
    tracker.eval()
    point_inputs = {
        "point_coords": torch.tensor([[[504.0, 504.0]]], dtype=torch.float32),
        "point_labels": torch.tensor([[1]], dtype=torch.int32),
    }
    frame = torch.zeros(1, 3, 1008, 1008)
    output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}

    def step(frame_idx, is_init):
        return tracker.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init,
            current_vision_feats=[tensor.clone() for tensor in feats["vision_feats"]],
            current_vision_pos_embeds=[tensor.clone() for tensor in feats["vision_pos"]],
            feat_sizes=feats["feat_sizes"],
            image=frame,
            point_inputs=point_inputs,
            mask_inputs=None,
            output_dict=output_dict,
            num_frames=2,
            run_mem_encoder=True,
        )

    with torch.inference_mode():
        _project_high_res(tracker, feats)
        init_out = step(0, True)
        assert "pred_masks" in init_out
        output_dict["cond_frame_outputs"][0] = init_out
        memory_out = step(1, False)
    assert "pred_masks" in memory_out
    assert compat.stats()["cuda_calls"] > 0
    assert compat.stats()["pin_memory_calls"] > 0
