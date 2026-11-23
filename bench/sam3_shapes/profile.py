# SPDX-License-Identifier: Apache-2.0
"""Time PyTorch MPS and MLX built-ins at the SAM 3 shapes from the op inventory.

This script is meant to be run on an Apple silicon Mac. It does not time
anything on Linux in a way that should be read as an
MPS result. ``--self-check`` runs one tiny linear on the CPU so the script
can be exercised where MPS is absent.

The shapes are the top of docs/sam3-op-inventory.md (SAM 3
31cb7e624e4fe95301b1be1d4e809111593d2e45, 1008px, bfloat16 autocast):

- ViT MLP fc1: linear+GELU, [5184, 1024] x [1024, 4736]
- ViT MLP fc2: [5184, 4736] x [4736, 1024]
- ViT QKV: [5184, 1024] x [1024, 3072]
- ViT attention projection: [5184, 1024] x [1024, 1024]
- Global SDPA: [1, 16, 5184, 64], 4 of these in a forward
- Window SDPA: [9, 16, 576, 64]
- Detector SDPA: [1, 8, 5184, 32]
- Tracker memory SDPA: Q [1, 1, 5184, 256], K/V [1, 1, 5188, 256]
- Tracker memory SDPA: [1, 1, 5184, 256]
- FPN 3x3 conv: [1, 256, 288, 288], kernel 3, padding 1
- Mask-head 3x3 conv: the same shape
- FPN transposed 2x2: [1, 1024, 72, 72], stride 2

MLX convolutions are NHWC. The script builds an NHWC tensor with the same
N, H, W, and C, and times the MLX op on that tensor. It does not transpose
inside the timed region.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import statistics
import sys
import time
import traceback
import warnings
from pathlib import Path

CASES = (
    {
        "case": "vit_mlp_fc1_linear_gelu",
        "kind": "linear_gelu",
        "m": 5184,
        "k": 1024,
        "n": 4736,
        "dtype": "bfloat16",
        "note": "32 calls, 25.2% of rank FLOPs. This is the shim's fallback for aten._addmm_activation.",
    },
    {
        "case": "vit_mlp_fc2",
        "kind": "linear",
        "m": 5184,
        "k": 4736,
        "n": 1024,
        "dtype": "bfloat16",
        "note": "32 calls, 25.2% of rank FLOPs.",
    },
    {
        "case": "vit_qkv",
        "kind": "linear",
        "m": 5184,
        "k": 1024,
        "n": 3072,
        "dtype": "bfloat16",
        "note": "32 calls, 16.4% of rank FLOPs.",
    },
    {
        "case": "vit_attn_proj",
        "kind": "linear",
        "m": 5184,
        "k": 1024,
        "n": 1024,
        "dtype": "bfloat16",
        "note": "32 calls, 5.5% of rank FLOPs.",
    },
    {
        "case": "vit_sdpa_global",
        "kind": "sdpa",
        "q": (1, 16, 5184, 64),
        "kv": (1, 16, 5184, 64),
        "dtype": "bfloat16",
        "note": "4 calls, 6.9% of rank FLOPs. Score matrix is about 0.86 GiB in bf16 if materialized.",
    },
    {
        "case": "vit_sdpa_window",
        "kind": "sdpa",
        "q": (9, 16, 576, 64),
        "kv": (9, 16, 576, 64),
        "dtype": "bfloat16",
        "note": "28 calls, 5.4% of rank FLOPs. Nine 24x24 windows.",
    },
    {
        "case": "detector_sdpa",
        "kind": "sdpa",
        "q": (1, 8, 5184, 32),
        "kv": (1, 8, 5184, 32),
        "dtype": "bfloat16",
        "note": "6 calls, 2.6% of rank FLOPs.",
    },
    {
        "case": "tracker_sdpa_cross",
        "kind": "sdpa",
        "q": (1, 1, 5184, 256),
        "kv": (1, 1, 5188, 256),
        "dtype": "bfloat16",
        "note": "4 calls, 1.7% of rank FLOPs. One head, unequal lengths.",
    },
    {
        "case": "tracker_sdpa_self",
        "kind": "sdpa",
        "q": (1, 1, 5184, 256),
        "kv": (1, 1, 5184, 256),
        "dtype": "bfloat16",
        "note": "4 calls, 1.7% of rank FLOPs.",
    },
    {
        "case": "neck_conv3x3_288",
        "kind": "conv2d",
        "input": (1, 256, 288, 288),
        "weight": (256, 256, 3, 3),
        "stride": 1,
        "padding": 1,
        "dtype": "bfloat16",
        "note": "FPN neck. 1.5% of rank FLOPs. Mask head repeats this shape.",
    },
    {
        "case": "mask_head_conv3x3_288",
        "kind": "conv2d",
        "input": (1, 256, 288, 288),
        "weight": (256, 256, 3, 3),
        "stride": 1,
        "padding": 1,
        "dtype": "bfloat16",
        "note": "Pixel decoder. Same shape as the neck 3x3, timed separately.",
    },
    {
        "case": "neck_convtranspose2x2",
        "kind": "conv_transpose2d",
        "input": (1, 1024, 72, 72),
        "weight": (1024, 512, 2, 2),
        "stride": 2,
        "padding": 0,
        "dtype": "bfloat16",
        "note": "Two of these in the FPN, 0.7% of rank FLOPs together.",
    },
)

SELF_CHECK_CASE = {
    "case": "self_check_linear",
    "kind": "linear",
    "m": 8,
    "k": 16,
    "n": 32,
    "dtype": "float32",
    "note": "Tiny CPU stand-in. Not an MPS measurement.",
}

COLUMNS = (
    "case",
    "backend",
    "device",
    "dtype",
    "median_ms",
    "min_ms",
    "iters",
    "status",
    "fallback",
    "detail",
)


def _torch_dtype(torch, name: str):
    return getattr(torch, name)


def _sync(torch, device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def _time_loop(fn, sync, warmup: int, iters: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    sync()
    samples = []
    for _ in range(iters):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples), min(samples)


def _fallback_text(caught) -> str:
    messages = []
    for item in caught:
        text = str(item.message)
        lowered = text.lower()
        if "fallback" in lowered or "fall back" in lowered:
            messages.append(text)
    return " | ".join(messages)


def _run_pytorch_case(torch, case: dict, device: str, warmup: int, iters: int) -> dict:
    dtype = _torch_dtype(torch, case["dtype"])
    kind = case["kind"]
    row = {
        "case": case["case"],
        "backend": "pytorch",
        "device": device,
        "dtype": case["dtype"],
        "median_ms": "",
        "min_ms": "",
        "iters": iters,
        "status": "ok",
        "fallback": "",
        "detail": case["note"],
    }
    try:
        if kind in ("linear", "linear_gelu"):
            weight = torch.randn(case["n"], case["k"], device=device, dtype=dtype)
            bias = torch.randn(case["n"], device=device, dtype=dtype)
            inputs = torch.randn(case["m"], case["k"], device=device, dtype=dtype)

            def fn():
                hidden = torch.nn.functional.linear(inputs, weight, bias)
                if kind == "linear_gelu":
                    hidden = torch.nn.functional.gelu(hidden)
                return hidden

        elif kind == "sdpa":
            query = torch.randn(*case["q"], device=device, dtype=dtype)
            key = torch.randn(*case["kv"], device=device, dtype=dtype)
            value = torch.randn(*case["kv"], device=device, dtype=dtype)

            def fn():
                return torch.nn.functional.scaled_dot_product_attention(query, key, value)

        elif kind == "conv2d":
            module = torch.nn.Conv2d(
                case["weight"][1],
                case["weight"][0],
                kernel_size=case["weight"][2],
                stride=case["stride"],
                padding=case["padding"],
                bias=True,
            ).to(device=device, dtype=dtype)
            inputs = torch.randn(*case["input"], device=device, dtype=dtype)

            def fn():
                return module(inputs)

        elif kind == "conv_transpose2d":
            module = torch.nn.ConvTranspose2d(
                case["weight"][0],
                case["weight"][1],
                kernel_size=case["weight"][2],
                stride=case["stride"],
                padding=case["padding"],
                bias=True,
            ).to(device=device, dtype=dtype)
            inputs = torch.randn(*case["input"], device=device, dtype=dtype)

            def fn():
                return module(inputs)

        else:
            raise ValueError(f"unknown kind {kind}")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            median_ms, min_ms = _time_loop(fn, lambda: _sync(torch, device), warmup, iters)
        row["median_ms"] = f"{median_ms:.4f}"
        row["min_ms"] = f"{min_ms:.4f}"
        fallback = _fallback_text(caught)
        if fallback:
            row["status"] = "fallback"
            row["fallback"] = fallback
    except Exception as exc:
        row["status"] = "error"
        row["detail"] = f"{case['note']} ERROR: {type(exc).__name__}: {exc}"
    return row


def _run_mlx_case(mx, nn, case: dict, warmup: int, iters: int) -> dict:
    row = {
        "case": case["case"],
        "backend": "mlx",
        "device": "gpu",
        "dtype": case["dtype"],
        "median_ms": "",
        "min_ms": "",
        "iters": iters,
        "status": "ok",
        "fallback": "",
        "detail": case["note"] + " MLX convs are NHWC with the same N,C,H,W.",
    }
    dtype = mx.bfloat16 if case["dtype"] == "bfloat16" else mx.float32
    kind = case["kind"]
    try:
        if kind in ("linear", "linear_gelu"):
            # nn.Linear's parameters are float32. The timed op is the bfloat16
            # matmul the ViT actually runs, plus bias, plus GELU when this is fc1.
            weight = mx.random.normal(shape=(case["n"], case["k"])).astype(dtype)
            bias = mx.random.normal(shape=(case["n"],)).astype(dtype)
            inputs = mx.random.normal(shape=(case["m"], case["k"])).astype(dtype)
            gelu = nn.GELU()

            def fn():
                hidden = inputs @ weight.T + bias
                if kind == "linear_gelu":
                    hidden = gelu(hidden)
                mx.eval(hidden)

        elif kind == "sdpa":
            query = mx.random.normal(shape=case["q"]).astype(dtype)
            key = mx.random.normal(shape=case["kv"]).astype(dtype)
            value = mx.random.normal(shape=case["kv"]).astype(dtype)
            scale = float(case["q"][-1]) ** -0.5

            def fn():
                mx.eval(mx.fast.scaled_dot_product_attention(query, key, value, scale=scale))

        elif kind == "conv2d":
            channels_in = case["input"][1]
            channels_out = case["weight"][0]
            spatial = case["input"][2:]
            layer = nn.Conv2d(
                channels_in,
                channels_out,
                kernel_size=case["weight"][2],
                stride=case["stride"],
                padding=case["padding"],
                bias=True,
            )
            _cast_mlx_module(layer, dtype)
            inputs = mx.random.normal(
                shape=(case["input"][0], spatial[0], spatial[1], channels_in)
            ).astype(dtype)

            def fn():
                mx.eval(layer(inputs))

        elif kind == "conv_transpose2d":
            channels_in = case["input"][1]
            channels_out = case["weight"][1]
            spatial = case["input"][2:]
            layer = nn.ConvTranspose2d(
                channels_in,
                channels_out,
                kernel_size=case["weight"][2],
                stride=case["stride"],
                padding=case["padding"],
                bias=True,
            )
            _cast_mlx_module(layer, dtype)
            inputs = mx.random.normal(
                shape=(case["input"][0], spatial[0], spatial[1], channels_in)
            ).astype(dtype)

            def fn():
                mx.eval(layer(inputs))

        else:
            raise ValueError(f"unknown kind {kind}")

        median_ms, min_ms = _time_loop(fn, lambda: None, warmup, iters)
        row["median_ms"] = f"{median_ms:.4f}"
        row["min_ms"] = f"{min_ms:.4f}"
    except Exception as exc:
        row["status"] = "error"
        row["detail"] = f"{case['note']} ERROR: {type(exc).__name__}: {exc}"
    return row


def _cast_mlx_module(module, dtype) -> None:
    def cast(tree):
        if isinstance(tree, dict):
            return {key: cast(value) for key, value in tree.items()}
        return tree.astype(dtype)

    module.update(cast(module.parameters()))


def _skip_row(case: dict, backend: str, reason: str) -> dict:
    return {
        "case": case["case"],
        "backend": backend,
        "device": "",
        "dtype": case["dtype"],
        "median_ms": "",
        "min_ms": "",
        "iters": "",
        "status": "skipped",
        "fallback": "",
        "detail": reason,
    }


def run_cases(cases, device: str, warmup: int, iters: int, backends: set[str]) -> list[dict]:
    rows = []
    torch = None
    if "pytorch" in backends:
        import torch as torch_mod

        torch = torch_mod
        if device == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            for case in cases:
                rows.append(
                    _skip_row(case, "pytorch", "MPS is not available in this PyTorch build.")
                )
            torch = None
            device = ""
    if torch is not None:
        for case in cases:
            rows.append(_run_pytorch_case(torch, case, device, warmup, iters))
    if "mlx" in backends:
        try:
            import mlx.core as mx
            import mlx.nn as nn
        except Exception as exc:
            for case in cases:
                rows.append(_skip_row(case, "mlx", f"MLX is not importable: {exc}"))
        else:
            for case in cases:
                rows.append(_run_mlx_case(mx, nn, case, warmup, iters))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict], device: str, full_forward: str) -> None:
    timed = [row for row in rows if row["status"] in ("ok", "fallback") and row["median_ms"]]
    timed.sort(key=lambda row: float(row["median_ms"]), reverse=True)
    lines = [
        "# SAM 3 shape timings",
        "",
        f"Host: `{platform.platform()}`",
        f"Requested device: `{device}`",
        f"PYTORCH_ENABLE_MPS_FALLBACK: `{os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '')}`",
        "",
        "These are medians of a synchronized loop. A `fallback` status means PyTorch warned that an op ran on the CPU. An `error` status means the op raised, which is what MPS does when the op is missing and `PYTORCH_ENABLE_MPS_FALLBACK` is unset. This file is a measurement only on the machine named above.",
        "",
        "| Case | Backend | Device | Median ms | Status |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for row in timed:
        lines.append(
            f"| {row['case']} | {row['backend']} | {row['device']} | {row['median_ms']} | {row['status']} |"
        )
    problems = [row for row in rows if row["status"] in ("error", "fallback")]
    lines.extend(["", "## Fallbacks and errors", ""])
    if not problems:
        lines.append("None recorded.")
    else:
        for row in problems:
            lines.append(
                f"- `{row['case']}` / {row['backend']}: {row['status']}. {row['fallback'] or row['detail']}"
            )
    skipped = [row for row in rows if row["status"] == "skipped"]
    if skipped:
        lines.extend(["", "## Skipped", ""])
        for row in skipped:
            lines.append(f"- `{row['case']}` / {row['backend']}: {row['detail']}")
    lines.extend(["", "## Full SAM 3 image forward", "", full_forward, ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def time_full_forward(device: str) -> str:
    """One random-init image forward through the shim. Not a kernel benchmark."""
    try:
        from metal_kernels.integrations.sam3 import SAM3_COMMIT, build_image_model, install

        install(device=device)
        import torch
        from PIL import Image

        torch.manual_seed(0)
        model = build_image_model(
            load_from_HF=False, enable_inst_interactivity=False, eval_mode=True
        )
        from sam3.model.sam3_image_processor import Sam3Processor

        processor = Sam3Processor(model, device=device)
        image = Image.new("RGB", (1008, 1008), (16, 32, 64))
        start = time.perf_counter()
        state = processor.set_image(image)
        state = processor.set_text_prompt("a red cube", state)
        if device == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - start
        fpn = state["backbone_out"]["backbone_fpn"]
        shapes = [tuple(tensor.shape) for tensor in fpn]
        return (
            f"Completed on `{device}` in {elapsed:.2f}s including the Python dispatcher "
            f"(SAM 3 `{SAM3_COMMIT}`, random init, no weights). "
            f"FPN shapes: {shapes}. scores kept: {tuple(state['scores'].shape)}."
        )
    except Exception:
        return "Did not complete.\n\n```\n" + traceback.format_exc() + "```"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("bench/sam3_shapes/output"))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--device", default="mps", help="PyTorch device. Use cpu only for --self-check."
    )
    parser.add_argument(
        "--self-check", action="store_true", help="Time one tiny CPU linear and exit."
    )
    parser.add_argument("--list", action="store_true", help="Print the cases and exit.")
    parser.add_argument("--skip-mlx", action="store_true")
    parser.add_argument("--skip-pytorch", action="store_true")
    parser.add_argument(
        "--full-forward",
        action="store_true",
        help="Also time one patched random-init SAM 3 image forward. Requires the sam3 package.",
    )
    args = parser.parse_args(argv)
    if args.list:
        for case in CASES:
            print(f"{case['case']}: {case['note']}")
        return 0
    if args.self_check:
        rows = run_cases((SELF_CHECK_CASE,), "cpu", warmup=1, iters=2, backends={"pytorch"})
        write_csv(args.output / "shapes.csv", rows)
        write_summary(args.output / "summary.md", rows, "cpu", "Self-check does not run SAM 3.")
        if rows[0]["status"] != "ok":
            print(rows[0]["detail"], file=sys.stderr)
            return 1
        print(f"self-check wrote {args.output}")
        return 0
    if args.device == "mps":
        import torch

        available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not available:
            print(
                "MPS is not available. This script's measurements are for an Apple silicon Mac. "
                "Refusing to time the inventory shapes on CPU. Use --self-check to smoke-test the script.",
                file=sys.stderr,
            )
            return 2
    backends = set()
    if not args.skip_pytorch:
        backends.add("pytorch")
    if not args.skip_mlx:
        backends.add("mlx")
    rows = run_cases(CASES, args.device, args.warmup, args.iters, backends)
    full = "Not requested. Re-run with --full-forward once sam3 is installed."
    if args.full_forward:
        full = time_full_forward(args.device)
        print(full)
    write_csv(args.output / "shapes.csv", rows)
    write_summary(args.output / "summary.md", rows, args.device, full)
    print(f"wrote {args.output / 'shapes.csv'}")
    errors = [row for row in rows if row["status"] == "error"]
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
