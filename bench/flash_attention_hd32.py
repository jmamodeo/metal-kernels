#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Time FlashAttention hd32 against built-in SDPA at the SAM 3 encoder shape.

Four variants run on the same bf16 inputs at ``[1, 8, 5184, 32]``:

- ``mlx kernel``: ``flash_attention_hd32(..., backend="mlx")``
- ``mlx builtin``: ``mx.fast.scaled_dot_product_attention``
- ``mps kernel``: ``flash_attention_hd32(..., backend="mps")``
- ``mps builtin``: ``torch.nn.functional.scaled_dot_product_attention`` on MPS

Each prints median, p10 and p90 ms per call and TFLOP/s, with
FLOPs = ``4 * B * H * Lq * Lk * D``. Errors are measured against
``reference.sdpa.sdpa_reference`` in fp32 on the bf16-rounded inputs.
Results are appended through ``bench/record.py`` only with ``--record``.

Run on Apple silicon from the repository root:

    python bench/flash_attention_hd32.py
    python bench/flash_attention_hd32.py --record --version v1 --design "TEXT"
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
for _path in (_REPO / "python" / "src", _REPO):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from bench.record import (  # noqa: E402
    _percentile,
    add_record_arguments,
    record_if_requested,
    validate_record_arguments,
)

KERNEL = "flash_attention_hd32"
SHAPE = {"batch": 1, "heads": 8, "query_length": 5184, "key_length": 5184, "head_dim": 32}
DTYPE = "bfloat16"
RTOL = 2e-2
ATOL = 1e-2
FRAMEWORKS = ("mlx", "mps")
VARIANTS = ("kernel", "builtin")


def flops_per_call(
    batch: int, heads: int, query_length: int, key_length: int, head_dim: int
) -> int:
    """Two GEMMs of ``2 * Lq * Lk * D`` FLOPs each, per batch and head."""
    return 4 * batch * heads * query_length * key_length * head_dim


def bytes_per_call(
    batch: int, heads: int, query_length: int, key_length: int, head_dim: int, itemsize: int = 2
) -> int:
    """Compulsory traffic: read q, k, v once and write o once."""
    return itemsize * batch * heads * head_dim * (2 * query_length + 2 * key_length)


def summarize(samples: list[float], flops: int) -> dict:
    """Median, p10 and p90 in ms, and TFLOP/s at the median."""
    values = sorted(samples)
    median = statistics.median(values)
    return {
        "median_ms": median,
        "p10_ms": _percentile(values, 0.1),
        "p90_ms": _percentile(values, 0.9),
        "tflops": flops / (median * 1e9),
    }


def abs_errors(actual, expected) -> dict:
    import numpy as np

    error = np.abs(np.asarray(actual, dtype=np.float64) - expected)
    return {"max_abs_error": float(np.max(error)), "mean_abs_error": float(np.mean(error))}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--warmup", type=int, default=3, help="untimed calls per variant")
    parser.add_argument("--iterations", type=int, default=20, help="timed calls per variant")
    parser.add_argument("--seed", type=int, default=0, help="input random seed")
    add_record_arguments(parser)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_record_arguments(parser, args)
    if args.warmup < 0 or args.iterations < 1:
        parser.error("--warmup must be >= 0 and --iterations >= 1")
    return args


def _reference(q, k, v):
    import numpy as np

    from reference.sdpa import sdpa_reference

    # One head at a time keeps the fp32 score matrix to one head's worth.
    out = np.empty_like(q)
    for b in range(q.shape[0]):
        for h in range(q.shape[1]):
            sl = (slice(b, b + 1), slice(h, h + 1))
            out[sl] = sdpa_reference(q[sl], k[sl], v[sl])
    return out


def _time(call, synchronize, warmup: int, iterations: int) -> tuple[list[float], object]:
    out = None
    for _ in range(warmup):
        out = call()
        synchronize(out)
    samples = []
    for _ in range(iterations):
        synchronize(None)
        start = time.perf_counter()
        out = call()
        synchronize(out)
        samples.append((time.perf_counter() - start) * 1e3)
    return samples, out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    import mlx.core as mx
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from metal_kernels.flash_attention import (
        KEY_BLOCK,
        QUERY_BLOCK,
        THREADGROUP_MEMORY_BYTES,
        flash_attention_hd32,
        launch_geometry,
    )

    if not torch.backends.mps.is_available():
        print("PyTorch MPS is not available.", file=sys.stderr)
        return 1

    batch, heads = SHAPE["batch"], SHAPE["heads"]
    query_length, key_length, head_dim = (
        SHAPE["query_length"],
        SHAPE["key_length"],
        SHAPE["head_dim"],
    )
    flops = flops_per_call(batch, heads, query_length, key_length, head_dim)
    traffic = bytes_per_call(batch, heads, query_length, key_length, head_dim)
    scale = 1.0 / math.sqrt(head_dim)

    rng = np.random.default_rng(args.seed)
    q_shape = (batch, heads, query_length, head_dim)
    kv_shape = (batch, heads, key_length, head_dim)
    host = [rng.standard_normal(shape, dtype=np.float32) for shape in (q_shape, kv_shape, kv_shape)]
    # Both frameworks round fp32 to bf16 to nearest even, so the inputs match.
    torch_inputs = [torch.from_numpy(x).to("mps", torch.bfloat16) for x in host]
    rounded = [x.float().cpu().numpy() for x in torch_inputs]
    mlx_inputs = [mx.array(x).astype(mx.bfloat16) for x in rounded]
    expected = _reference(*rounded)

    def mlx_sync(out):
        if out is not None:
            mx.eval(out)

    def mps_sync(_out):
        torch.mps.synchronize()

    calls = {
        "mlx": {
            "kernel": lambda: flash_attention_hd32(*mlx_inputs, backend="mlx"),
            "builtin": lambda: mx.fast.scaled_dot_product_attention(*mlx_inputs, scale=scale),
        },
        "mps": {
            "kernel": lambda: flash_attention_hd32(*torch_inputs, backend="mps"),
            "builtin": lambda: functional.scaled_dot_product_attention(*torch_inputs),
        },
    }
    to_numpy = {
        "mlx": lambda out: np.array(out.astype(mx.float32)),
        "mps": lambda out: out.float().cpu().numpy(),
    }
    sync = {"mlx": mlx_sync, "mps": mps_sync}

    timings: dict = {}
    outputs: dict = {}
    for framework in FRAMEWORKS:
        timings[framework] = {}
        outputs[framework] = {}
        for variant in VARIANTS:
            samples, out = _time(
                calls[framework][variant], sync[framework], args.warmup, args.iterations
            )
            timings[framework][variant] = samples
            outputs[framework][variant] = to_numpy[framework](out)

    reference_errors = {
        (framework, variant): abs_errors(outputs[framework][variant], expected)
        for framework in FRAMEWORKS
        for variant in VARIANTS
    }
    mps_sdpa_max = reference_errors[("mps", "builtin")]["max_abs_error"]

    correctness = {}
    for framework in FRAMEWORKS:
        kernel_out = outputs[framework]["kernel"]
        builtin_out = outputs[framework]["builtin"]
        reference_error = reference_errors[(framework, "kernel")]
        passed = bool(
            np.allclose(kernel_out, expected, rtol=RTOL, atol=ATOL)
            and np.allclose(kernel_out, builtin_out, rtol=RTOL, atol=ATOL)
            and reference_error["max_abs_error"] <= mps_sdpa_max
        )
        correctness[framework] = {
            "reference": reference_error,
            "builtin": abs_errors(kernel_out, builtin_out),
            "rtol": RTOL,
            "atol": ATOL,
            "passed": passed,
        }

    print(
        f"shape q=k=v={list(q_shape)} dtype={DTYPE} "
        f"flops/call={flops:.4g} iterations={args.iterations} warmup={args.warmup}"
    )
    print(
        f"{'variant':<12} {'median ms':>10} {'p10 ms':>10} {'p90 ms':>10} {'TFLOP/s':>8} "
        f"{'max abs err':>12} {'mean abs err':>13}"
    )
    for framework in FRAMEWORKS:
        for variant in VARIANTS:
            stats = summarize(timings[framework][variant], flops)
            errors = reference_errors[(framework, variant)]
            print(
                f"{framework + ' ' + variant:<12} {stats['median_ms']:>10.3f} "
                f"{stats['p10_ms']:>10.3f} {stats['p90_ms']:>10.3f} {stats['tflops']:>8.3f} "
                f"{errors['max_abs_error']:>12.3e} {errors['mean_abs_error']:>13.3e}"
            )
    print("Errors are against sdpa_reference (fp32) on the bf16-rounded inputs.")
    for framework in FRAMEWORKS:
        check = correctness[framework]
        print(
            f"{framework} kernel: {'passed' if check['passed'] else 'FAILED'} "
            f"(rtol={RTOL}, atol={ATOL} vs reference and built-in; "
            f"max abs error <= MPS SDPA's {mps_sdpa_max:.3e}); "
            f"vs built-in max {check['builtin']['max_abs_error']:.3e}"
        )

    _, threads, threadgroup = launch_geometry(q_shape, kv_shape, kv_shape)
    record = record_if_requested(
        args,
        KERNEL,
        shape=SHAPE,
        dtype=DTYPE,
        flops=flops,
        timings=timings,
        bytes_per_call={
            framework: {variant: traffic for variant in VARIANTS} for framework in FRAMEWORKS
        },
        correctness=correctness,
        launch_config={
            "threads": threads,
            "threadgroup": threadgroup,
            "query_block": QUERY_BLOCK,
            "key_block": KEY_BLOCK,
            "rows_per_simdgroup": 8,
            "threadgroup_memory_bytes": THREADGROUP_MEMORY_BYTES,
        },
    )
    if record is not None:
        print(f"Recorded {args.version} in bench/results/{KERNEL}.jsonl")
    return 0 if all(check["passed"] for check in correctness.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
