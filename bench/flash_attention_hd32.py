#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Time FlashAttention hd32 against built-in SDPA at the SAM 3 encoder shape.

Each dtype in ``DTYPES`` (bf16, then fp32) runs six variants on the same
inputs at ``[1, 8, 5184, 32]``:

- ``mlx <baseline>`` and ``mps <baseline>``: a frozen earlier shader from
  ``bench/baselines/``, launched only from here. bf16 uses v7
  (``flash_attention_hd32_v7.metal``), fp32 uses v6
  (``flash_attention_hd32_v6.metal``); see ``BASELINE_VERSIONS``.
- ``mlx kernel``: ``flash_attention_hd32(..., backend="mlx")``, the candidate
- ``mlx builtin``: ``mx.fast.scaled_dot_product_attention``
- ``mps kernel``: ``flash_attention_hd32(..., backend="mps")``, the candidate
- ``mps builtin``: ``torch.nn.functional.scaled_dot_product_attention`` on MPS

For each dtype and framework, every warmup and timed iteration calls the
baseline, the candidate and the built-in once each, in an order taken from
``BALANCED_SCHEDULE`` (a Williams design: every variant takes every position,
and follows each other variant, equally often over one six-iteration cycle).
Each prints median, min, p10 and p90 ms per call and TFLOP/s, with FLOPs =
``4 * B * H * Lq * Lk * D``. Errors are measured against
``reference.sdpa.sdpa_reference`` in fp32 on the dtype-rounded inputs.

Samples from the same iteration are paired, and every rule below uses the
bootstrap 95% CI of the median ``candidate / baseline`` ratio
(``record.median_ratio_ci``: 10,000 resamples, seed 0), per framework:

- bf16 gate (against v7): the CI's upper bound is at most ``RATIO_LIMIT``
  (1.01). "Includes 1" was too strict for build-to-build noise.
- fp32 gate (against v6): the CI's upper bound is at most ``RATIO_LIMIT``.

The paired CI of the bf16 MPS candidate over MPS SDPA is reported as the M3
target check but does not gate. All are printed and recorded, one record per
dtype; correctness alone decides the exit code. Results are appended through
``bench/record.py`` only with ``--record``.

Run on Apple silicon from the repository root:

    python bench/flash_attention_hd32.py
    python bench/flash_attention_hd32.py --record --version v7.1 --design "TEXT"
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
    BASELINE,
    add_record_arguments,
    median_ratio_ci,
    percentile,
    ratio_bound_gate,
    record_if_requested,
    validate_record_arguments,
)

KERNEL = "flash_attention_hd32"
SHAPE = {"batch": 1, "heads": 8, "query_length": 5184, "key_length": 5184, "head_dim": 32}
DTYPES = ("bfloat16", "float32")
ITEMSIZE = {"bfloat16": 2, "float32": 4}
# Output tolerance against the fp32 reference, per dtype, as in the kernel
# README. bf16 must also have a max abs error no larger than MPS SDPA's.
TOLERANCE = {"bfloat16": (2e-2, 1e-2), "float32": (1e-3, 1e-4)}
FRAMEWORKS = ("mlx", "mps")
VARIANTS = (BASELINE, "kernel", "builtin")
# Frozen baseline timed against the candidate, per dtype.
BASELINE_VERSIONS = {"bfloat16": "v7", "float32": "v6"}
# Gate for both dtypes: the paired candidate/baseline CI's upper bound may
# not exceed this.
RATIO_LIMIT = 1.01
GATES = {dtype: ratio_bound_gate(RATIO_LIMIT) for dtype in DTYPES}
GATE_RULES = {
    "bfloat16": (
        f"bf16 gate: paired candidate/v7 median-ratio bootstrap 95% CI upper bound <= {RATIO_LIMIT}"
    ),
    "float32": (
        f"fp32 gate: paired candidate/v6 median-ratio bootstrap 95% CI upper bound <= {RATIO_LIMIT}"
    ),
}
# Williams design for three variants: all six orders, sequenced so that over
# one cycle each variant takes each position twice and, counting the step from
# one iteration's last call to the next iteration's first, is preceded by each
# other variant three times and never by itself. A plain cyclic rotation
# balances positions but would always run the baseline right after the
# built-in. Iteration i uses BALANCED_SCHEDULE[i % 6], as indices into
# VARIANTS.
BALANCED_SCHEDULE = ((0, 1, 2), (0, 2, 1), (2, 1, 0), (1, 0, 2), (1, 2, 0), (2, 0, 1))
SCHEDULE_CYCLE = len(BALANCED_SCHEDULE)
# Whole cycles, so the defaults are exactly balanced.
DEFAULT_WARMUP = 2 * SCHEDULE_CYCLE
DEFAULT_ITERATIONS = 17 * SCHEDULE_CYCLE

# Frozen baselines, byte-for-byte copies of merged kernels, each with its own
# source and independent of the candidate in kernels/flash_attention_hd32/.
# v6 (merged in b2e6847) stages one K and one V block; v7 (merged in 8917b5a)
# double buffers that stage. Both share this launch geometry and MLX body.
BASELINE_SOURCES = {
    "v6": _REPO / "bench" / "baselines" / "flash_attention_hd32_v6.metal",
    "v7": _REPO / "bench" / "baselines" / "flash_attention_hd32_v7.metal",
}
BASELINE_KV_LOAD = {
    "v6": "device to threadgroup memory (8 KB for bf16, 16 KB for fp32), shared by 4 simdgroups",
    "v7": (
        "device to double-buffered threadgroup memory (16 KB for bf16, 32 KB for fp32), "
        "shared by 4 simdgroups"
    ),
}
BASELINE_QUERY_BLOCK = 32
BASELINE_KEY_BLOCK = 64
BASELINE_THREADGROUP_SIZE = 128
# Entry point of each shader's PyTorch MPS instantiation, per dtype.
MPS_ENTRY = {"bfloat16": "flash_attention_hd32_bfloat", "float32": "flash_attention_hd32_float"}
_BASELINE_MLX_BODY = """
    FA_HD32_KV_STAGE(kv, fa_hd32_bits(k));
    flash_attention_hd32_block(
        fa_hd32_bits(q),
        fa_hd32_bits(k),
        fa_hd32_bits(v),
        fa_hd32_bits(bias),
        fa_hd32_bits(out),
        kv,
        fa_hd32_params(dims, scale[0]),
        threadgroup_position_in_grid.x,
        thread_index_in_threadgroup);
"""


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
    """Median, min, p10 and p90 in ms, and TFLOP/s at the median."""
    values = sorted(samples)
    median = statistics.median(values)
    return {
        "median_ms": median,
        "min_ms": values[0],
        "p10_ms": percentile(values, 0.1),
        "p90_ms": percentile(values, 0.9),
        "tflops": flops / (median * 1e9),
    }


def format_ratio(ratio: dict) -> str:
    """``median [low, high] (95% CI, N resamples)`` for a :func:`median_ratio_ci` result."""
    return (
        f"{ratio['median_ratio']:.4f} [{ratio['ci_low']:.4f}, {ratio['ci_high']:.4f}] "
        f"({ratio['level']:.0%} CI, {ratio['resamples']} resamples)"
    )


def labels(dtype: str) -> dict:
    """Printed variant names for ``dtype``: the baseline is its frozen version."""
    return {BASELINE: BASELINE_VERSIONS[dtype], "kernel": "kernel", "builtin": "builtin"}


def baseline_launch(q_shape, kv_shape) -> tuple[list[int], int, int]:
    """``(dims, threads, threadgroup)`` for the frozen shaders, no bias.

    ``dims`` follows the shader's ``FaHd32Dim`` layout. One threadgroup of
    ``BASELINE_THREADGROUP_SIZE`` threads per ``BASELINE_QUERY_BLOCK`` query
    rows of each (batch, head).
    """
    batch, heads, query_length, _ = q_shape
    key_length = kv_shape[2]
    dims = [batch * heads * query_length, heads, query_length, key_length, 0, 0, 0, 0, 0]
    groups = batch * heads * -(-query_length // BASELINE_QUERY_BLOCK)
    return dims, groups * BASELINE_THREADGROUP_SIZE, BASELINE_THREADGROUP_SIZE


def abs_errors(actual, expected) -> dict:
    import numpy as np

    error = np.abs(np.asarray(actual, dtype=np.float64) - expected)
    return {"max_abs_error": float(np.max(error)), "mean_abs_error": float(np.mean(error))}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--warmup", type=int, default=DEFAULT_WARMUP, help="untimed calls per variant"
    )
    parser.add_argument(
        "--iterations", type=int, default=DEFAULT_ITERATIONS, help="timed calls per variant"
    )
    parser.add_argument("--seed", type=int, default=0, help="input random seed")
    add_record_arguments(parser)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_record_arguments(parser, args)
    if args.warmup < 0 or args.iterations < 1:
        parser.error("--warmup must be >= 0 and --iterations >= 1")
    if args.warmup % SCHEDULE_CYCLE or args.iterations % SCHEDULE_CYCLE:
        print(
            f"note: --warmup and --iterations that are multiples of {SCHEDULE_CYCLE} keep "
            "the call order exactly balanced",
            file=sys.stderr,
        )
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


def balanced_schedule(names, iterations: int, start: int = 0) -> list[tuple]:
    """Call order for each of ``iterations`` iterations, from ``BALANCED_SCHEDULE``.

    ``names`` are the three variants; iteration ``i`` uses cycle entry
    ``(start + i) % SCHEDULE_CYCLE``. Exactly balanced when ``iterations`` is
    a multiple of ``SCHEDULE_CYCLE``.
    """
    names = tuple(names)
    if len(names) != len(BALANCED_SCHEDULE[0]):
        raise ValueError(f"the balanced schedule needs exactly {len(BALANCED_SCHEDULE[0])} calls")
    return [
        tuple(names[index] for index in BALANCED_SCHEDULE[(start + i) % SCHEDULE_CYCLE])
        for i in range(iterations)
    ]


def time_interleaved(calls: dict, synchronize, warmup: int, iterations: int) -> tuple[dict, dict]:
    """Time ``calls`` interleaved and return ``(samples_ms, last_outputs)``.

    Each warmup and timed iteration runs every call once before the next
    iteration starts, in the order given by :func:`balanced_schedule`. The
    schedule continues from warmup into the timed iterations. Each timed call
    is synchronized before and after, so samples are per-call GPU wall time.
    Samples are in iteration order, so index ``i`` of every variant comes from
    the same iteration and the variants can be paired.
    """
    outputs = dict.fromkeys(calls)
    samples = {name: [] for name in calls}
    for order in balanced_schedule(calls, warmup):
        for name in order:
            outputs[name] = calls[name]()
            synchronize(outputs[name])
    for order in balanced_schedule(calls, iterations, start=warmup):
        for name in order:
            call = calls[name]
            synchronize(None)
            start = time.perf_counter()
            outputs[name] = call()
            synchronize(outputs[name])
            samples[name].append((time.perf_counter() - start) * 1e3)
    return samples, outputs


def _baseline_mlx_call(version, inputs, q_shape, kv_shape, scale):
    import mlx.core as mx

    kernel = mx.fast.metal_kernel(
        name=f"flash_attention_hd32_{version}_baseline",
        input_names=["q", "k", "v", "bias", "dims", "scale"],
        output_names=["out"],
        source=_BASELINE_MLX_BODY,
        header=BASELINE_SOURCES[version].read_text(encoding="utf-8"),
        ensure_row_contiguous=True,
    )
    dims, threads, threadgroup = baseline_launch(q_shape, kv_shape)
    constants = (
        mx.array(dims, dtype=mx.int32),
        mx.array([scale], dtype=mx.float32),
        mx.zeros((1,), dtype=inputs[0].dtype),
    )
    mx.eval(*constants)
    dims_array, scale_array, placeholder = constants

    def call():
        (out,) = kernel(
            inputs=[*inputs, placeholder, dims_array, scale_array],
            output_shapes=[inputs[0].shape],
            output_dtypes=[inputs[0].dtype],
            grid=(threads, 1, 1),
            threadgroup=(threadgroup, 1, 1),
        )
        return out

    return call


def _baseline_mps_call(version, dtype, inputs, q_shape, kv_shape, scale):
    import torch

    library = torch.mps.compile_shader(BASELINE_SOURCES[version].read_text(encoding="utf-8"))
    entry = getattr(library, MPS_ENTRY[dtype])
    dims, threads, threadgroup = baseline_launch(q_shape, kv_shape)
    q, k, v = inputs
    device = q.device
    dims_tensor = torch.tensor(dims, dtype=torch.int32, device=device)
    scale_tensor = torch.tensor([scale], dtype=torch.float32, device=device)
    placeholder = torch.zeros(1, dtype=q.dtype, device=device)

    def call():
        out = torch.empty(q.shape, dtype=q.dtype, device=device)
        entry(
            q,
            k,
            v,
            placeholder,
            out,
            dims_tensor,
            scale_tensor,
            threads=threads,
            group_size=threadgroup,
        )
        return out

    return call


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    import mlx.core as mx
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from metal_kernels.flash_attention import (
        KEY_BLOCK,
        QUERY_BLOCK,
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
    scale = 1.0 / math.sqrt(head_dim)

    rng = np.random.default_rng(args.seed)
    q_shape = (batch, heads, query_length, head_dim)
    kv_shape = (batch, heads, key_length, head_dim)
    host = [rng.standard_normal(shape, dtype=np.float32) for shape in (q_shape, kv_shape, kv_shape)]
    torch_dtypes = {"bfloat16": torch.bfloat16, "float32": torch.float32}
    mlx_dtypes = {"bfloat16": mx.bfloat16, "float32": mx.float32}

    def mlx_sync(out):
        # time_interleaved's pre-timer synchronize(None) is a no-op here on purpose; MPS syncs.
        if out is not None:
            mx.eval(out)

    def mps_sync(_out):
        torch.mps.synchronize()

    to_numpy = {
        "mlx": lambda out: np.array(out.astype(mx.float32)),
        "mps": lambda out: out.float().cpu().numpy(),
    }
    sync = {"mlx": mlx_sync, "mps": mps_sync}
    _, threads, threadgroup = launch_geometry(q_shape, kv_shape, kv_shape)
    all_passed = True

    for dtype in DTYPES:
        baseline_version = BASELINE_VERSIONS[dtype]
        names = labels(dtype)
        rtol, atol = TOLERANCE[dtype]
        traffic = bytes_per_call(
            batch, heads, query_length, key_length, head_dim, itemsize=ITEMSIZE[dtype]
        )
        # Both frameworks round fp32 to bf16 to nearest even, so the inputs match.
        torch_inputs = [torch.from_numpy(x).to("mps", torch_dtypes[dtype]) for x in host]
        rounded = [x.float().cpu().numpy() for x in torch_inputs]
        mlx_inputs = [mx.array(x).astype(mlx_dtypes[dtype]) for x in rounded]
        expected = _reference(*rounded)

        calls = {
            "mlx": {
                BASELINE: _baseline_mlx_call(
                    baseline_version, mlx_inputs, q_shape, kv_shape, scale
                ),
                "kernel": lambda: flash_attention_hd32(*mlx_inputs, backend="mlx"),
                "builtin": lambda: mx.fast.scaled_dot_product_attention(*mlx_inputs, scale=scale),
            },
            "mps": {
                BASELINE: _baseline_mps_call(
                    baseline_version, dtype, torch_inputs, q_shape, kv_shape, scale
                ),
                "kernel": lambda: flash_attention_hd32(*torch_inputs, backend="mps"),
                "builtin": lambda: functional.scaled_dot_product_attention(*torch_inputs),
            },
        }

        timings: dict = {}
        outputs: dict = {}
        for framework in FRAMEWORKS:
            ordered = {variant: calls[framework][variant] for variant in VARIANTS}
            samples, last = time_interleaved(ordered, sync[framework], args.warmup, args.iterations)
            timings[framework] = samples
            outputs[framework] = {
                variant: to_numpy[framework](last[variant]) for variant in VARIANTS
            }

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
                np.allclose(kernel_out, expected, rtol=rtol, atol=atol)
                and np.allclose(kernel_out, builtin_out, rtol=rtol, atol=atol)
                and (dtype != "bfloat16" or reference_error["max_abs_error"] <= mps_sdpa_max)
            )
            correctness[framework] = {
                "reference": reference_error,
                "builtin": abs_errors(kernel_out, builtin_out),
                "rtol": rtol,
                "atol": atol,
                "passed": passed,
            }
        all_passed = all_passed and all(check["passed"] for check in correctness.values())

        print(
            f"shape q=k=v={list(q_shape)} dtype={dtype} baseline={baseline_version} "
            f"flops/call={flops:.4g} iterations={args.iterations} warmup={args.warmup}"
        )
        print(
            f"{'variant':<12} {'median ms':>10} {'min ms':>10} {'p10 ms':>10} {'p90 ms':>10} "
            f"{'TFLOP/s':>8} {'max abs err':>12} {'mean abs err':>13}"
        )
        stats = {
            (framework, variant): summarize(timings[framework][variant], flops)
            for framework in FRAMEWORKS
            for variant in VARIANTS
        }
        for framework in FRAMEWORKS:
            for variant in VARIANTS:
                row = stats[(framework, variant)]
                errors = reference_errors[(framework, variant)]
                print(
                    f"{framework + ' ' + names[variant]:<12} {row['median_ms']:>10.3f} "
                    f"{row['min_ms']:>10.3f} {row['p10_ms']:>10.3f} {row['p90_ms']:>10.3f} "
                    f"{row['tflops']:>8.3f} "
                    f"{errors['max_abs_error']:>12.3e} {errors['mean_abs_error']:>13.3e}"
                )
        print(f"Errors are against sdpa_reference (fp32) on the {dtype}-rounded inputs.")
        bound = f"; max abs error <= MPS SDPA's {mps_sdpa_max:.3e}" if dtype == "bfloat16" else ""
        for framework in FRAMEWORKS:
            check = correctness[framework]
            print(
                f"{framework} kernel: {'passed' if check['passed'] else 'FAILED'} "
                f"(rtol={rtol}, atol={atol} vs reference and built-in{bound}); "
                f"vs built-in max {check['builtin']['max_abs_error']:.3e}"
            )
        mlx_sdpa = stats[("mlx", "builtin")]["median_ms"]
        mps_sdpa = stats[("mps", "builtin")]["median_ms"]
        for framework in FRAMEWORKS:
            kernel = stats[(framework, "kernel")]
            baseline = stats[(framework, BASELINE)]
            ratio = median_ratio_ci(timings[framework]["kernel"], timings[framework][BASELINE])
            print(
                f"{framework} {GATE_RULES[dtype]}: "
                f"{'passed' if GATES[dtype](ratio) else 'not passed'} "
                f"(kernel/{baseline_version} median ratio {format_ratio(ratio)}); "
                f"speedup {mps_sdpa / kernel['median_ms']:.2f}x vs MPS SDPA, "
                f"{mlx_sdpa / kernel['median_ms']:.2f}x vs MLX SDPA, "
                f"{baseline['median_ms'] / kernel['median_ms']:.2f}x vs {baseline_version}"
            )
        if dtype == "bfloat16":
            ratio = median_ratio_ci(timings["mps"]["kernel"], timings["mps"]["builtin"])
            print(
                f"mps kernel vs MPS SDPA (M3 target check, not a gate): "
                f"median ratio {format_ratio(ratio)}; "
                f"{'below' if ratio['ci_high'] < 1 else 'not below'} 1"
            )

        short = "bf16" if dtype == "bfloat16" else "fp32"
        record = record_if_requested(
            args,
            KERNEL,
            shape=SHAPE,
            dtype=dtype,
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
                "kv_load": (
                    "device to threadgroup memory, double buffered for bf16 and single "
                    "buffered for fp32 (16 KB for both), shared by 4 simdgroups"
                ),
                "mma_inputs": f"{short} Q, K, V; fp32 P",
                "mma_accumulator": "fp32",
                "baseline": {
                    "version": baseline_version,
                    "source": BASELINE_SOURCES[baseline_version].relative_to(_REPO).as_posix(),
                    "query_block": BASELINE_QUERY_BLOCK,
                    "key_block": BASELINE_KEY_BLOCK,
                    "threadgroup": BASELINE_THREADGROUP_SIZE,
                    "kv_load": BASELINE_KV_LOAD[baseline_version],
                },
            },
            gate=GATES[dtype],
            gate_rule=GATE_RULES[dtype],
        )
        if record is not None:
            print(f"Recorded {args.version} {dtype} in bench/results/{KERNEL}.jsonl")
        print()
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
