# Benchmarks

Benchmark scripts land here next to the kernel they measure. Each one times
that kernel against the built-in framework op (MPS, MLX, or Metal Performance
Shaders) on a named machine, shape, and dtype.

`run.py` prints the registry and does not record timings.

## FlashAttention hd32

`flash_attention_hd32.py` times three variants per framework at
`[1, 8, 5184, 32]`, through MLX and through PyTorch MPS, once in bf16 and once
in fp32 on the same seeded inputs:

- the frozen baseline: v7 for bf16, v6 for fp32 (`BASELINE_VERSIONS`).
  - `v7`: `bench/baselines/flash_attention_hd32_v7.metal`, a byte-for-byte
    copy of the kernel merged in `8917b5a` (K/V stage double buffered: 16 KB
    for bf16, 32 KB for fp32).
  - `v6`: `bench/baselines/flash_attention_hd32_v6.metal`, a byte-for-byte
    copy of the kernel merged in `b2e6847` (one K/V stage: 8 KB for bf16,
    16 KB for fp32).

  Both use 32 query rows and 64-key blocks. Only the benchmark compiles and
  launches them; the public `flash_attention_hd32` still runs only the kernel
  in `kernels/flash_attention_hd32/`. Do not edit the frozen copies to track
  the candidate. The earlier frozen v1 shader,
  `bench/baselines/flash_attention_hd32_v1.metal`, is kept unchanged for the
  v1–v6 records that were timed against it, but is no longer launched.
- `kernel`: the candidate, `flash_attention_hd32`.
- `builtin`: `mx.fast.scaled_dot_product_attention` or
  `torch.nn.functional.scaled_dot_product_attention` on MPS.

For each dtype and framework, every warmup and timed iteration calls the
baseline, the candidate and the built-in once each before the next iteration
starts, so all three see the same machine state. The order is not fixed:
iterations walk `BALANCED_SCHEDULE`, a Williams design over all six orders of
the three variants. Over each six-iteration cycle, every variant takes every
position twice and runs right after each other variant three times (counting
from one iteration's last call to the next iteration's first), never after
itself. A fixed order, or a plain cyclic rotation, would keep the baseline
running after the slower built-in more often than the candidate does. Defaults
are 102 timed iterations plus 12 warmup (`--iterations`, `--warmup`), whole
cycles, so the default run is exactly balanced; other counts that are not
multiples of 6 print a note. For each dtype it prints median, min, p10 and p90
ms per call and TFLOP/s (FLOPs = `4 * B * H * Lq * Lk * D`) for all six
variants, plus the max and mean abs error of each against `sdpa_reference` in
fp32 on the dtype-rounded inputs, and the candidate's speedup over MPS SDPA,
MLX SDPA and the baseline.

A kernel passes correctness when it is within the kernel README tolerances of
both the reference and the built-in (bf16 `rtol = 2e-2`, `atol = 1e-2`; fp32
`rtol = 1e-3`, `atol = 1e-4`) and, in bf16, its max abs error is no larger
than PyTorch MPS SDPA's. Correctness alone decides the exit code. Separately,
each rule pairs the candidate and baseline samples from the same iteration of
each framework's interleave and takes the per-iteration ratio
`candidate_ms / baseline_ms`, then the bootstrap 95% confidence interval of
the median ratio (`record.median_ratio_ci`: percentile bootstrap, 10,000
resamples, fixed seed 0, so the result is reproducible):

- bf16 gate, against v7: passes for a framework when the interval includes 1
  or lies entirely below it (`record.no_slower_gate`).
- fp32 check, against v6: passes when the interval's upper bound is at most
  1.01 (`FP32_RATIO_LIMIT`). It does not gate speed.

p10, median and p90 are still printed and recorded, but do not decide either
rule. The bf16 paired interval for the MPS candidate over MPS SDPA is printed
as the M3 target check; it does not gate. No interval is formed across the MLX
and MPS loops, which run separately. There is one record per dtype; its
`speed_gate` holds that dtype's rule result per framework and `gate_rule`
describes the rule. The recorded `bytes_per_call` is the compulsory traffic
(q, k and v read once, O written once) for every variant.

On the Mac, one command installs the `mac` extra, runs `pytest -m metal`, runs
the benchmark with `--record --version <version>`, runs the Swift step (`swift
build`, `swift test`, and the bf16/fp32 `FlashAttentionHD32Bench` timing), and
prints the git SHA:

```bash
bench/run_mac.sh v7.1
```

It uses `./.venv` when present. v7.1 is the current kernel. Its bf16 code is
the same as v7's, so its bf16 gate asks only that it be no slower than the
frozen v7; its fp32 change is checked against the frozen v6. `bench/run_mac.sh
swift` runs only the Swift step. `v0`, `v1`, `v6`, `v7` and `v7.1` have a
default design text; other versions pass theirs as a second argument:
`bench/run_mac.sh v3 "TEXT"`. The script appends to
`bench/results/flash_attention_hd32.jsonl` and commits nothing.

## Recording kernel versions

Every PR that changes a kernel must add a measured line from John's M3 Max
and regenerate its table. Do not commit invented numbers. The first record will
be FlashAttention hd32 v0, after its Metal kernel and benchmark land.

Each kernel benchmark should accept `--record --version LABEL --design "TEXT"`.
Without `--record`, it must print and behave as before. Use the shared helpers:

```python
from record import add_record_arguments, record_if_requested, validate_record_arguments

# After constructing the benchmark's ArgumentParser:
add_record_arguments(parser)
args = parser.parse_args()
validate_record_arguments(parser, args)
# Run and print the benchmark as usual, then only if requested:
record_if_requested(
    args,
    kernel_name,
    shape=shape,
    dtype=dtype,
    flops=flops_per_call,
    timings=timings,
    bytes_per_call=bytes_per_call,
    correctness=correctness,
    launch_config=launch_config,
)
```

`timings` maps framework names `mlx` and `mps` to `kernel` and `builtin`
lists of synchronized per-iteration times in milliseconds, excluding warmup,
and optionally a `baseline` list for a frozen earlier kernel timed in the same
run. Use only frameworks actually measured. `bytes_per_call` has the same keys,
with the bytes read and written per call for each variant. Median, min,
linearly interpolated p10/p90, iterations, TFLOP/s, GB/s, and kernel speedup
against the corresponding built-in are calculated from those measurements.
With a baseline, the record also has the baseline's statistics, the kernel's
`speedup_vs_baseline`, its `ratio_vs_baseline`, and `speed_gate`, which maps
each framework to `record.speed_gate`: `true` only if the `ci_high` of
`ratio_vs_baseline` is below 1. `speed_gate` is `null` when no baseline was
timed. Each kernel entry also has `ratio_vs_builtin`, reported only. Both ratio
entries come from `record.median_ratio_ci` on the samples paired by index, so
every variant of a framework must have the same number of samples in iteration
order, and hold `median_ratio`, `ci_low`, `ci_high`, `level`, `resamples`,
`seed` and `pairs`.
`shape` and the optional `launch_config` are JSON-serializable dictionaries;
the latter can capture query/key blocks, threads, and threadgroup memory
(`threadgroup_memory_bytes` is 16384 for `flash_attention_hd32`, its double-buffered bf16 K/V stage, and 8192 for the frozen v6 under `launch_config.baseline`).

`correctness` is keyed by the same frameworks. Each entry contains
`reference` and `builtin` dictionaries with `max_abs_error` and
`mean_abs_error`, plus `rtol`, `atol`, and boolean `passed`. Measure the
kernel against both the fp32 NumPy reference and the framework built-in,
using the same bf16-rounded inputs for all three. `passed` must reflect the
benchmark's checks against both comparisons at the stated tolerances.
Collect real measurements, including failures; do not substitute missing
frameworks or fabricate values. For failed comparisons, nonfinite errors are
serialized as the strings `"NaN"` or `"Infinity"`; passing comparisons must
have finite errors.

The helper appends to `bench/results/<kernel>.jsonl`, including the git commit,
dirty status, UTC timestamp and machine/software metadata. Generate the
reviewable table from the repository root:

```bash
python tools/results_table.py <kernel>
```

Commit both the JSONL and `bench/results/<kernel>.md`. Records are append-only;
the table follows their recorded order. Commit the code to be measured before
running the benchmark when possible: otherwise the record is marked dirty.
Failed framework measurements are marked in the table and excluded from its
best TFLOP/s column.
