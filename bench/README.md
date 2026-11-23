# Benchmarks

Benchmark scripts land here next to the kernel they measure. Each one times
that kernel against the built-in framework op (MPS, MLX, or Metal Performance
Shaders) on a named machine, shape, and dtype.

`run.py` prints the registry and does not record timings.

## FlashAttention hd32

`flash_attention_hd32.py` times four variants on the same bf16 inputs at
`[1, 8, 5184, 32]`: our kernel through MLX and through PyTorch MPS,
`mx.fast.scaled_dot_product_attention`, and
`torch.nn.functional.scaled_dot_product_attention` on MPS. It prints median,
p10 and p90 ms per call and TFLOP/s (FLOPs = `4 * B * H * Lq * Lk * D`), plus
the max and mean abs error of each variant against `sdpa_reference` in fp32 on
the bf16-rounded inputs. A kernel passes when it is within `rtol = 2e-2`,
`atol = 1e-2` of both the reference and the built-in, and its max abs error is
no larger than PyTorch MPS SDPA's. The recorded `bytes_per_call` is the
compulsory traffic (q, k and v read once, O written once) for every variant.

On the Mac, one command installs the `mac` extra, runs `pytest -m metal`, runs
the benchmark with `--record --version <version>`, and prints the git SHA:

```bash
bench/run_mac.sh v2
```

It uses `./.venv` when present. `v0`, `v1` and `v2` have a default design text;
later versions pass theirs as a second argument: `bench/run_mac.sh v3 "TEXT"`. The script appends to
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
    args, kernel_name, shape=shape, dtype=dtype, flops=flops_per_call,
    timings=timings, bytes_per_call=bytes_per_call, correctness=correctness,
    launch_config=launch_config,
)
```

`timings` maps framework names `mlx` and `mps` to `kernel` and `builtin`
lists of synchronized per-iteration times in milliseconds, excluding warmup.
Use only frameworks actually measured. `bytes_per_call` has the same keys,
with the bytes read and written per call for each variant. Median, min,
linearly interpolated p10/p90, iterations, TFLOP/s, GB/s, and kernel speedup
against the corresponding built-in are calculated from those measurements.
`shape` and the optional `launch_config` are JSON-serializable dictionaries;
the latter can capture query/key blocks, threads, and threadgroup memory
(`threadgroup_memory_bytes` is 0 for `flash_attention_hd32`, which uses none).

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
