# SPDX-License-Identifier: Apache-2.0
"""The benchmark placeholder does not time anything."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from bench import record

_REPO = Path(__file__).resolve().parents[1]


def _load_bench_run():
    path = _REPO / "bench" / "run.py"
    spec = importlib.util.spec_from_file_location("bench_run", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_placeholder_exits_cleanly(monkeypatch, capsys):
    monkeypatch.delenv("METAL_KERNELS_ROOT", raising=False)
    assert _load_bench_run().main() == 0
    assert "No timings recorded" in capsys.readouterr().out


def test_benchmark_placeholder_with_no_kernels(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("METAL_KERNELS_ROOT", str(tmp_path))
    assert _load_bench_run().main() == 0
    assert "No kernels to benchmark." in capsys.readouterr().out


def test_benchmark_index_points_at_existing_scripts(monkeypatch, capsys):
    monkeypatch.delenv("METAL_KERNELS_ROOT", raising=False)
    assert _load_bench_run().main() == 0
    out = capsys.readouterr().out
    assert "flash_attention_hd32: python bench/flash_attention_hd32.py" in out
    assert "without a benchmark script" not in out


def test_benchmark_index_lists_kernels_without_a_script(tmp_path, monkeypatch, capsys):
    kernel = tmp_path / "no_bench_yet"
    kernel.mkdir(parents=True)
    (kernel / "no_bench_yet.metal").write_text("")
    monkeypatch.setenv("METAL_KERNELS_ROOT", str(tmp_path))
    assert _load_bench_run().main() == 0
    out = capsys.readouterr().out
    assert "Kernels without a benchmark script yet: no_bench_yet" in out
    assert "python bench/no_bench_yet.py" not in out


def _load_flash_attention_bench():
    path = _REPO / "bench" / "flash_attention_hd32.py"
    spec = importlib.util.spec_from_file_location("bench_flash_attention_hd32", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_flash_attention_bench_import_does_not_load_gpu_frameworks():
    script = (
        "import runpy, sys; "
        f"runpy.run_path({str(_REPO / 'bench' / 'flash_attention_hd32.py')!r}); "
        "assert 'torch' not in sys.modules; "
        "assert 'mlx' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_flash_attention_bench_flops_and_bytes():
    bench = _load_flash_attention_bench()
    assert bench.flops_per_call(1, 8, 5184, 5184, 32) == 4 * 8 * 5184 * 5184 * 32
    assert bench.bytes_per_call(1, 8, 5184, 5184, 32) == 2 * 8 * 32 * 4 * 5184
    shape = bench.SHAPE
    assert (shape["batch"], shape["heads"], shape["query_length"], shape["head_dim"]) == (
        1,
        8,
        5184,
        32,
    )


def test_flash_attention_bench_summary():
    bench = _load_flash_attention_bench()
    stats = bench.summarize([4.0, 1.0, 3.0, 2.0, 5.0], flops=2_000_000_000)
    assert stats == {"median_ms": 3.0, "p10_ms": 1.4, "p90_ms": 4.6, "tflops": 2e9 / 3e9}
    assert bench.percentile is record.percentile
    assert record.percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.1) == pytest.approx(1.4)


def test_flash_attention_bench_record_arguments(capsys):
    bench = _load_flash_attention_bench()
    args = bench.parse_args([])
    assert not args.record
    args = bench.parse_args(["--record", "--version", "v0", "--design", "per-row"])
    assert (args.record, args.version, args.design) == (True, "v0", "per-row")
    with pytest.raises(SystemExit):
        bench.parse_args(["--record", "--version", "v0"])
    with pytest.raises(SystemExit):
        bench.parse_args(["--iterations", "0"])
    assert "--record requires --version and --design" in capsys.readouterr().err


def test_flash_attention_bench_defaults_to_100_iterations_plus_warmup():
    bench = _load_flash_attention_bench()
    args = bench.parse_args([])
    assert args.iterations >= 100 and args.warmup >= 1
    assert (args.iterations, args.warmup) == (bench.DEFAULT_ITERATIONS, bench.DEFAULT_WARMUP)
    args = bench.parse_args(["--iterations", "7", "--warmup", "0"])
    assert (args.iterations, args.warmup) == (7, 0)


def test_flash_attention_bench_interleaves_variants():
    bench = _load_flash_attention_bench()
    log = []

    def make(name):
        def call():
            log.append(name)
            return name

        return call

    calls = {name: make(name) for name in bench.VARIANTS}
    syncs = []
    samples, outputs = bench.time_interleaved(calls, syncs.append, warmup=2, iterations=3)
    assert bench.VARIANTS == ("baseline", "kernel", "builtin")
    assert log == list(bench.VARIANTS) * 5
    assert {name: len(values) for name, values in samples.items()} == dict.fromkeys(calls, 3)
    assert outputs == {name: name for name in calls}
    # Every timed call is synchronized before it starts and after it returns.
    assert syncs[6:] == [value for name in bench.VARIANTS * 3 for value in (None, name)]


def test_flash_attention_bench_frozen_v1_baseline():
    bench = _load_flash_attention_bench()
    source = bench.BASELINE_SOURCE.read_text()
    assert bench.BASELINE_SOURCE == _REPO / "bench" / "baselines" / "flash_attention_hd32_v1.metal"
    assert source.startswith("// SPDX-License-Identifier: Apache-2.0\n")
    assert "constant constexpr uint FA_HD32_BLOCK_Q = 32;" in source
    assert "constant constexpr uint FA_HD32_BLOCK_K = 64;" in source
    assert "inline void flash_attention_hd32_block(" in source
    assert "FA_HD32_INSTANTIATE(bfloat, ushort)" in source
    assert "threadgroup float" not in source and "threadgroup_barrier" not in source
    # v1 rescales O and the running sum on every block, unconditionally.
    assert "o_tiles[t].thread_elements() *= correction;" in source
    assert "running_sum = running_sum * correction + fa_hd32_row_sum(block_sum);" in source
    dims, threads, threadgroup = bench.v1_launch((1, 8, 5184, 32), (1, 8, 5184, 32))
    assert dims == [41472, 8, 5184, 5184, 0, 0, 0, 0, 0]
    assert (threads, threadgroup) == (8 * 162 * 128, 128)
    dims, threads, threadgroup = bench.v1_launch((1, 8, 201, 32), (1, 8, 5184, 32))
    assert dims[:4] == [1608, 8, 201, 5184]
    assert (threads, threadgroup) == (8 * 7 * 128, 128)


def test_public_kernel_has_one_source_and_no_version_argument(monkeypatch):
    import inspect

    from metal_kernels.flash_attention import flash_attention_hd32
    from metal_kernels.loader import kernel_dir

    monkeypatch.delenv("METAL_KERNELS_ROOT", raising=False)
    assert [path.name for path in kernel_dir("flash_attention_hd32").glob("*.metal")] == [
        "flash_attention_hd32.metal"
    ]
    parameters = list(inspect.signature(flash_attention_hd32).parameters)
    assert parameters == ["q", "k", "v", "bias", "scale", "backend"]
