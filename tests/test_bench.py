# SPDX-License-Identifier: Apache-2.0
"""The benchmark placeholder does not time anything."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

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
