# SPDX-License-Identifier: Apache-2.0
"""The benchmark placeholder reports an empty registry and does not time anything."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def test_benchmark_placeholder_exits_cleanly(monkeypatch, capsys):
    monkeypatch.delenv("METAL_KERNELS_ROOT", raising=False)
    path = _REPO / "bench" / "run.py"
    spec = importlib.util.spec_from_file_location("bench_run", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main() == 0
    assert "No kernels to benchmark." in capsys.readouterr().out
