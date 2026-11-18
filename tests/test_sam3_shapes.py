# SPDX-License-Identifier: Apache-2.0
"""The Mac profiler exposes the inventory shapes and can smoke-test on CPU."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _profile():
    path = Path(__file__).resolve().parents[1] / "bench" / "sam3_shapes" / "profile.py"
    spec = importlib.util.spec_from_file_location("sam3_shapes_profile", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cases_cover_the_inventory_leaders():
    profile = _profile()
    names = [case["case"] for case in profile.CASES]
    for required in (
        "vit_mlp_fc1_linear_gelu",
        "vit_mlp_fc2",
        "vit_qkv",
        "vit_sdpa_global",
        "vit_sdpa_window",
        "detector_sdpa",
        "tracker_sdpa_cross",
        "tracker_sdpa_self",
        "neck_conv3x3_288",
        "neck_convtranspose2x2",
    ):
        assert required in names
    global_attn = next(case for case in profile.CASES if case["case"] == "vit_sdpa_global")
    assert global_attn["q"] == (1, 16, 5184, 64)
    cross = next(case for case in profile.CASES if case["case"] == "tracker_sdpa_cross")
    assert cross["q"] == (1, 1, 5184, 256)
    assert cross["kv"] == (1, 1, 5188, 256)


def test_self_check_writes_csv(tmp_path):
    pytest = __import__("pytest")
    pytest.importorskip("torch")
    profile = _profile()
    assert profile.main(["--self-check", "--output", str(tmp_path)]) == 0
    text = (tmp_path / "shapes.csv").read_text(encoding="utf-8")
    assert "self_check_linear" in text
    assert "ok" in text
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "Self-check does not run SAM 3" in summary
