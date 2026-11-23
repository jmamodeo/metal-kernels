# SPDX-License-Identifier: Apache-2.0
"""Tests for recording and rendering without Metal."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime

import pytest

from bench import record
from tools import results_table


@pytest.fixture
def inputs():
    check = {
        "reference": {"max_abs_error": 0.02, "mean_abs_error": 0.01},
        "builtin": {"max_abs_error": 0.03, "mean_abs_error": 0.015},
        "rtol": 0.01,
        "atol": 0.02,
        "passed": True,
    }
    return {
        "version": "v0-tile",
        "design": "Tile | to <fit>\nregisters",
        "shape": {"q": [1, 8, 32, 32]},
        "dtype": "bf16",
        "flops": 2_000_000_000,
        "bytes_per_call": {
            "mlx": {"kernel": 2_000_000, "builtin": 4_000_000},
            "mps": {"kernel": 2_000_000, "builtin": 4_000_000},
        },
        "timings": {
            "mlx": {"kernel": [1, 2, 3], "builtin": [4, 6, 8]},
            "mps": {"kernel": [2, 4], "builtin": [3, 5]},
        },
        "correctness": {"mlx": check, "mps": check},
        "launch_config": {"query_block": 16, "threads": 64},
    }


def test_record_fields_and_append(tmp_path, monkeypatch, inputs):
    monkeypatch.setattr(record, "_git", lambda: ("abc123", True))
    monkeypatch.setattr(
        record,
        "_machine",
        lambda: {"chip": "M3 Max", "macos": "15", "python": "3", "torch": "2", "mlx": "1"},
    )
    first = record.append_record("attention", directory=tmp_path, **inputs)
    inputs["version"] = "v1"
    second = record.append_record("attention", directory=tmp_path, **inputs)
    lines = (tmp_path / "attention.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in lines] == [first, second]
    assert datetime.fromisoformat(first["recorded_at"]).utcoffset().total_seconds() == 0
    assert first["commit"] == "abc123" and first["dirty"] is True
    assert first["machine"]["chip"] == "M3 Max"
    assert first["shape"] == inputs["shape"] and first["dtype"] == "bf16"
    assert first["flops_per_call"] == inputs["flops"]
    assert first["launch_config"] == inputs["launch_config"]
    assert first["correctness"] == inputs["correctness"]
    kernel = first["variants"]["mlx"]["kernel"]
    assert kernel == {
        "median_ms": 2.0,
        "min_ms": 1.0,
        "p10_ms": 1.2,
        "p90_ms": 2.8,
        "iterations": 3,
        "bytes_per_call": 2_000_000,
        "tflops": 1.0,
        "gb_per_s": 1.0,
        "speedup": 3.0,
        "ratio_vs_builtin": record.median_ratio_ci([1, 2, 3], [4, 6, 8]),
    }
    assert kernel["ratio_vs_builtin"]["median_ratio"] == pytest.approx(1 / 3)
    assert first["variants"]["mlx"]["builtin"]["gb_per_s"] == pytest.approx(2 / 3)
    assert "baseline" not in first["variants"]["mlx"]
    assert "speedup_vs_baseline" not in kernel
    assert "ratio_vs_baseline" not in kernel
    assert first["speed_gate"] is None


def test_record_includes_baseline_and_speed_gate(tmp_path, monkeypatch, inputs):
    monkeypatch.setattr(record, "_git", lambda: ("abc123", False))
    inputs["timings"]["mlx"]["baseline"] = [3, 4, 5]
    inputs["timings"]["mps"]["baseline"] = [3, 4]
    for framework in ("mlx", "mps"):
        inputs["bytes_per_call"][framework]["baseline"] = 2_000_000
    data = record.append_record("attention", directory=tmp_path, **inputs)
    assert json.loads((tmp_path / "attention.jsonl").read_text()) == data
    baseline = data["variants"]["mlx"]["baseline"]
    assert baseline == {
        "median_ms": 4.0,
        "min_ms": 3.0,
        "p10_ms": 3.2,
        "p90_ms": 4.8,
        "iterations": 3,
        "bytes_per_call": 2_000_000,
        "tflops": 0.5,
        "gb_per_s": 0.5,
    }
    assert set(data["variants"]["mlx"]) == {"baseline", "kernel", "builtin"}
    assert data["variants"]["mlx"]["kernel"]["speedup_vs_baseline"] == 2.0
    ratio = data["variants"]["mlx"]["kernel"]["ratio_vs_baseline"]
    assert ratio == record.median_ratio_ci([1, 2, 3], [3, 4, 5])
    # Paired mlx ratios 1/3, 1/2, 3/5 are all below 1; mps ratios 2/3 and 1
    # include 1, so a resample of only the second pair has a median ratio of 1.
    assert data["speed_gate"] == {"mlx": True, "mps": False}
    inputs["timings"]["mps"]["kernel"] = [2.7, 3.6]
    data = record.append_record("attention", directory=tmp_path, **inputs)
    assert data["speed_gate"] == {"mlx": True, "mps": True}
    assert len((tmp_path / "attention.jsonl").read_text().splitlines()) == 2
    assert data["gate_rule"] is None
    # A caller-supplied rule replaces speed_gate and is described in the record.
    inputs["timings"]["mps"]["kernel"] = [3.3, 4.4]
    data = record.append_record(
        "attention",
        directory=tmp_path,
        gate=record.ratio_bound_gate(1.01),
        gate_rule="upper bound <= 1.01",
        **inputs,
    )
    assert data["speed_gate"] == {"mlx": True, "mps": False}
    assert data["gate_rule"] == "upper bound <= 1.01"


@pytest.mark.parametrize(
    ("low", "high", "bounded"),
    [
        (0.95, 0.99, True),  # entirely below 1
        (0.99, 1.005, True),  # includes 1
        (1.0, 1.01, True),  # touches the bound
        (1.002, 1.009, True),  # above 1, within 1.01
        (1.005, 1.02, False),  # above the bound
    ],
)
def test_ratio_bound_gate(low, high, bounded):
    ratio = {"ci_low": low, "ci_high": high}
    assert record.ratio_bound_gate(1.01)(ratio) is bounded


def test_record_rejects_mismatched_baseline(tmp_path, inputs):
    inputs["timings"]["mlx"]["baseline"] = [3, 4, 5]
    with pytest.raises(ValueError, match="byte counts"):
        record.append_record("attention", directory=tmp_path, **inputs)
    inputs["timings"]["mlx"]["other"] = [1]
    inputs["bytes_per_call"]["mlx"]["baseline"] = 1
    inputs["bytes_per_call"]["mlx"]["other"] = 1
    with pytest.raises(ValueError, match="optional baseline"):
        record.append_record("attention", directory=tmp_path, **inputs)
    assert not list(tmp_path.iterdir())


def _paired(ratios, start=3.7, stop=4.5):
    """v1 samples spread from ``start`` to ``stop`` ms, and candidates at ``ratios`` of them."""
    count = len(ratios)
    baseline = [start + (stop - start) * ((7 * i) % count) / (count - 1) for i in range(count)]
    return [ratio * value for ratio, value in zip(ratios, baseline)], baseline


def _jitter(center, spread, count=102):
    """Deterministic ratios within ``center +- spread``."""
    return [center + spread * (((5 * i) % 11) - 5) / 5 for i in range(count)]


@pytest.mark.parametrize(
    ("ratios", "passed"),
    [
        (_jitter(0.95, 0.02), True),  # clear 5% gain: CI entirely below 1
        (_jitter(1.0, 0.03), False),  # CI straddles 1
        (_jitter(1.05, 0.02), False),  # CI above 1
        ([1.0] * 102, False),  # identical timings: CI is exactly 1
    ],
)
def test_speed_gate_rule(ratios, passed):
    candidate, baseline = _paired(ratios)
    ratio = record.median_ratio_ci(candidate, baseline)
    assert ratio["ci_low"] <= ratio["median_ratio"] <= ratio["ci_high"]
    assert record.speed_gate(ratio) is passed
    if passed:
        assert ratio["ci_high"] < 1
    elif ratios[0] != 1.0:
        assert ratio["ci_high"] >= 1


def test_paired_gate_detects_gain_the_old_gate_missed():
    candidate, baseline = _paired(_jitter(0.95, 0.02))
    values = sorted(baseline)
    candidate_median = sorted(candidate)[len(candidate) // 2]
    # The old rule: median below v1's and outside v1's p10 to p90 range.
    assert record.percentile(values, 0.1) <= candidate_median <= record.percentile(values, 0.9)
    ratio = record.median_ratio_ci(candidate, baseline)
    assert ratio["median_ratio"] == pytest.approx(0.95)
    assert ratio["ci_high"] < 0.97
    assert record.speed_gate(ratio) is True


def test_median_ratio_ci_is_deterministic_and_documented():
    candidate, baseline = _paired(_jitter(1.0, 0.03))
    first = record.median_ratio_ci(candidate, baseline)
    assert first == record.median_ratio_ci(candidate, baseline)
    assert first["resamples"] == record.BOOTSTRAP_RESAMPLES == 10_000
    assert first["seed"] == record.BOOTSTRAP_SEED
    assert first["level"] == record.CI_LEVEL == 0.95
    assert first["pairs"] == 102
    assert first["ci_low"] < 1 <= first["ci_high"]
    other = record.median_ratio_ci(candidate, baseline, seed=1, resamples=500)
    assert other["seed"] == 1 and other["resamples"] == 500


@pytest.mark.parametrize(
    ("candidate", "baseline", "message"),
    [
        ([1.0, 2.0], [1.0], "same length"),
        ([], [], "same length"),
        ([1.0, 0.0], [1.0, 1.0], "positive finite"),
        ([1.0, float("nan")], [1.0, 1.0], "positive finite"),
    ],
)
def test_median_ratio_ci_rejects_bad_pairs(candidate, baseline, message):
    with pytest.raises(ValueError, match=message):
        record.median_ratio_ci(candidate, baseline)


def test_table_preserves_order_and_escapes_cells(tmp_path, monkeypatch, inputs):
    monkeypatch.setattr(record, "_git", lambda: ("abc123", False))
    record.append_record("attention", directory=tmp_path, **inputs)
    inputs["version"] = "v1"
    record.append_record("attention", directory=tmp_path, **inputs)
    target = results_table.generate("attention", tmp_path)
    text = target.read_text()
    assert text.index("v0-tile") < text.index("v1")
    assert "Tile &#124; to &lt;fit&gt; registers" in text
    assert "2.000 | 3.000 | 1.00 | mlx: 3.00x, mps: 1.33x" in text
    assert len(text.splitlines()) == 4
    assert results_table.generate("attention", tmp_path).read_text() == text


def test_no_record_preserves_behavior(monkeypatch, inputs):
    parser = argparse.ArgumentParser()
    record.add_record_arguments(parser)
    args = parser.parse_args([])
    record.validate_record_arguments(parser, args)
    monkeypatch.setattr(record, "append_record", lambda *a, **k: pytest.fail("unexpected write"))
    assert record.record_if_requested(args, "attention", **inputs) is None
    args = parser.parse_args(["--record"])
    with pytest.raises(SystemExit):
        record.validate_record_arguments(parser, args)


def test_record_flag_appends(tmp_path, monkeypatch, inputs):
    parser = argparse.ArgumentParser()
    record.add_record_arguments(parser)
    args = parser.parse_args(["--record", "--version", "v2", "--design", "Try tiling"])
    record.validate_record_arguments(parser, args)
    monkeypatch.setattr(record, "_git", lambda: ("abc123", False))
    inputs.pop("version")
    inputs.pop("design")
    data = record.record_if_requested(args, "attention", directory=tmp_path, **inputs)
    assert data["version"] == "v2"
    assert data["design"] == "Try tiling"
    assert json.loads((tmp_path / "attention.jsonl").read_text()) == data


def test_git_captures_clean_and_dirty(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "test",
        ],
        check=True,
    )
    monkeypatch.setattr(record, "REPO", tmp_path)
    commit, dirty = record._git()
    assert len(commit) == 40 and not dirty
    (tmp_path / "untracked").write_text("change")
    assert record._git() == (commit, True)


@pytest.mark.parametrize("kernel", ["../outside", "A", "", "a/b"])
def test_invalid_kernel_paths(tmp_path, inputs, kernel):
    with pytest.raises(ValueError):
        record.append_record(kernel, directory=tmp_path, **inputs)
    assert not list(tmp_path.iterdir())


def test_rejects_bad_data_before_writing(tmp_path, inputs):
    inputs["timings"]["mlx"]["kernel"] = [0]
    with pytest.raises(ValueError):
        record.append_record("attention", directory=tmp_path, **inputs)
    assert not list(tmp_path.iterdir())


def test_single_framework_and_failed_correctness(tmp_path, monkeypatch, inputs):
    monkeypatch.setattr(record, "_git", lambda: ("abc123", False))
    for key in ("timings", "bytes_per_call", "correctness"):
        inputs[key].pop("mps")
    inputs["correctness"]["mlx"]["passed"] = False
    data = record.append_record("attention", directory=tmp_path, **inputs)
    assert data["correctness"]["mlx"]["passed"] is False
    table = results_table.render([data])
    assert "2.000 (failed) | — | — | mlx: 3.00x (failed)" in table


def test_nonfinite_failure_and_dirty_marker(tmp_path, monkeypatch, inputs):
    monkeypatch.setattr(record, "_git", lambda: ("abc123", True))
    inputs["correctness"]["mlx"]["passed"] = False
    inputs["correctness"]["mlx"]["reference"] = {
        "max_abs_error": float("inf"),
        "mean_abs_error": float("nan"),
    }
    data = record.append_record("attention", directory=tmp_path, **inputs)
    assert data["correctness"]["mlx"]["reference"] == {
        "max_abs_error": "Infinity",
        "mean_abs_error": "NaN",
    }
    assert json.loads((tmp_path / "attention.jsonl").read_text()) == data
    assert "abc123 (dirty)" in results_table.render([data])
    inputs["correctness"]["mlx"]["passed"] = True
    with pytest.raises(ValueError):
        record.append_record("attention", directory=tmp_path, **inputs)
