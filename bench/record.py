# SPDX-License-Identifier: Apache-2.0
"""Shared, hardware-independent helpers for kernel benchmark records."""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "bench" / "results"
# Timing variants per framework. ``baseline`` (a frozen earlier kernel timed in
# the same run) is optional; ``kernel`` and ``builtin`` are required.
REQUIRED_VARIANTS = ("kernel", "builtin")
BASELINE = "baseline"


def results_path(kernel: str, suffix: str, directory: Path = RESULTS) -> Path:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", kernel):
        raise ValueError("kernel must be a lowercase name, not a path")
    return Path(directory) / f"{kernel}.{suffix}"


def add_record_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--record", action="store_true", help="append benchmark results")
    parser.add_argument("--version", help="kernel version label (required with --record)")
    parser.add_argument("--design", help="design description (required with --record)")


def validate_record_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.record and (not (args.version or "").strip() or not (args.design or "").strip()):
        parser.error("--record requires --version and --design")


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _machine() -> dict:
    chip = None
    if sys.platform == "darwin":
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    return {
        "chip": chip,
        "macos": platform.mac_ver()[0] or None,
        "python": platform.python_version(),
        "torch": _version("torch"),
        "mlx": _version("mlx"),
    }


def _git() -> tuple[str, bool]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=REPO, text=True
    )
    return commit, bool(status.strip())


def percentile(sorted_values: list[float], fraction: float) -> float:
    position = (len(sorted_values) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def speed_gate(kernel: dict, baseline: dict) -> bool:
    """Whether a candidate is measurably faster than the baseline in one run.

    ``kernel`` and ``baseline`` carry ``median_ms``, ``p10_ms`` and ``p90_ms``.
    The candidate passes only if its median is below the baseline's median
    and outside the baseline's p10 to p90 range.
    """
    median = kernel["median_ms"]
    inside = baseline["p10_ms"] <= median <= baseline["p90_ms"]
    return median < baseline["median_ms"] and not inside


def _timing(samples, flops: int, bytes_per_call: int) -> dict:
    if not math.isfinite(bytes_per_call) or bytes_per_call <= 0:
        raise ValueError("bytes per call must be positive and finite")
    values = sorted(float(value) for value in samples)
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("timings must contain positive finite milliseconds")
    median = statistics.median(values)
    return {
        "median_ms": median,
        "min_ms": values[0],
        "p10_ms": percentile(values, 0.1),
        "p90_ms": percentile(values, 0.9),
        "iterations": len(values),
        "bytes_per_call": bytes_per_call,
        "tflops": flops / (median * 1e9),
        "gb_per_s": bytes_per_call / (median * 1e6),
    }


def append_record(
    kernel: str,
    *,
    version: str,
    design: str,
    shape: dict,
    dtype: str,
    flops: int,
    bytes_per_call: dict,
    timings: dict,
    correctness: dict,
    launch_config: dict | None = None,
    directory: Path = RESULTS,
) -> dict:
    """Append a run; timings and correctness are keyed by framework.

    Each timings entry has ``kernel`` and ``builtin`` millisecond samples,
    and optionally ``baseline`` samples from a frozen earlier kernel timed in
    the same run. ``bytes_per_call`` has the same framework and variant keys
    as timings. With a baseline, the framework's kernel entry gains
    ``speedup_vs_baseline``, and the record's ``speed_gate`` maps the
    framework to the :func:`speed_gate` result. ``speed_gate`` is ``None``
    when no framework has a baseline.
    Correctness entries contain ``reference`` and ``builtin`` error dicts
    (``max_abs_error``, ``mean_abs_error``), ``rtol``, ``atol``, and ``passed``.
    The caller computes correctness on the same inputs used by both implementations.
    """
    path = results_path(kernel, "jsonl", directory)
    if not version.strip() or not design.strip() or not dtype.strip():
        raise ValueError("version, design and dtype must not be empty")
    if not math.isfinite(flops) or flops <= 0:
        raise ValueError("flops must be positive and finite")
    if not timings or set(timings) != set(correctness) or set(timings) != set(bytes_per_call):
        raise ValueError("timings, byte counts and correctness must cover the same frameworks")
    variants = {}
    checks = {}
    gates = {}
    required = set(REQUIRED_VARIANTS)
    for framework, samples in timings.items():
        if set(samples) not in (required, required | {BASELINE}):
            raise ValueError(
                "each framework needs kernel and builtin timings, plus optional baseline"
            )
        if set(bytes_per_call[framework]) != set(samples):
            raise ValueError("each framework needs byte counts for the same variants as timings")
        check = correctness[framework]
        if set(check) != {"reference", "builtin", "rtol", "atol", "passed"}:
            raise ValueError("correctness must include reference, builtin, tolerances and passed")
        if not isinstance(check["passed"], bool):
            raise ValueError("correctness passed must be a boolean")
        normalized = dict(check)
        for field in ("reference", "builtin"):
            errors = check[field]
            if set(errors) != {"max_abs_error", "mean_abs_error"}:
                raise ValueError("correctness errors must include max and mean")
            normalized[field] = {}
            for name, value in errors.items():
                if math.isnan(value) or value == math.inf:
                    if check["passed"]:
                        raise ValueError("nonfinite errors cannot pass correctness")
                    normalized[field][name] = "NaN" if math.isnan(value) else "Infinity"
                elif not math.isfinite(value) or value < 0:
                    raise ValueError("correctness errors must be nonnegative")
                else:
                    normalized[field][name] = value
        if any(not math.isfinite(check[k]) or check[k] < 0 for k in ("rtol", "atol")):
            raise ValueError("tolerances must be nonnegative and finite")
        checks[framework] = normalized
        variants[framework] = {
            name: _timing(samples[name], flops, bytes_per_call[framework][name]) for name in samples
        }
        kernel_timing = variants[framework]["kernel"]
        kernel_timing["speedup"] = (
            variants[framework]["builtin"]["median_ms"] / kernel_timing["median_ms"]
        )
        if BASELINE in samples:
            baseline_timing = variants[framework][BASELINE]
            kernel_timing["speedup_vs_baseline"] = (
                baseline_timing["median_ms"] / kernel_timing["median_ms"]
            )
            gates[framework] = speed_gate(kernel_timing, baseline_timing)
    commit, dirty = _git()
    record = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "commit": commit,
        "dirty": dirty,
        "version": version,
        "design": design,
        "machine": _machine(),
        "shape": shape,
        "dtype": dtype,
        "flops_per_call": flops,
        "variants": variants,
        "correctness": checks,
        "speed_gate": gates or None,
        "launch_config": launch_config,
    }
    line = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(line + "\n")
    return record


def record_if_requested(args: argparse.Namespace, kernel: str, **kwargs) -> dict | None:
    """Call after printing timings; no recording or metadata lookup without --record."""
    if not args.record:
        return None
    if not args.version or not args.design:
        raise ValueError("--record requires --version and --design")
    return append_record(kernel, version=args.version, design=args.design, **kwargs)
