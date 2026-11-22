#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Regenerate a kernel's Markdown results table from its JSONL records."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from bench.record import RESULTS, results_path  # noqa: E402


def _cell(value: object) -> str:
    return html.escape(str(value)).replace("|", "&#124;").replace("\n", " ").replace("\r", " ")


def render(records: list[dict]) -> str:
    lines = [
        "| Version | Date (UTC) | Commit | Design | MLX kernel ms | PyTorch kernel ms | "
        "Best TFLOP/s | Speedup vs built-in |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for record in records:
        variants = record["variants"]
        kernels = {name: data["kernel"] for name, data in variants.items()}
        mlx = kernels.get("mlx")
        mps = kernels.get("mps")
        passed = {name: record["correctness"][name]["passed"] for name in kernels}
        valid = [item["tflops"] for name, item in kernels.items() if passed[name]]
        speedups = ", ".join(
            f"{name}: {item['speedup']:.2f}x" + ("" if passed[name] else " (failed)")
            for name, item in kernels.items()
        )
        cells = [
            record["version"],
            record["recorded_at"][:10],
            record["commit"] + (" (dirty)" if record["dirty"] else ""),
            record["design"],
            (f"{mlx['median_ms']:.3f}" + ("" if passed["mlx"] else " (failed)")) if mlx else "—",
            (f"{mps['median_ms']:.3f}" + ("" if passed["mps"] else " (failed)")) if mps else "—",
            f"{max(valid):.2f}" if valid else "—",
            speedups,
        ]
        lines.append("| " + " | ".join(_cell(cell) for cell in cells) + " |")
    return "\n".join(lines) + "\n"


def generate(kernel: str, directory: Path = RESULTS) -> Path:
    source = results_path(kernel, "jsonl", directory)
    target = results_path(kernel, "md", directory)
    with source.open(encoding="utf-8") as input_file:
        records = [json.loads(line) for line in input_file if line.strip()]
    table = render(records)
    target.write_text(table, encoding="utf-8")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel")
    args = parser.parse_args()
    print(generate(args.kernel))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
