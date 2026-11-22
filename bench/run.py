#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark index.

It lists registered kernels and does not time them. This script does not
invent timings. A real benchmark is added under ``bench/<name>.py`` as step 4
of that kernel; kernels that have one are pointed at it.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BENCH = Path(__file__).resolve().parent

# `python bench/run.py` from a checkout works before an editable install.
_SOURCE = _BENCH.parent / "python" / "src"
if str(_SOURCE) not in sys.path:
    sys.path.insert(0, str(_SOURCE))

from metal_kernels import list_kernels  # noqa: E402


def main() -> int:
    names = list_kernels()
    if not names:
        print("No kernels to benchmark.")
        return 0
    print("No timings recorded.")
    for name in names:
        if (_BENCH / f"{name}.py").is_file():
            print(f"{name}: python bench/{name}.py")
    missing = [name for name in names if not (_BENCH / f"{name}.py").is_file()]
    if missing:
        print("Kernels without a benchmark script yet: " + ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
