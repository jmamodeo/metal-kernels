#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark placeholder.

There is nothing to time until a kernel exists. This script does not invent
timings. A real benchmark is added under ``bench/<name>.py`` with its kernel.
"""

from __future__ import annotations

import sys
from pathlib import Path

# `python bench/run.py` from a checkout works before an editable install.
_SOURCE = Path(__file__).resolve().parents[1] / "python" / "src"
if str(_SOURCE) not in sys.path:
    sys.path.insert(0, str(_SOURCE))

from metal_kernels import list_kernels  # noqa: E402


def main() -> int:
    names = list_kernels()
    if not names:
        print("No kernels to benchmark.")
        return 0
    print("No benchmark is implemented for: " + ", ".join(names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
