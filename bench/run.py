#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark placeholder.

It lists registered kernels and does not time them. This script does not
invent timings. A real benchmark is added under ``bench/<name>.py`` as step 4
of that kernel.
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
    print("No timings recorded. Kernels without a benchmark script yet: " + ", ".join(names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
