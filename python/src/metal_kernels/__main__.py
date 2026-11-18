# SPDX-License-Identifier: Apache-2.0
"""Print the kernel registry. It is empty until a kernel is added."""

from __future__ import annotations

import sys

from metal_kernels.loader import kernels_root, list_kernels


def main() -> int:
    try:
        root = kernels_root()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"kernels root: {root}")
    names = list_kernels()
    if not names:
        print("No kernels registered.")
        return 0
    for name in names:
        print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
