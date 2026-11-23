# SPDX-License-Identifier: Apache-2.0
"""Locate Metal kernel sources for the metal-kernels collection.

This package reads ``.metal`` source text from ``kernels/``. Importing it
does not compile or run kernels, and it does not import MLX or PyTorch.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from metal_kernels.loader import (
    KernelNotFoundError,
    kernel_dir,
    kernels_root,
    list_kernels,
    metal_source,
    metal_source_path,
)
from metal_kernels.platform import host_is_macos

try:
    __version__ = version("metal-kernels")
except PackageNotFoundError:
    __version__ = "0.1.1"

__all__ = [
    "KernelNotFoundError",
    "host_is_macos",
    "kernel_dir",
    "kernels_root",
    "list_kernels",
    "metal_source",
    "metal_source_path",
]
