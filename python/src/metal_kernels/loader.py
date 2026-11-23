# SPDX-License-Identifier: Apache-2.0
"""Find Metal kernel sources shared with the Swift package.

Compilation is left to a later change. MLX (``mx.fast.metal_kernel``) and
PyTorch MPS (``torch.mps.compile_shader``) will take the string returned by
:func:`metal_source`. Neither backend is imported here.
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV_VAR = "METAL_KERNELS_ROOT"


class KernelNotFoundError(KeyError):
    """Raised when a name is not a registered kernel directory."""


def kernels_root() -> Path:
    """Return the ``kernels`` directory.

    ``METAL_KERNELS_ROOT`` wins when it is set. Otherwise a wheel or sdist
    install uses the ``kernels`` directory vendored next to this module. If
    that is missing, the loader walks up from this file looking for
    ``kernels/_template``, which an editable install from a checkout can see.
    """
    override = os.environ.get(_ENV_VAR, "").strip()
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"{_ENV_VAR}={override!r} is not a directory")
        return path

    packaged = Path(__file__).resolve().parent / "kernels"
    if packaged.is_dir() and any(_is_kernel_dir(child) for child in packaged.iterdir()):
        return packaged

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "kernels"
        if (candidate / "_template").is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not find the kernels directory. Reinstall metal-kernels, "
        "install it editable from a git checkout (pip install -e ./python), "
        f"or set {_ENV_VAR} to the kernels directory."
    )


def list_kernels() -> list[str]:
    """Names of registered kernel directories, in sorted order.

    Directories whose names start with ``_`` or ``.`` are skipped, including
    ``kernels/_template``. A directory counts only when it contains a
    ``.metal`` file.
    """
    names = [child.name for child in kernels_root().iterdir() if _is_kernel_dir(child)]
    return sorted(names)


def kernel_dir(name: str) -> Path:
    """Return the directory for one kernel."""
    _validate_name(name)
    directory = kernels_root() / name
    if not _is_kernel_dir(directory):
        raise KernelNotFoundError(name)
    return directory


def metal_source_path(name: str) -> Path:
    """Path of the single ``.metal`` file in the kernel directory."""
    directory = kernel_dir(name)
    sources = sorted(path for path in directory.glob("*.metal") if path.is_file())
    if len(sources) != 1:
        raise FileNotFoundError(
            f"Kernel {name!r} must contain exactly one .metal file; found {len(sources)}."
        )
    return sources[0]


def metal_source(name: str) -> str:
    """UTF-8 text of the kernel's ``.metal`` file."""
    return metal_source_path(name).read_text(encoding="utf-8")


def _validate_name(name: str) -> None:
    if (
        not name
        or name.startswith(("_", "."))
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or "\x00" in name
    ):
        raise KernelNotFoundError(name)


def _is_kernel_dir(path: Path) -> bool:
    if not path.is_dir() or path.name.startswith(("_", ".")):
        return False
    return any(child.is_file() and child.suffix == ".metal" for child in path.iterdir())
