# SPDX-License-Identifier: Apache-2.0
"""The built wheel must vendor the FlashAttention hd32 shader."""

from __future__ import annotations

import os
import subprocess
import sys
import venv
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"


def test_wheel_vendors_flash_attention_hd32_shader(tmp_path: Path) -> None:
    pytest.importorskip("build")

    dist_dir = tmp_path / "dist"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--outdir",
            str(dist_dir),
            str(PYTHON_DIR),
        ],
        check=True,
        cwd=tmp_path,
    )
    wheels = list(dist_dir.glob("metal_kernels-*.whl"))
    assert len(wheels) == 1, wheels

    venv_dir = tmp_path / "venv"
    venv.create(venv_dir, with_pip=True, symlinks=True)
    python = venv_dir / ("Scripts" if sys.platform == "win32" else "bin") / "python"

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.pop("METAL_KERNELS_ROOT", None)

    subprocess.run(
        [str(python), "-m", "pip", "install", "--upgrade", "pip"],
        check=True,
        env=env,
        cwd=tmp_path,
    )
    subprocess.run(
        [str(python), "-m", "pip", "install", str(wheels[0])],
        check=True,
        env=env,
        cwd=tmp_path,
    )

    script = r"""
from pathlib import Path

import metal_kernels

assert metal_kernels.__version__ == "0.1.1"
assert metal_kernels.list_kernels() == ["flash_attention_hd32"]
source = metal_kernels.metal_source("flash_attention_hd32")
assert "SPDX-License-Identifier: Apache-2.0" in source
assert "flash_attention_hd32" in source
root = metal_kernels.kernels_root()
package_dir = Path(metal_kernels.__file__).resolve().parent
assert root == package_dir / "kernels"
assert (root / "flash_attention_hd32" / "flash_attention_hd32.metal").is_file()
"""
    subprocess.run(
        [str(python), "-c", script],
        check=True,
        env=env,
        cwd=tmp_path,
    )
