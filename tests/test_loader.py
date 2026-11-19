# SPDX-License-Identifier: Apache-2.0
"""Loader tests. These do not compile or launch Metal."""

from __future__ import annotations

import subprocess
import sys

import pytest

from metal_kernels import (
    KernelNotFoundError,
    host_is_macos,
    kernel_dir,
    kernels_root,
    list_kernels,
    metal_source,
)
from metal_kernels.__main__ import main


def test_import_does_not_load_gpu_frameworks():
    # A fresh interpreter. Other tests may import PyTorch in this process.
    script = (
        "import metal_kernels, sys; "
        "assert 'torch' not in sys.modules; "
        "assert 'mlx' not in sys.modules; "
        "assert 'sam3' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_host_flag_matches_platform():
    assert host_is_macos() is (sys.platform == "darwin")


def test_checkout_registers_conv3x3_without_a_kernel_function(monkeypatch):
    monkeypatch.delenv("METAL_KERNELS_ROOT", raising=False)
    root = kernels_root()
    template = (root / "_template" / "kernel.metal").read_text(encoding="utf-8")
    assert "metal-kernels-template: not a kernel" in template
    assert list_kernels() == ["conv3x3"]
    source = metal_source("conv3x3")
    assert "kernel void" not in source
    assert kernel_dir("conv3x3").name == "conv3x3"
    with pytest.raises(KernelNotFoundError):
        kernel_dir("_template")


def test_module_main_lists_conv3x3(monkeypatch, capsys):
    monkeypatch.delenv("METAL_KERNELS_ROOT", raising=False)
    assert main() == 0
    captured = capsys.readouterr()
    assert "conv3x3" in captured.out
    assert "No kernels registered." not in captured.out


def test_missing_environment_root(monkeypatch):
    monkeypatch.setenv("METAL_KERNELS_ROOT", "/no/such/metal-kernels-root")
    with pytest.raises(FileNotFoundError):
        kernels_root()
    assert main() == 1


def test_discovers_one_metal_file_and_skips_drafts(tmp_path, monkeypatch):
    kernel = tmp_path / "scale"
    kernel.mkdir()
    (kernel / "scale.metal").write_text("// scale\n", encoding="utf-8")
    (kernel / "README.md").write_text("docs\n", encoding="utf-8")
    draft = tmp_path / "_draft"
    draft.mkdir()
    (draft / "draft.metal").write_text("// no\n", encoding="utf-8")
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "README.md").write_text("no shader\n", encoding="utf-8")
    monkeypatch.setenv("METAL_KERNELS_ROOT", str(tmp_path))

    assert list_kernels() == ["scale"]
    assert metal_source("scale") == "// scale\n"
    with pytest.raises(KernelNotFoundError):
        kernel_dir("_draft")
    with pytest.raises(KernelNotFoundError):
        kernel_dir("../scale")


def test_rejects_multiple_metal_files(tmp_path, monkeypatch):
    kernel = tmp_path / "scale"
    kernel.mkdir()
    (kernel / "a.metal").write_text("a", encoding="utf-8")
    (kernel / "b.metal").write_text("b", encoding="utf-8")
    monkeypatch.setenv("METAL_KERNELS_ROOT", str(tmp_path))
    assert list_kernels() == ["scale"]
    with pytest.raises(FileNotFoundError, match="exactly one"):
        metal_source("scale")
