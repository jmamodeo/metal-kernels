# SPDX-License-Identifier: Apache-2.0
"""Metal conv3x3 against the NumPy reference.

Tests marked ``metal`` launch the shader. conftest skips those on hosts that
are not macOS, so Linux CI does not need MLX or an MPS device. The launch
geometry checks below do not launch anything and always run.
"""

from __future__ import annotations

import numpy as np
import pytest
from metal_kernels import metal_source
from metal_kernels.conv3x3 import OC_PER_THREAD, TILE, conv3x3, launch_config

from reference.conv3x3 import conv3x3_im2col, conv3x3_naive

_RTOL = 1e-4
_ATOL = 1e-4

# (N, Cin, Cout, H, W). Includes a 1x1, odd sizes, Cin != Cout, an exact
# 8x8 tile, and a tile that hangs off the right and bottom edges.
_CASES = (
    (1, 1, 1, 1, 1),
    (1, 1, 2, 1, 1),
    (1, 3, 5, 5, 5),
    (2, 4, 4, 7, 7),
    (1, 8, 1, 7, 5),
    (1, 2, 3, 8, 8),
    (1, 4, 6, 9, 10),
)


def test_shader_names_the_kernel_and_the_tile():
    source = metal_source("conv3x3")
    assert "kernel void conv3x3" in source
    assert "conv3x3_thread" in source
    assert f"CONV3X3_TILE = {TILE}" in source
    assert f"CONV3X3_OC_PER_THREAD = {OC_PER_THREAD}" in source


def test_launch_covers_partial_tiles():
    threads, group = launch_config(2, 5, 9, 10)
    assert group == (TILE, TILE, 1)
    assert threads[0] % TILE == 0 and threads[0] >= 10
    assert threads[1] % TILE == 0 and threads[1] >= 9
    # Cout 5 is two blocks of four, times batch 2.
    assert threads[2] == 2 * 2

    threads, group = launch_config(1, 256, 288, 288)
    assert group == (TILE, TILE, 1)
    assert threads == (288, 288, 64)


@pytest.mark.metal
def test_metal_matches_numpy_reference():
    mlx_ready = _mlx_ready()
    mps_ready = _mps_ready()
    if not mlx_ready and not mps_ready:
        pytest.skip("neither MLX nor PyTorch MPS is available")

    for n, cin, cout, height, width in _CASES:
        x, weight, bias = _inputs(n, cin, cout, height, width, seed=height + 17 * cin)
        naive = conv3x3_naive(x, weight, bias)
        lowered = conv3x3_im2col(x, weight, bias)
        if mlx_ready:
            got = _run_mlx(x, weight, bias)
            np.testing.assert_allclose(got, naive, rtol=_RTOL, atol=_ATOL)
            np.testing.assert_allclose(got, lowered, rtol=_RTOL, atol=_ATOL)
        if mps_ready:
            got = _run_mps(x, weight, bias)
            np.testing.assert_allclose(got, naive, rtol=_RTOL, atol=_ATOL)
            np.testing.assert_allclose(got, lowered, rtol=_RTOL, atol=_ATOL)


def _inputs(n, cin, cout, height, width, seed):
    rng = np.random.default_rng(seed)
    scale = np.float32(1.0 / np.sqrt(cin * 9))
    x = rng.standard_normal((n, cin, height, width), dtype=np.float32)
    weight = rng.standard_normal((cout, cin, 3, 3), dtype=np.float32) * scale
    bias = rng.standard_normal((cout,), dtype=np.float32)
    return x, weight, bias


def _mlx_ready() -> bool:
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        return False
    return True


def _mps_ready() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.backends.mps.is_available())


def _run_mlx(x, weight, bias):
    import mlx.core as mx

    out = conv3x3(
        mx.array(x),
        mx.array(weight),
        mx.array(bias),
    )
    mx.eval(out)
    return np.array(out)


def _run_mps(x, weight, bias):
    import torch

    out = conv3x3(
        torch.from_numpy(x).to("mps"),
        torch.from_numpy(weight).to("mps"),
        torch.from_numpy(bias).to("mps"),
    )
    torch.mps.synchronize()
    return out.cpu().numpy()
