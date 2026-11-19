# SPDX-License-Identifier: Apache-2.0
"""NumPy reference for conv3x3. No Metal launch."""

from __future__ import annotations

import numpy as np
import pytest

from reference.conv3x3 import (
    SAM3_FLOPS,
    SAM3_IM2COL_BYTES,
    SAM3_INPUT_SHAPE,
    SAM3_TENSOR_BYTES,
    SAM3_WEIGHT_SHAPE,
    conv3x3_flops,
    conv3x3_im2col,
    conv3x3_im2col_bytes,
    conv3x3_naive,
    conv3x3_tensor_bytes,
)

_RTOL = 1e-4
_ATOL = 1e-4


def test_sam3_accounting_matches_the_inventory():
    n, cin, height, width = SAM3_INPUT_SHAPE
    cout = SAM3_WEIGHT_SHAPE[0]
    assert conv3x3_flops(n, cin, cout, height, width) == SAM3_FLOPS
    assert SAM3_FLOPS == 2 * 256 * 256 * 9 * 288 * 288
    assert f"{SAM3_FLOPS:.3e}" == "9.784e+10"
    assert conv3x3_tensor_bytes(n, cin, cout, height, width) == SAM3_TENSOR_BYTES
    assert conv3x3_im2col_bytes(n, cin, height, width) == SAM3_IM2COL_BYTES


def test_naive_matches_im2col_on_small_and_odd_shapes():
    cases = [
        (1, 1, 2, 1, 1),
        (1, 3, 5, 5, 5),
        (2, 4, 4, 7, 7),
        (1, 8, 1, 7, 5),
    ]
    for n, cin, cout, height, width in cases:
        x, weight, bias = _inputs(n, cin, cout, height, width, seed=height + 10 * cin)
        naive = conv3x3_naive(x, weight, bias)
        lowered = conv3x3_im2col(x, weight, bias)
        assert naive.shape == (n, cout, height, width)
        assert naive.dtype == np.float32
        assert lowered.dtype == np.float32
        assert lowered.flags.c_contiguous
        np.testing.assert_allclose(naive, lowered, rtol=_RTOL, atol=_ATOL)


def test_matches_torch_conv2d_when_installed():
    torch_conv = _torch_conv2d()
    if torch_conv is None:
        pytest.skip("torch is not installed")
    cases = [
        (1, 1, 2, 1, 1),
        (1, 3, 5, 5, 5),
        (2, 4, 4, 7, 7),
        (1, 2, 3, 288, 288),
    ]
    for n, cin, cout, height, width in cases:
        x, weight, bias = _inputs(n, cin, cout, height, width, seed=100 + height)
        got = conv3x3_im2col(x, weight, bias)
        expected = torch_conv(x, weight, bias)
        np.testing.assert_allclose(got, expected, rtol=_RTOL, atol=_ATOL)


def test_im2col_at_the_sam3_shape():
    """The real neck and mask-head shape. Naive is not run here.

    A few pixels are checked against an independent scalar dot product,
    including the padded corner. The full map is checked against CPU
    ``conv2d`` when PyTorch is installed.
    """
    n, cin, height, width = SAM3_INPUT_SHAPE
    cout = SAM3_WEIGHT_SHAPE[0]
    x, weight, bias = _inputs(n, cin, cout, height, width, seed=3)
    out = conv3x3_im2col(x, weight, bias)
    assert out.shape == (n, cout, height, width)
    assert out.dtype == np.float32

    for oc, y, xcoord in ((0, 0, 0), (7, 100, 200), (255, height - 1, width - 1)):
        expected = _explicit_pixel(x, weight, bias, n=0, oc=oc, y=y, xcoord=xcoord)
        np.testing.assert_allclose(out[0, oc, y, xcoord], expected, rtol=_RTOL, atol=_ATOL)

    torch_conv = _torch_conv2d()
    if torch_conv is None:
        return
    np.testing.assert_allclose(out, torch_conv(x, weight, bias), rtol=_RTOL, atol=_ATOL)


def test_rejects_bad_shapes_and_dtypes():
    x, weight, bias = _inputs(1, 2, 3, 5, 5, seed=1)
    with pytest.raises(ValueError, match="float32"):
        conv3x3_naive(x.astype(np.float64), weight, bias)
    with pytest.raises(ValueError, match="3x3"):
        conv3x3_im2col(x, np.zeros((3, 2, 1, 1), dtype=np.float32), bias)
    with pytest.raises(ValueError, match="input channels"):
        conv3x3_naive(x, np.zeros((3, 4, 3, 3), dtype=np.float32), bias)
    with pytest.raises(ValueError, match="bias"):
        conv3x3_im2col(x, weight, np.zeros((4,), dtype=np.float32))
    with pytest.raises(ValueError, match=r"\(N, Cin, H, W\)"):
        conv3x3_naive(x[0], weight, bias)
    with pytest.raises(TypeError, match="ndarray"):
        conv3x3_im2col(x.tolist(), weight, bias)


def _inputs(n, cin, cout, height, width, seed):
    rng = np.random.default_rng(seed)
    scale = np.float32(1.0 / np.sqrt(cin * 9))
    x = rng.standard_normal((n, cin, height, width), dtype=np.float32)
    weight = rng.standard_normal((cout, cin, 3, 3), dtype=np.float32) * scale
    bias = rng.standard_normal((cout,), dtype=np.float32)
    return x, weight, bias


def _explicit_pixel(x, weight, bias, n, oc, y, xcoord):
    """One output pixel, scalar taps, independent of both implementations."""
    padded = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    acc = np.float32(bias[oc])
    cin = x.shape[1]
    for ic in range(cin):
        for ky in range(3):
            for kx in range(3):
                acc += weight[oc, ic, ky, kx] * padded[n, ic, y + ky, xcoord + kx]
    return acc


def _torch_conv2d():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        return None

    def run(x, weight, bias):
        with torch.no_grad():
            result = F.conv2d(
                torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(np.ascontiguousarray(weight)),
                torch.from_numpy(np.ascontiguousarray(bias)),
                stride=1,
                padding=1,
                dilation=1,
                groups=1,
            )
        return result.numpy()

    return run
