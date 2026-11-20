# SPDX-License-Identifier: Apache-2.0
"""NumPy reference checks for head-dimension-32 attention; no Metal required."""

from __future__ import annotations

import numpy as np
import pytest

from reference.flash_attention import flash_attention


def test_single_key_attention_returns_the_value():
    q = np.zeros((1, 2, 3, 32), dtype=np.float32)
    k = np.ones((1, 2, 1, 32), dtype=np.float32)
    v = np.arange(64, dtype=np.float32).reshape(1, 2, 1, 32)

    actual = flash_attention(q, k, v)

    assert actual.shape == q.shape
    assert actual.dtype == np.float32
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, np.broadcast_to(v, q.shape))


def test_scaled_dot_product_and_softmax_are_applied():
    q = np.zeros((1, 1, 1, 32), dtype=np.float32)
    k = np.zeros((1, 1, 2, 32), dtype=np.float32)
    v = np.zeros((1, 1, 2, 32), dtype=np.float32)
    q[..., 0] = np.sqrt(np.float32(32))
    k[..., 0, 0] = 1
    v[..., 0, 0] = 1

    actual = flash_attention(q, k, v)

    np.testing.assert_allclose(actual[..., 0], 1 / (1 + np.exp(-1)), rtol=1e-6)
    np.testing.assert_array_equal(actual[..., 1:], 0)


def test_matches_torch_sdpa_with_additive_bias_on_cpu():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as functional

    rng = np.random.default_rng(32)
    q = rng.standard_normal((1, 2, 5, 32), dtype=np.float32)
    k = rng.standard_normal((1, 2, 7, 32), dtype=np.float32)
    v = rng.standard_normal((1, 2, 7, 32), dtype=np.float32)
    bias = rng.standard_normal((1, 2, 5, 7), dtype=np.float32)

    actual = flash_attention(q, k, v, bias)
    expected = functional.scaled_dot_product_attention(
        torch.from_numpy(q),
        torch.from_numpy(k),
        torch.from_numpy(v),
        attn_mask=torch.from_numpy(bias),
    ).numpy()

    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


def test_all_masked_rows_produce_zero_output():
    q = np.ones((1, 1, 2, 32), dtype=np.float32)
    k = np.ones((1, 1, 3, 32), dtype=np.float32)
    v = np.ones((1, 1, 3, 32), dtype=np.float32)
    bias = np.full((1, 1, 2, 3), -np.inf, dtype=np.float32)

    actual = flash_attention(q, k, v, bias)

    np.testing.assert_array_equal(actual, np.zeros_like(q))


def test_rejects_incompatible_shapes_and_dtypes():
    q = np.zeros((1, 1, 2, 32), dtype=np.float32)
    k = np.zeros((1, 1, 3, 32), dtype=np.float32)
    v = np.zeros((1, 1, 3, 32), dtype=np.float32)

    with pytest.raises(ValueError, match="head dimension 32"):
        flash_attention(q[..., :16], k[..., :16], v[..., :16])
    with pytest.raises(ValueError, match="k and v sequence lengths"):
        flash_attention(q, k, v[:, :, :2])
    with pytest.raises(ValueError, match="broadcastable"):
        flash_attention(q, k, v, np.zeros((4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="float32"):
        flash_attention(q.astype(np.float64), k, v)
