# SPDX-License-Identifier: Apache-2.0
"""NumPy reference checks for scaled dot-product attention; no Metal required."""

from __future__ import annotations

import numpy as np
import pytest

from reference.sdpa import sdpa_reference


@pytest.mark.parametrize(
    ("shape", "use_bias"),
    [
        ((1, 1, 1, 1, 4), False),
        ((1, 1, 1, 1, 4), True),
        ((1, 2, 5, 7, 8), False),
        ((1, 2, 5, 7, 8), True),
        ((2, 3, 4, 6, 32), False),
        ((2, 3, 4, 6, 32), True),
    ],
)
def test_matches_naive_loop(shape, use_bias):
    batch, heads, query_length, key_length, head_dim = shape
    rng = np.random.default_rng(sum(shape) + use_bias)
    q, k, v = _random_qkv(rng, batch, heads, query_length, key_length, head_dim)
    bias = (
        rng.standard_normal((batch, heads, query_length, key_length), dtype=np.float32)
        if use_bias
        else None
    )

    actual = sdpa_reference(q, k, v, bias)
    expected = _sdpa_naive(q, k, v, bias, None)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_matches_naive_loop_with_explicit_scale():
    rng = np.random.default_rng(30)
    q, k, v = _random_qkv(rng, 1, 2, 3, 5, 8)
    bias = rng.standard_normal((1, 2, 3, 5), dtype=np.float32)

    actual = sdpa_reference(q, k, v, bias, scale=0.3)
    expected = _sdpa_naive(q, k, v, bias, scale=0.3)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_broadcast_bias_matches_naive_loop():
    rng = np.random.default_rng(44)
    q, k, v = _random_qkv(rng, 2, 2, 3, 5, 8)
    bias = rng.standard_normal((1, 1, 3, 5), dtype=np.float32)

    actual = sdpa_reference(q, k, v, bias)
    expected = _sdpa_naive(q, k, v, bias, None)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_large_scores_remain_finite_and_match_naive_loop():
    q = np.full((1, 1, 2, 32), 30, dtype=np.float32)
    k = np.full((1, 1, 3, 32), 30, dtype=np.float32)
    k[:, :, 1, :] = -30
    v = np.arange(96, dtype=np.float32).reshape(1, 1, 3, 32)
    bias = np.array([[[[1e4, -1e4, 1e4], [-1e4, 1e4, -1e4]]]], dtype=np.float32)

    actual = sdpa_reference(q, k, v, bias)
    expected = _sdpa_naive(q, k, v, bias, None)

    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_single_key_attention_returns_the_value():
    q = np.zeros((1, 2, 3, 32), dtype=np.float32)
    k = np.ones((1, 2, 1, 32), dtype=np.float32)
    v = np.arange(64, dtype=np.float32).reshape(1, 2, 1, 32)

    actual = sdpa_reference(q, k, v)

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

    actual = sdpa_reference(q, k, v)

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

    actual = sdpa_reference(q, k, v, bias)
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

    actual = sdpa_reference(q, k, v, bias)

    np.testing.assert_array_equal(actual, np.zeros_like(q))


def test_rejects_incompatible_shapes_and_dtypes():
    q = np.zeros((1, 1, 2, 32), dtype=np.float32)
    k = np.zeros((1, 1, 3, 32), dtype=np.float32)
    v = np.zeros((1, 1, 3, 32), dtype=np.float32)

    with pytest.raises(ValueError, match="same positive head dimension"):
        sdpa_reference(q[..., :16], k, v)
    with pytest.raises(ValueError, match="k and v sequence lengths"):
        sdpa_reference(q, k, v[:, :, :2])
    with pytest.raises(ValueError, match="broadcastable"):
        sdpa_reference(q, k, v, np.zeros((4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="float32"):
        sdpa_reference(q.astype(np.float64), k, v)


def _random_qkv(rng, batch, heads, query_length, key_length, head_dim):
    q = rng.standard_normal((batch, heads, query_length, head_dim), dtype=np.float32)
    k = rng.standard_normal((batch, heads, key_length, head_dim), dtype=np.float32)
    v = rng.standard_normal((batch, heads, key_length, head_dim), dtype=np.float32)
    return q, k, v


def _sdpa_naive(q, k, v, bias, scale):
    batch, heads, query_length, head_dim = q.shape
    key_length = k.shape[2]
    if scale is None:
        scale = 1.0 / np.sqrt(head_dim)
    broadcast_bias = (
        np.broadcast_to(bias, (batch, heads, query_length, key_length))
        if bias is not None
        else None
    )
    output = np.zeros_like(q)

    for b in range(batch):
        for h in range(heads):
            for i in range(query_length):
                scores = np.empty(key_length, dtype=np.float32)
                for j in range(key_length):
                    scores[j] = np.dot(q[b, h, i], k[b, h, j]) * scale
                    if broadcast_bias is not None:
                        scores[j] += broadcast_bias[b, h, i, j]
                scores -= np.max(scores)
                probabilities = np.exp(scores)
                probabilities /= np.sum(probabilities)
                for j in range(key_length):
                    output[b, h, i] += probabilities[j] * v[b, h, j]
    return output
