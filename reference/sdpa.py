# SPDX-License-Identifier: Apache-2.0
"""NumPy reference for scaled dot-product attention."""

from __future__ import annotations

import numpy as np


def sdpa_reference(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    bias: np.ndarray | None = None,
    scale: float | None = None,
) -> np.ndarray:
    """Compute ``softmax(Q Kᵀ * scale + bias) V``.

    Inputs use ``(batch, heads, sequence, head_dim)`` layout and float32 dtype.
    ``bias``, when provided, must be float32 and broadcastable to
    ``(batch, heads, query sequence, key sequence)``.
    If ``scale`` is omitted, it defaults to ``1 / sqrt(head_dim)``.
    """
    q = _as_float32(q, "q")
    k = _as_float32(k, "k")
    v = _as_float32(v, "v")

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape (batch, heads, sequence, head_dim)")
    batch, heads, query_length, head_dim = q.shape
    key_batch, key_heads, key_length, key_dim = k.shape
    value_batch, value_heads, value_length, value_dim = v.shape
    if head_dim < 1 or key_dim != head_dim or value_dim != head_dim:
        raise ValueError("q, k, and v must have the same positive head dimension")
    if (key_batch, key_heads) != (batch, heads):
        raise ValueError("q and k batch and head dimensions must match")
    if (value_batch, value_heads) != (batch, heads):
        raise ValueError("q and v batch and head dimensions must match")
    if key_length != value_length:
        raise ValueError("k and v sequence lengths must match")
    if min(batch, heads, query_length, key_length) < 1:
        raise ValueError("batch, heads, and sequence lengths must be positive")

    scores = np.matmul(q, np.swapaxes(k, -1, -2))
    if scale is None:
        scale = 1.0 / np.sqrt(head_dim)
    scores *= scale
    if bias is not None:
        bias = _as_float32(bias, "bias")
        try:
            scores = scores + bias
        except ValueError as exc:
            raise ValueError(
                f"bias must be broadcastable to {(batch, heads, query_length, key_length)}"
            ) from exc

    maximum = np.max(scores, axis=-1, keepdims=True)
    shifted = np.full_like(scores, -np.inf)
    np.subtract(scores, maximum, out=shifted, where=np.isfinite(maximum))
    probabilities = np.exp(shifted)
    denominator = np.sum(probabilities, axis=-1, keepdims=True)
    np.divide(
        probabilities,
        denominator,
        out=probabilities,
        where=denominator != 0,
    )
    return np.ascontiguousarray(np.matmul(probabilities, v), dtype=np.float32)


def _as_float32(value: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray; got {type(value).__name__}")
    if value.dtype != np.float32:
        raise ValueError(f"{name} must have dtype float32; got {value.dtype}")
    return value
