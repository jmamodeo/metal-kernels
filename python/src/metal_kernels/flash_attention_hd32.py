# SPDX-License-Identifier: Apache-2.0
"""Launch the FlashAttention hd32 kernel from MLX or PyTorch MPS.

Both launchers compile ``kernels/flash_attention_hd32/flash_attention_hd32.metal``
on first use. MLX and PyTorch are imported only inside the launcher that needs
them, so importing this module needs neither.

Version 0 is the simple per-row online-softmax kernel: correct, not tuned.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence

from metal_kernels.loader import metal_source

KERNEL_NAME = "flash_attention_hd32"
HEAD_DIM = 32
THREADGROUP_SIZE = 256

# Buffer offsets in the shader are 32-bit, and MLX grids are 32-bit ints.
_MAX_ELEMENTS = 2**31 - 1

_MLX_BODY = """
    flash_attention_hd32_row(
        reinterpret_cast<const device ushort*>(q),
        reinterpret_cast<const device ushort*>(k),
        reinterpret_cast<const device ushort*>(v),
        reinterpret_cast<device ushort*>(out),
        uint(dims[0]),
        uint(dims[1]),
        uint(dims[2]),
        scale[0],
        thread_position_in_grid.x);
"""


def launch_geometry(
    q_shape: Sequence[int],
    k_shape: Sequence[int],
    v_shape: Sequence[int],
) -> tuple[tuple[int, int, int], int, int]:
    """Validate shapes and return ``(dims, threads, threadgroup)``.

    ``q`` is ``(batch, heads, query_length, 32)``; ``k`` and ``v`` are
    ``(batch, heads, key_length, 32)``. ``dims`` is
    ``(batch * heads * query_length, query_length, key_length)``, the values
    the shader reads. ``threads`` is the 1-D grid size in threads (32 per
    query row) and ``threadgroup`` the threadgroup size in threads.
    """
    q_shape, k_shape, v_shape = tuple(q_shape), tuple(k_shape), tuple(v_shape)
    if len(q_shape) != 4 or len(k_shape) != 4 or len(v_shape) != 4:
        raise ValueError("q, k, and v must have shape (batch, heads, sequence, 32)")
    batch, heads, query_length, head_dim = q_shape
    if head_dim != HEAD_DIM or k_shape[3] != HEAD_DIM or v_shape[3] != HEAD_DIM:
        raise ValueError(f"q, k, and v must have head dimension {HEAD_DIM}")
    if k_shape[:2] != (batch, heads) or v_shape[:2] != (batch, heads):
        raise ValueError("q, k, and v batch and head dimensions must match")
    if k_shape[2] != v_shape[2]:
        raise ValueError("k and v sequence lengths must match")
    key_length = k_shape[2]
    if min(batch, heads, query_length, key_length) < 1:
        raise ValueError("batch, heads, and sequence lengths must be positive")

    rows = batch * heads * query_length
    threads = rows * HEAD_DIM
    if max(threads, batch * heads * key_length * HEAD_DIM) > _MAX_ELEMENTS:
        raise ValueError(f"inputs must have at most {_MAX_ELEMENTS} elements")
    threadgroup = min(THREADGROUP_SIZE, threads)
    return (rows, query_length, key_length), threads, threadgroup


def resolve_scale(scale: float | None) -> float:
    """Return ``scale``, defaulting to ``1 / sqrt(32)`` like the reference."""
    if scale is None:
        return 1.0 / math.sqrt(HEAD_DIM)
    scale = float(scale)
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")
    return scale


def flash_attention_hd32_mlx(q, k, v, scale: float | None = None):
    """``softmax(q kᵀ * scale) v`` for bfloat16 ``mlx.core.array`` inputs.

    Returns a bfloat16 array shaped like ``q``.
    """
    import mlx.core as mx

    for name, value in (("q", q), ("k", k), ("v", v)):
        if value.dtype != mx.bfloat16:
            raise ValueError(f"{name} must have dtype bfloat16; got {value.dtype}")
    dims, threads, threadgroup = launch_geometry(q.shape, k.shape, v.shape)
    scale = resolve_scale(scale)

    (out,) = _mlx_kernel()(
        inputs=[
            q,
            k,
            v,
            mx.array(dims, dtype=mx.int32),
            mx.array([scale], dtype=mx.float32),
        ],
        output_shapes=[q.shape],
        output_dtypes=[mx.bfloat16],
        grid=(threads, 1, 1),
        threadgroup=(threadgroup, 1, 1),
    )
    return out


def flash_attention_hd32_mps(q, k, v, scale: float | None = None):
    """``softmax(q kᵀ * scale) v`` for bfloat16 ``torch.Tensor`` inputs on MPS.

    Returns a contiguous bfloat16 tensor shaped like ``q``.
    """
    import torch

    for name, value in (("q", q), ("k", k), ("v", v)):
        if value.dtype != torch.bfloat16:
            raise ValueError(f"{name} must have dtype bfloat16; got {value.dtype}")
        if value.device.type != "mps":
            raise ValueError(f"{name} must be on the mps device; got {value.device}")
    dims, threads, threadgroup = launch_geometry(q.shape, k.shape, v.shape)
    scale = resolve_scale(scale)

    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = torch.empty_like(q)
    getattr(_mps_library(), KERNEL_NAME)(
        q,
        k,
        v,
        out,
        torch.tensor(dims, dtype=torch.int32, device=q.device),
        torch.tensor([scale], dtype=torch.float32, device=q.device),
        threads=threads,
        group_size=threadgroup,
    )
    return out


@functools.lru_cache(maxsize=1)
def _mlx_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name=KERNEL_NAME,
        input_names=["q", "k", "v", "dims", "scale"],
        output_names=["out"],
        source=_MLX_BODY,
        header=metal_source(KERNEL_NAME),
        ensure_row_contiguous=True,
    )


@functools.lru_cache(maxsize=1)
def _mps_library():
    import torch

    return torch.mps.compile_shader(metal_source(KERNEL_NAME))
