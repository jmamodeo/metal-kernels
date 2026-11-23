# SPDX-License-Identifier: Apache-2.0
"""FlashAttention for head dim 32, launched from MLX or PyTorch MPS.

:func:`flash_attention_hd32` compiles
``kernels/flash_attention_hd32/flash_attention_hd32.metal`` on first use.
MLX and PyTorch are imported only by the backend that runs, so importing this
module needs neither.

Version 6 is the tiled ``simdgroup_matrix`` kernel from the design note:
one threadgroup of four simdgroups per 32 query rows of one (batch, head),
K/V walked in 64-key blocks that the whole threadgroup copies once into
threadgroup memory, in the input dtype, and the four simdgroups share.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence

from metal_kernels.loader import metal_source

KERNEL_NAME = "flash_attention_hd32"
HEAD_DIM = 32
# Mirrors FA_HD32_BLOCK_Q, FA_HD32_BLOCK_K and FA_HD32_THREADS in the shader:
# 8 query rows per simdgroup, 32 threads per simdgroup.
QUERY_BLOCK = 32
KEY_BLOCK = 64
THREADGROUP_SIZE = QUERY_BLOCK // 8 * 32
# One 64-key K block and one V block staged in bf16 (fp32 inputs stage twice
# this). Mirrors FA_HD32_KV_ELEMENTS in the shader.
THREADGROUP_MEMORY_BYTES = 2 * KEY_BLOCK * HEAD_DIM * 2
BACKENDS = ("mlx", "mps")

# Buffer offsets in the shader are 32-bit, and MLX grids are 32-bit ints.
_MAX_ELEMENTS = 2**31 - 1

# The MLX-generated kernel declares the threadgroup K/V stage and calls the
# block function in the shared source.
_MLX_BODY = """
    FA_HD32_KV_STAGE(kv, fa_hd32_bits(k));
    flash_attention_hd32_block(
        fa_hd32_bits(q),
        fa_hd32_bits(k),
        fa_hd32_bits(v),
        fa_hd32_bits(bias),
        fa_hd32_bits(out),
        kv,
        fa_hd32_params(dims, scale[0]),
        threadgroup_position_in_grid.x,
        thread_index_in_threadgroup);
"""


def flash_attention_hd32(q, k, v, bias=None, scale=None, *, backend):
    """Compute ``softmax(q kᵀ * scale + bias) v`` on the GPU.

    ``q`` is ``(batch, heads, query_length, 32)``; ``k`` and ``v`` are
    ``(batch, heads, key_length, 32)``. All inputs share one dtype, bfloat16
    or float32, and the output has that dtype and the shape of ``q``.

    ``bias`` is optional, additive, applied after scaling, and broadcastable
    to ``(batch, heads, query_length, key_length)``. A ``-inf`` bias masks a
    key; a query row with every key masked returns zeros. ``scale`` defaults
    to ``1 / sqrt(32)``.

    ``backend`` is ``"mlx"`` for ``mlx.core.array`` inputs or ``"mps"`` for
    ``torch.Tensor`` inputs on the ``mps`` device.
    """
    if backend == "mlx":
        return _flash_attention_hd32_mlx(q, k, v, bias, scale)
    if backend == "mps":
        return _flash_attention_hd32_mps(q, k, v, bias, scale)
    raise ValueError(f"backend must be one of {BACKENDS}; got {backend!r}")


def launch_geometry(
    q_shape: Sequence[int],
    k_shape: Sequence[int],
    v_shape: Sequence[int],
) -> tuple[tuple[int, int, int, int], int, int]:
    """Validate shapes and return ``(dims, threads, threadgroup)``.

    ``dims`` is ``(batch * heads * query_length, heads, query_length,
    key_length)``. ``threads`` is the 1-D grid size in threads: one
    threadgroup of ``THREADGROUP_SIZE`` threads per ``QUERY_BLOCK`` query
    rows of each (batch, head). ``threadgroup`` is the threadgroup size.
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
    if max(rows, batch * heads * key_length) * HEAD_DIM > _MAX_ELEMENTS:
        raise ValueError(f"inputs must have at most {_MAX_ELEMENTS} elements")
    groups = batch * heads * -(-query_length // QUERY_BLOCK)
    threads = groups * THREADGROUP_SIZE
    if threads > _MAX_ELEMENTS:
        raise ValueError(f"the launch grid must have at most {_MAX_ELEMENTS} threads")
    return (rows, heads, query_length, key_length), threads, THREADGROUP_SIZE


def bias_strides(
    bias_shape: Sequence[int], target_shape: Sequence[int]
) -> tuple[tuple[int, ...], tuple[int, int, int, int]]:
    """Validate a bias shape and return ``(shape4, strides)``.

    ``target_shape`` is ``(batch, heads, query_length, key_length)``.
    ``shape4`` is ``bias_shape`` left-padded with ones to four dimensions.
    ``strides`` are the element strides of a contiguous bias of that shape,
    with 0 on broadcast dimensions.
    """
    bias_shape, target_shape = tuple(bias_shape), tuple(target_shape)
    message = f"bias must be broadcastable to {target_shape}; got {bias_shape}"
    if len(bias_shape) > 4:
        raise ValueError(message)
    shape4 = (1,) * (4 - len(bias_shape)) + bias_shape
    if any(size not in (1, target) for size, target in zip(shape4, target_shape)):
        raise ValueError(message)
    if math.prod(shape4) > _MAX_ELEMENTS:
        raise ValueError(f"bias must have at most {_MAX_ELEMENTS} elements")
    strides = []
    stride = 1
    for size, target in reversed(tuple(zip(shape4, target_shape))):
        strides.append(stride if size == target and target > 1 else 0)
        stride *= size
    return shape4, tuple(reversed(strides))


def resolve_scale(scale: float | None) -> float:
    """Return ``scale``, defaulting to ``1 / sqrt(32)`` like the reference."""
    if scale is None:
        return 1.0 / math.sqrt(HEAD_DIM)
    scale = float(scale)
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")
    return scale


def _dims(geometry_dims, strides, has_bias: bool) -> list[int]:
    rows, heads, query_length, key_length = geometry_dims
    return [rows, heads, query_length, key_length, int(has_bias), *strides]


def _check_dtypes(names_and_dtypes, allowed, expected=None):
    for name, dtype in names_and_dtypes:
        if dtype not in allowed:
            raise ValueError(f"{name} must have dtype bfloat16 or float32; got {dtype}")
        if expected is not None and dtype != expected:
            raise ValueError(f"{name} must have the same dtype as q; got {dtype}")
        expected = dtype


def _flash_attention_hd32_mlx(q, k, v, bias, scale):
    import mlx.core as mx

    tensors = [("q", q), ("k", k), ("v", v)] + ([("bias", bias)] if bias is not None else [])
    _check_dtypes([(name, x.dtype) for name, x in tensors], (mx.bfloat16, mx.float32))
    geometry, threads, threadgroup = launch_geometry(q.shape, k.shape, v.shape)
    scale = resolve_scale(scale)
    target = (q.shape[0], q.shape[1], q.shape[2], k.shape[2])
    has_bias = bias is not None
    if has_bias:
        shape4, strides = bias_strides(bias.shape, target)
        bias = bias.reshape(shape4)
    else:
        strides = (0, 0, 0, 0)
    dims, scale_array, placeholder = _mlx_constants(
        tuple(_dims(geometry, strides, has_bias)), scale, q.dtype
    )

    (out,) = _mlx_kernel()(
        inputs=[q, k, v, bias if has_bias else placeholder, dims, scale_array],
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
        grid=(threads, 1, 1),
        threadgroup=(threadgroup, 1, 1),
    )
    return out


def _flash_attention_hd32_mps(q, k, v, bias, scale):
    import torch

    tensors = [("q", q), ("k", k), ("v", v)] + ([("bias", bias)] if bias is not None else [])
    _check_dtypes([(name, x.dtype) for name, x in tensors], (torch.bfloat16, torch.float32))
    for name, value in tensors:
        if value.device.type != "mps":
            raise ValueError(f"{name} must be on the mps device; got {value.device}")
    geometry, threads, threadgroup = launch_geometry(q.shape, k.shape, v.shape)
    scale = resolve_scale(scale)
    target = (q.shape[0], q.shape[1], q.shape[2], k.shape[2])
    has_bias = bias is not None
    if has_bias:
        shape4, strides = bias_strides(bias.shape, target)
        bias = _contiguous(bias.reshape(shape4))
    else:
        strides = (0, 0, 0, 0)
    dims, scale_tensor, placeholder = _mps_constants(
        tuple(_dims(geometry, strides, has_bias)), scale, q.dtype, q.device
    )

    q, k, v = _contiguous(q), _contiguous(k), _contiguous(v)
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    name = "bfloat" if q.dtype == torch.bfloat16 else "float"
    getattr(_mps_library(), f"{KERNEL_NAME}_{name}")(
        q,
        k,
        v,
        bias if has_bias else placeholder,
        out,
        dims,
        scale_tensor,
        threads=threads,
        group_size=threadgroup,
    )
    return out


def _contiguous(tensor):
    return tensor if tensor.is_contiguous() else tensor.contiguous()


# The `dims` and `scale` buffers and the unused-bias placeholder depend only
# on the launch, so repeated calls at one shape reuse them instead of
# allocating on every launch. The kernel never writes them.
@functools.lru_cache(maxsize=32)
def _mlx_constants(dims: tuple[int, ...], scale: float, dtype):
    import mlx.core as mx

    constants = (
        mx.array(dims, dtype=mx.int32),
        mx.array([scale], dtype=mx.float32),
        mx.zeros((1,), dtype=dtype),
    )
    mx.eval(*constants)
    return constants


@functools.lru_cache(maxsize=32)
def _mps_constants(dims: tuple[int, ...], scale: float, dtype, device):
    import torch

    return (
        torch.tensor(dims, dtype=torch.int32, device=device),
        torch.tensor([scale], dtype=torch.float32, device=device),
        torch.zeros(1, dtype=dtype, device=device),
    )


@functools.lru_cache(maxsize=1)
def _mlx_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name=KERNEL_NAME,
        input_names=["q", "k", "v", "bias", "dims", "scale"],
        output_names=["out"],
        source=_MLX_BODY,
        header=metal_source(KERNEL_NAME),
        ensure_row_contiguous=True,
    )


@functools.lru_cache(maxsize=1)
def _mps_library():
    import torch

    return torch.mps.compile_shader(metal_source(KERNEL_NAME))
