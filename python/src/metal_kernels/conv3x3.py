# SPDX-License-Identifier: Apache-2.0
"""Launch the conv3x3 Metal kernel from MLX or PyTorch MPS.

Importing ``metal_kernels`` does not import this module, and this module
does not import MLX or PyTorch until a launch function runs.

The shader text is ``kernels/conv3x3/conv3x3.metal``. PyTorch compiles that
file and calls ``conv3x3``. MLX pastes the same file in front of a short
body that calls ``conv3x3_thread``, because ``mx.fast.metal_kernel`` generates
its own kernel signature. The arithmetic lives in the ``.metal`` file either way.
"""

from __future__ import annotations

from metal_kernels.loader import metal_source

# Must match the enums in kernels/conv3x3/conv3x3.metal.
TILE = 8
OC_PER_THREAD = 4

# MLX scans this string for attribute names and builds the kernel signature.
# The body is the call. The function it calls is in the .metal file.
_MLX_BODY = """
    threadgroup float tile[CONV3X3_TILE_PAD * CONV3X3_TILE_PAD];
    conv3x3_thread(
        x,
        weight,
        bias,
        out,
        tile,
        dims[0],
        dims[1],
        dims[2],
        dims[3],
        dims[4],
        thread_position_in_threadgroup,
        threadgroup_position_in_grid);
"""

_mlx_kernel = None
_mps_library = None


def launch_config(
    n: int, cout: int, height: int, width: int
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Total threads and threads-per-group for ``dispatchThreads``.

    The grid is rounded up to whole 8x8 tiles so the last group is full.
    Threads that land past the image take part in the loads and the barriers
    and do not write. ``z`` is ``N * ceil(Cout / 4)``: one group-step per
    batch item per block of four output channels.
    """
    tiles_x = (width + TILE - 1) // TILE
    tiles_y = (height + TILE - 1) // TILE
    oc_blocks = (cout + OC_PER_THREAD - 1) // OC_PER_THREAD
    threads = (tiles_x * TILE, tiles_y * TILE, n * oc_blocks)
    group = (TILE, TILE, 1)
    return threads, group


def conv3x3(x, weight, bias):
    """Run the Metal 3x3 on an MLX array or a PyTorch MPS tensor.

    Layout matches the NumPy reference: ``x`` is NCHW, ``weight`` is OIHW,
    ``bias`` is ``[Cout]``, all float32. Stride, padding, dilation, and groups
    are fixed at 1, 1, 1, and 1.
    """
    module = type(x).__module__
    if module.startswith("mlx"):
        return conv3x3_mlx(x, weight, bias)
    if module.startswith("torch"):
        return conv3x3_mps(x, weight, bias)
    raise TypeError(
        "conv3x3 expects an MLX array or a PyTorch tensor, "
        f"got {type(x).__module__}.{type(x).__name__}"
    )


def conv3x3_mlx(x, weight, bias):
    """Launch via ``mx.fast.metal_kernel``."""
    import mlx.core as mx

    _check(x, weight, bias, float_dtype=mx.float32)
    n, cin, height, width = (int(v) for v in x.shape)
    cout = int(weight.shape[0])
    # MLX puts an input with fewer than 8 elements in the constant address
    # space. The shader takes device pointers, so tiny maps are copied into
    # an 8-element buffer. The kernel never reads the padding.
    x_buf = _mlx_device_buffer(x)
    weight_buf = _mlx_device_buffer(weight)
    bias_buf = _mlx_device_buffer(bias)
    dims = mx.array([n, cin, height, width, cout], dtype=mx.int32)
    threads, group = launch_config(n, cout, height, width)
    outputs = _get_mlx_kernel()(
        inputs=[x_buf, weight_buf, bias_buf, dims],
        output_shapes=[(n, cout, height, width)],
        output_dtypes=[mx.float32],
        grid=threads,
        threadgroup=group,
    )
    return outputs[0]


def conv3x3_mps(x, weight, bias):
    """Launch via ``torch.mps.compile_shader``."""
    import torch

    _check(x, weight, bias, float_dtype=torch.float32)
    if x.device.type != "mps" or weight.device.type != "mps" or bias.device.type != "mps":
        raise ValueError("conv3x3_mps tensors must be on device 'mps'")
    n, cin, height, width = (int(v) for v in x.shape)
    cout = int(weight.shape[0])
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    out = torch.empty((n, cout, height, width), device="mps", dtype=torch.float32)
    threads, group = launch_config(n, cout, height, width)
    _get_mps_kernel()(
        x,
        weight,
        bias,
        out,
        [n, cin, height, width, cout],
        threads=list(threads),
        group_size=list(group),
        arg_casts={4: "int32"},
    )
    return out


def _check(x, weight, bias, float_dtype) -> None:
    if x.dtype != float_dtype or weight.dtype != float_dtype or bias.dtype != float_dtype:
        raise ValueError(
            "conv3x3 requires float32, got "
            f"x={x.dtype}, weight={weight.dtype}, bias={bias.dtype}"
        )
    if x.ndim != 4 or weight.ndim != 4 or bias.ndim != 1:
        raise ValueError(
            "expected x (N, Cin, H, W), weight (Cout, Cin, 3, 3), bias (Cout,)"
        )
    n, cin, height, width = (int(v) for v in x.shape)
    cout, weight_cin, kh, kw = (int(v) for v in weight.shape)
    if (kh, kw) != (3, 3):
        raise ValueError(f"weight must be 3x3; got {(kh, kw)}")
    if weight_cin != cin:
        raise ValueError(
            f"weight input channels ({weight_cin}) must match x channels ({cin})"
        )
    if tuple(int(v) for v in bias.shape) != (cout,):
        raise ValueError(f"bias must have shape ({cout},); got {tuple(bias.shape)}")
    if min(n, cin, cout, height, width) < 1:
        raise ValueError(
            f"N, Cin, Cout, H, and W must be positive; got "
            f"N={n}, Cin={cin}, Cout={cout}, H={height}, W={width}"
        )


def _mlx_device_buffer(array):
    import mlx.core as mx

    array = mx.contiguous(array)
    if array.size >= 8:
        return array
    tail = mx.zeros((8 - array.size,), dtype=array.dtype)
    return mx.concatenate([array.reshape((array.size,)), tail])


def _get_mlx_kernel():
    global _mlx_kernel
    if _mlx_kernel is None:
        import mlx.core as mx

        _mlx_kernel = mx.fast.metal_kernel(
            name="conv3x3",
            input_names=["x", "weight", "bias", "dims"],
            output_names=["out"],
            source=_MLX_BODY,
            header=metal_source("conv3x3"),
        )
    return _mlx_kernel


def _get_mps_kernel():
    global _mps_library
    if _mps_library is None:
        import torch

        library = torch.mps.compile_shader(metal_source("conv3x3"))
        _mps_library = library.conv3x3
    return _mps_library
