# SPDX-License-Identifier: Apache-2.0
"""NumPy reference for the SAM 3 3x3 convolution.

This is the math for one kernel, written so it can be read before any Metal.
Two functions compute the same result:

- ``conv3x3_naive`` loops over the batch, the output channel, the input
  channel, and the nine taps. Height and width stay in array slices, because
  those pixels do not depend on each other.
- ``conv3x3_im2col`` builds a matrix whose rows are those same patches and
  multiplies it once by the weights. A GPU kernel later wants this view:
  the convolution is a matrix multiply, and the patch matrix is the lowering.

The SAM 3 call, commit 31cb7e62, is the FPN neck ``conv_3x3`` and the mask-head
``PixelDecoder`` convolution. Both are ``nn.Conv2d`` with kernel 3, stride 1,
padding 1, dilation 1, groups 1, and a bias. The image forward traced them in
bfloat16 on input ``[1, 256, 288, 288]`` and weight ``[256, 256, 3, 3]``.
This module accepts float32 and computes in float32. The functions also
accept other ``N``, ``Cin``, ``Cout``, ``H``, and ``W`` so the tests can use
small maps. The kernel geometry does not change.

Layout. The input is NCHW: batch, channel, height, width. A C-contiguous
array stores one row of width ``W`` together, then the next row, then the
next channel. The weight is OIHW, which is PyTorch's ``Conv2d`` layout:
output channel, input channel, then the 3x3 tap. The bias has one value per
output channel. The output is NCHW again.

Padding. One zero is added on the top, bottom, left, and right of the input.
Output pixel ``(y, x)`` then reads padded coordinates ``(y + ky, x + kx)`` for
``ky, kx`` in ``0, 1, 2``. The center tap (``ky = kx = 1``) lands on the
input pixel ``(y, x)``. The output map is the same height and width as the
input. A 1x1 input becomes a 3x3 padded map, and the single output pixel
sees eight zeros around that one input value.

One output pixel. For output channel ``oc`` at ``(y, x)``:

    out[n, oc, y, x] = bias[oc]
      + sum over ic, ky, kx of
          weight[oc, ic, ky, kx] * padded[n, ic, y+ky, x+kx]

At the SAM 3 width that sum has ``256 * 9 = 2304`` products. In general it
has ``Cin * 9`` products. The bias is added once, after the sum.

FLOPs. Each product and the add that accumulates it count as one FLOP
each, so a dot product of length ``K`` is ``2 * K`` FLOPs. The inventory
uses that convention. Over every output channel and every pixel:

    flops = 2 * N * Cout * Cin * 9 * H * W

For ``N = 1``, ``Cout = Cin = 256``, ``H = W = 288`` that is 97844723712,
which the op inventory writes as 9.784e10. The bias add is another
``N * Cout * H * W`` FLOPs, about 0.02% of that number, and the inventory
does not include it.

Memory, and why the op is compute-bound. The tensors themselves, read or
written once, are the input, the weight, the bias, and the output. In
float32 at the SAM 3 shape that is 81 MiB + 2.25 MiB + 1 KiB + 81 MiB =
164.25 MiB (172229632 bytes). Dividing the multiply-add FLOPs by those
bytes is about 568 FLOPs per byte. Each input value is reused by every
output channel and by every window that covers it (up to 9 windows times
256 output channels). The 2.25 MiB of weights are reused at all 288x288
positions. A kernel that does hundreds of FLOPs per byte of unique tensor
traffic spends its time on the multiply-adds. That is the compute-bound
regime. A memory-bound kernel, such as a copy or a single-tap scale, does a
handful of FLOPs per byte and then waits on the load.

im2col spends memory to make the matmul obvious. Each of the ``H * W``
output pixels gets its own row of ``Cin * 9`` values, so the patch matrix
at this shape is ``288 * 288 * 2304`` float32 values, 729 MiB. The same
input pixel is stored once for every window that covers it. The matmul
``(H*W, Cin*9) @ (Cin*9, Cout)`` is exactly the formula above. The extra
buffer is the cost of that view, not extra math.
"""

from __future__ import annotations

import numpy as np

# Geometry of both SAM 3 call sites. Not parameters of the reference.
_KERNEL = 3
_PADDING = 1

# SAM 3 image forward: one 3x3 on the finest FPN map, in the neck and again
# in the mask head. The integer is 2 * 256 * 256 * 9 * 288 * 288.
SAM3_INPUT_SHAPE = (1, 256, 288, 288)
SAM3_WEIGHT_SHAPE = (256, 256, 3, 3)
SAM3_FLOPS = 97_844_723_712
SAM3_TENSOR_BYTES = 172_229_632
SAM3_IM2COL_BYTES = 764_411_904


def conv3x3_flops(n: int, cin: int, cout: int, height: int, width: int) -> int:
    """Multiply-add FLOPs for this conv: ``2 * N * Cout * Cin * 9 * H * W``.

    The ``2`` counts each product and each add. Bias is not included. With
    ``N = 1`` this is the inventory formula ``2 * Cout * Cin * 9 * H * W``.
    """
    return 2 * n * cout * cin * _KERNEL * _KERNEL * height * width


def conv3x3_tensor_bytes(n: int, cin: int, cout: int, height: int, width: int) -> int:
    """Bytes of input, weight, bias, and output, each counted once.

    This is the traffic of the convolution itself, in float32. It does not
    count the im2col patch matrix.
    """
    elements = (
        n * cin * height * width
        + cout * cin * _KERNEL * _KERNEL
        + cout
        + n * cout * height * width
    )
    return elements * np.dtype(np.float32).itemsize


def conv3x3_im2col_bytes(n: int, cin: int, height: int, width: int) -> int:
    """Bytes in the im2col patch matrix, float32.

    One row per output pixel, ``Cin * 9`` columns. At the SAM 3 shape this
    is 729 MiB.
    """
    elements = n * height * width * cin * _KERNEL * _KERNEL
    return elements * np.dtype(np.float32).itemsize


def conv3x3_naive(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """3x3 conv by looping over channels and taps.

    The loops are the batch item, the output channel, the input channel, and
    the nine kernel positions. Only height and width are slices. Those pixels
    do not depend on each other, and a Python loop over ``H * W`` would not
    finish at 288x288 in reasonable time. One tap is a multiply of a scalar
    by an ``(H, W)`` window, then an add into the accumulator.

    ``weight[oc, ic, ky, kx]`` multiplies the window of input channel ``ic``
    whose top-left corner sits ``ky`` rows and ``kx`` columns above output
    pixel ``(0, 0)`` in the padded image. Sliding that window by one pixel
    is just taking the next slice. That is the whole spatial loop.
    """
    x, weight, bias = _check_conv3x3(x, weight, bias)
    n, cin, height, width = x.shape
    cout = weight.shape[0]
    padded = _pad(x)
    out = np.empty((n, cout, height, width), dtype=np.float32)

    for batch in range(n):
        for oc in range(cout):
            # Float32 accumulator. Bias is added once after the Cin*9 taps.
            acc = np.zeros((height, width), dtype=np.float32)
            for ic in range(cin):
                for ky in range(_KERNEL):
                    for kx in range(_KERNEL):
                        window = padded[batch, ic, ky : ky + height, kx : kx + width]
                        acc += weight[oc, ic, ky, kx] * window
            out[batch, oc] = acc + bias[oc]
    return out


def conv3x3_im2col(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """3x3 conv as one matmul of an im2col patch matrix.

    For every output pixel, collect the ``Cin * 3 * 3`` input values that
    the naive loops would have dotted with ``weight[oc]``. Stack those
    patches as rows. Reshape the weights to ``(Cin * 9, Cout)``, one column
    per output channel, in the same tap order. Their product is every dot
    product in the output. Adding ``bias`` is a broadcast over the rows.

    At 256 channels each row has 2304 columns. The matmul shape is
    ``(N*H*W, Cin*9) @ (Cin*9, Cout)``.
    """
    x, weight, bias = _check_conv3x3(x, weight, bias)
    n, cin, height, width = x.shape
    cout = weight.shape[0]
    padded = _pad(x)

    # patches[n, y, x, ic, ky, kx] is the input value that tap (ky, kx) of
    # input channel ic contributes to output pixel (y, x). The last three
    # axes ravel in that order, which matches weight[oc].ravel(): OIHW with
    # the output channel removed is (Cin, 3, 3).
    patches = np.empty((n, height, width, cin, _KERNEL, _KERNEL), dtype=np.float32)
    for ky in range(_KERNEL):
        for kx in range(_KERNEL):
            window = padded[:, :, ky : ky + height, kx : kx + width]
            patches[:, :, :, :, ky, kx] = np.transpose(window, (0, 2, 3, 1))

    # (N*H*W, Cin*9) times (Cin*9, Cout) -> (N*H*W, Cout).
    col = patches.reshape(n * height * width, cin * _KERNEL * _KERNEL)
    w_col = weight.reshape(cout, cin * _KERNEL * _KERNEL).T
    out = col @ w_col
    out += bias
    out = out.reshape(n, height, width, cout).transpose(0, 3, 1, 2)
    return np.ascontiguousarray(out, dtype=np.float32)


def _pad(x: np.ndarray) -> np.ndarray:
    """Pad H and W by one zero. N and C are unchanged."""
    return np.pad(
        x,
        pad_width=((0, 0), (0, 0), (_PADDING, _PADDING), (_PADDING, _PADDING)),
        mode="constant",
        constant_values=0,
    )


def _check_conv3x3(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Require NCHW input, OIHW 3x3 weight, and a bias vector, all float32."""
    x = _as_float32(x, "x")
    weight = _as_float32(weight, "weight")
    bias = _as_float32(bias, "bias")

    if x.ndim != 4:
        raise ValueError(f"x must have shape (N, Cin, H, W); got ndim {x.ndim}")
    if weight.ndim != 4:
        raise ValueError(
            f"weight must have shape (Cout, Cin, 3, 3); got ndim {weight.ndim}"
        )
    n, cin, height, width = (int(dim) for dim in x.shape)
    cout, weight_cin, kh, kw = (int(dim) for dim in weight.shape)
    if (kh, kw) != (_KERNEL, _KERNEL):
        raise ValueError(f"weight must be 3x3; got {(kh, kw)}")
    if weight_cin != cin:
        raise ValueError(
            f"weight input channels ({weight_cin}) must match x channels ({cin})"
        )
    if n < 1 or cin < 1 or cout < 1 or height < 1 or width < 1:
        raise ValueError(
            f"N, Cin, Cout, H, and W must be positive; got "
            f"N={n}, Cin={cin}, Cout={cout}, H={height}, W={width}"
        )
    if bias.shape != (cout,):
        raise ValueError(f"bias must have shape ({cout},); got {tuple(bias.shape)}")
    return x, weight, bias


def _as_float32(value: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray; got {type(value).__name__}")
    if value.dtype != np.float32:
        raise ValueError(f"{name} must have dtype float32; got {value.dtype}")
    return value
