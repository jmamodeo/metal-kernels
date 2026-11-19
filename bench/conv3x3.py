#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Time conv3x3 against MLX and PyTorch MPS at the SAM 3 shape.

Run this on a Mac. It does not record a number by itself, and it does not
run on the Linux CI. The shape is input [1, 256, 288, 288], weight
[256, 256, 3, 3], bias [256], float32. FLOPs per call are 9.784e10
(2 * 256 * 256 * 9 * 288 * 288), the same count as the op inventory.

MLX ``conv2d`` takes NHWC input and weight ``(Cout, KH, KW, Cin)``, and it
has no bias argument. Those tensors are built before the timer. The timed
MLX built-in is ``conv2d`` plus the bias add. PyTorch ``F.conv2d`` includes
the bias. Neither baseline builds an im2col matrix in this process.

Example, from the repository root, after ``pip install -e "python[mac]"``::

    python bench/conv3x3.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SOURCE = _REPO / "python" / "src"
for _path in (_SOURCE, _REPO):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from reference.conv3x3 import SAM3_FLOPS, SAM3_INPUT_SHAPE, SAM3_WEIGHT_SHAPE  # noqa: E402


def main() -> int:
    if sys.platform != "darwin":
        print("conv3x3 benchmark runs on macOS. This host is not macOS.")
        return 1

    n, cin, height, width = SAM3_INPUT_SHAPE
    cout = SAM3_WEIGHT_SHAPE[0]
    print(
        f"conv3x3  input {list(SAM3_INPUT_SHAPE)}  "
        f"weight {list(SAM3_WEIGHT_SHAPE)}  float32  "
        f"{SAM3_FLOPS:.3e} FLOP"
    )
    ran = False
    if _mlx_available():
        _report("mlx kernel", _time_mlx_kernel(n, cin, cout, height, width))
        _report("mlx conv2d + bias", _time_mlx_builtin(n, cin, cout, height, width))
        ran = True
    else:
        print("mlx: not installed")
    if _mps_available():
        _report("mps kernel", _time_mps_kernel(n, cin, cout, height, width))
        _report("mps conv2d", _time_mps_builtin(n, cin, cout, height, width))
        ran = True
    else:
        print("torch MPS: not available")
    if not ran:
        return 1
    return 0


def _report(name: str, milliseconds: float) -> None:
    seconds = milliseconds / 1e3
    tflops = (SAM3_FLOPS / seconds) / 1e12
    print(f"{name:<20} {milliseconds:8.2f} ms   {tflops:6.2f} TFLOP/s")


def _time_mlx_kernel(n, cin, cout, height, width) -> float:
    import mlx.core as mx
    from metal_kernels.conv3x3 import conv3x3_mlx

    x, weight, bias = _mlx_nchw(n, cin, cout, height, width)

    def run():
        mx.eval(conv3x3_mlx(x, weight, bias))

    return _time(run)


def _time_mlx_builtin(n, cin, cout, height, width) -> float:
    import mlx.core as mx

    # NHWC, and weight (Cout, KH, KW, Cin). Built once, outside the timer.
    x = mx.random.normal(shape=(n, height, width, cin))
    weight = mx.random.normal(shape=(cout, 3, 3, cin))
    bias = mx.random.normal(shape=(cout,))
    mx.eval(x, weight, bias)

    def run():
        mx.eval(mx.conv2d(x, weight, padding=1) + bias)

    return _time(run)


def _time_mps_kernel(n, cin, cout, height, width) -> float:
    import torch
    from metal_kernels.conv3x3 import conv3x3_mps

    x, weight, bias = _mps_nchw(n, cin, cout, height, width)

    def run():
        conv3x3_mps(x, weight, bias)
        torch.mps.synchronize()

    return _time(run)


def _time_mps_builtin(n, cin, cout, height, width) -> float:
    import torch
    import torch.nn.functional as F

    x, weight, bias = _mps_nchw(n, cin, cout, height, width)

    def run():
        F.conv2d(x, weight, bias, stride=1, padding=1)
        torch.mps.synchronize()

    return _time(run)


def _time(run, warmup: int = 5, iters: int = 10) -> float:
    for _ in range(warmup):
        run()
    start = time.perf_counter()
    for _ in range(iters):
        run()
    return (time.perf_counter() - start) * 1e3 / iters


def _mlx_nchw(n, cin, cout, height, width):
    import mlx.core as mx

    x = mx.random.normal(shape=(n, cin, height, width))
    weight = mx.random.normal(shape=(cout, cin, 3, 3))
    bias = mx.random.normal(shape=(cout,))
    mx.eval(x, weight, bias)
    return x, weight, bias


def _mps_nchw(n, cin, cout, height, width):
    import torch

    x = torch.randn(n, cin, height, width, device="mps", dtype=torch.float32)
    weight = torch.randn(cout, cin, 3, 3, device="mps", dtype=torch.float32)
    bias = torch.randn(cout, device="mps", dtype=torch.float32)
    torch.mps.synchronize()
    return x, weight, bias


def _mlx_available() -> bool:
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        return False
    return True


def _mps_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.backends.mps.is_available())


if __name__ == "__main__":
    raise SystemExit(main())
