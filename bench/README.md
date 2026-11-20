# Benchmarks

Benchmark scripts land here next to the kernel they measure. Each one times
that kernel against the built-in framework op (MPS, MLX, or Metal Performance
Shaders) on a named machine, shape, and dtype.

`run.py` prints the registry and does not record timings. `conv3x3.py` times
that kernel against MLX `conv2d` and PyTorch MPS `conv2d` on a Mac.
