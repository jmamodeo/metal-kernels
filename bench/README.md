# Benchmarks

Benchmark scripts land here next to the kernel they measure. Each one times
that kernel against the built-in framework op (MPS, MLX, or Metal Performance
Shaders) on a named machine, shape, and dtype.

`run.py` prints the registry and does not record timings. There is no kernel
benchmark script yet. FlashAttention hd32 v0 is a correctness version; its
benchmark is step 4 of that kernel.
