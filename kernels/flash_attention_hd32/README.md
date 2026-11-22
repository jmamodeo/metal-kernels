# flash_attention_hd32

Fused scaled dot-product attention for head dim 32, written for the SAM 3 detector encoder (`[1, 8, 5184, 32]` in bfloat16, 6 calls per image). The design is in [docs/flash-attention-hd32-design.md](../../docs/flash-attention-hd32-design.md).

## Status

Version 0: a simple per-row online-softmax kernel with MLX and PyTorch MPS launchers. Its job is to show the bf16 binding and both launch paths are correct. It has no speed target. Version 1, the tiled `simdgroup_matrix` kernel from the design note, replaces the shader body later and keeps this interface.

The `metal` tests have not yet been run on the M3 Max for v0. The v0 run has not been recorded with the per-version metrics tooling (MTL-4) yet.

## What it computes

`out = softmax(q · kᵀ · scale) · v`, per batch and head, with `scale = 1 / sqrt(32)` by default. This is `reference.sdpa.sdpa_reference` with no bias. The softmax is computed online (running max, running sum, rescaled output), so the `query_length × key_length` score matrix is never written to memory. Bias is not supported yet.

## Inputs

| Name | Shape | Dtype | Notes |
| --- | --- | --- | --- |
| `q` | `(batch, heads, query_length, 32)` | bfloat16 | Made contiguous by the launcher |
| `k` | `(batch, heads, key_length, 32)` | bfloat16 | `key_length ≥ 1` |
| `v` | `(batch, heads, key_length, 32)` | bfloat16 | |
| `dims` | `(3,)` | int32 | `(batch · heads · query_length, query_length, key_length)`, built by the launcher |
| `scale` | `(1,)` | float32 | Built by the launcher |

## Outputs

| Name | Shape | Dtype | Notes |
| --- | --- | --- | --- |
| `out` | `(batch, heads, query_length, 32)` | bfloat16 | fp32 result rounded to nearest even once |

## Threadgroup assumptions

- Grid size: 1-D, `batch · heads · query_length · 32` threads. One simdgroup per query row.
- Threadgroup size: `min(256, grid)`. It must be a multiple of 32 so each simdgroup maps to one row.
- Threadgroup memory: none.
- Layout: row-major contiguous `(…, sequence, 32)`. Lane `i` of a simdgroup owns head-dim column `i`, so each K/V row is one 64-byte load per simdgroup.
- Edge behavior: none needed. Every thread in the grid is a valid `(row, column)`; a row guard is kept for safety. All tensors are limited to fewer than 2³¹ elements because offsets are 32-bit.
- Math: bf16 is widened to fp32 by bit placement. Dot products use `simd_sum`. The scale is folded into `log2(e)` so softmax uses `exp2`. The running max is seeded with key 0, so no `-inf` is needed under fast math. O is rescaled only when the max rises.

## Numerical check

- Reference: `reference/sdpa.py`, in fp32, on inputs already rounded to bf16.
- Tolerance: bf16 output, `rtol = 2e-2`, `atol = 1e-2`. At the SAM 3 shapes the max abs error must be no larger than PyTorch MPS bf16 SDPA's on the same inputs.
- Shapes covered by `tests/test_flash_attention_hd32.py`: `(B, H, Lq, Lk)` = `(1, 1, 1, 1)`, `(1, 2, 5, 7)`, `(2, 3, 17, 33)`, `(1, 8, 64, 64)`, `(1, 2, 9, 300)`, an explicit scale, large scores, a single key, the encoder shape `(1, 8, 5184, 5184)` and the decoder cross-attention shape without bias `(1, 8, 201, 5184)`.

## Hosts

- Python (MLX): `metal_kernels.flash_attention_hd32.flash_attention_hd32_mlx(q, k, v, scale=None)`. It passes this file as `header` to `mx.fast.metal_kernel` and calls `flash_attention_hd32_row`.
- Python (PyTorch MPS): `metal_kernels.flash_attention_hd32.flash_attention_hd32_mps(q, k, v, scale=None)`. It compiles this file with `torch.mps.compile_shader` and calls the `flash_attention_hd32` kernel.
- Swift: not started.
- Benchmark against the built-in op: not started. MLX SDPA (unfused at D = 32) and PyTorch MPS SDPA are the baselines.

```python
import mlx.core as mx
from metal_kernels.flash_attention_hd32 import flash_attention_hd32_mlx

q, k, v = (mx.random.normal((1, 8, 5184, 32)).astype(mx.bfloat16) for _ in range(3))
out = flash_attention_hd32_mlx(q, k, v)
```
