# flash_attention_hd32

Fused scaled dot-product attention for head dim 32, written for the SAM 3 detector encoder (`[1, 8, 5184, 32]` in bfloat16, 6 calls per image). The design is in [docs/flash-attention-hd32-design.md](../../docs/flash-attention-hd32-design.md).

## Status

Version 0: a simple per-row online-softmax kernel with MLX and PyTorch MPS launchers. Its job is to show that the bf16 and fp32 bindings, the bias path and both launch paths are correct. It has no speed target. Version 1, the tiled `simdgroup_matrix` kernel from the design note, replaces the shader body later and keeps this interface.

The `metal` tests have not been run on the M3 Max for v0 yet, and no v0 benchmark run is recorded. `bench/run_mac.sh v0` does both.

## What it computes

`out = softmax(q · kᵀ · scale + bias) · v`, per batch and head, with `scale = 1 / sqrt(32)` by default. This is `reference.sdpa.sdpa_reference`. The softmax is computed online (running max, running sum, rescaled output), so the `query_length × key_length` score matrix is never written to memory.

The bias is optional and additive, and is added after scaling. A `-inf` bias masks that key: it is skipped and never enters the running state. A query row whose keys are all masked has an empty running sum (`l == 0`) and outputs zeros, as the reference does.

## Inputs

| Name | Shape | Dtype | Notes |
| --- | --- | --- | --- |
| `q` | `(batch, heads, query_length, 32)` | bfloat16 or float32 | Made contiguous by the launcher |
| `k` | `(batch, heads, key_length, 32)` | same as `q` | `key_length ≥ 1` |
| `v` | `(batch, heads, key_length, 32)` | same as `q` | |
| `bias` | broadcastable to `(batch, heads, query_length, key_length)` | same as `q` | Optional. Passed unbroadcast, with stride 0 on broadcast dimensions |
| `dims` | `(9,)` | int32 | Rows, heads, query length, key length, has-bias flag, four bias strides. Built by the launcher |
| `scale` | `(1,)` | float32 | Built by the launcher |

## Outputs

| Name | Shape | Dtype | Notes |
| --- | --- | --- | --- |
| `out` | `(batch, heads, query_length, 32)` | same as `q` | bf16 output is the fp32 result rounded to nearest even once |

## Threadgroup assumptions

- Grid size: 1-D, `batch · heads · query_length · 32` threads. One simdgroup per query row.
- Threadgroup size: `min(256, grid)`. It must be a multiple of 32 so each simdgroup maps to one row.
- Threadgroup memory: none.
- Layout: row-major contiguous `(…, sequence, 32)`. Lane `i` of a simdgroup owns head-dim column `i`, so each K/V row is one 64-byte (bf16) or 128-byte (fp32) load per simdgroup.
- Edge behavior: none needed. Every thread in the grid is a valid `(row, column)`, and any key length works, including tails that are not a multiple of 64. A row guard is kept for safety. Each tensor is limited to fewer than 2³¹ elements because offsets are 32-bit.
- Math: bf16 is widened to fp32 by bit placement. Dot products use `simd_sum`. The scale and bias are folded into `log2(e)` so softmax uses `exp2`. `-inf` is detected from the bits, so fast math cannot fold the test away, and no infinity is used in arithmetic. O is rescaled only when the max rises.
- Entry points: PyTorch MPS calls `flash_attention_hd32_bfloat` or `flash_attention_hd32_float`. MLX passes this file as `header` and calls `flash_attention_hd32_row` from its generated kernel body.

## Numerical check

- Reference: `reference/sdpa.py`, in fp32. bf16 cases use inputs already rounded to bf16.
- Tolerance: bf16 output `rtol = 2e-2`, `atol = 1e-2`; fp32 output `rtol = 1e-3`, `atol = 1e-4`. At the SAM 3 shapes the bf16 max abs error must also be no larger than PyTorch MPS bf16 SDPA's on the same inputs.
- `tests/test_flash_attention_hd32.py` runs every case on both hosts, in bf16 and fp32 unless noted:
  - `(B, H, Lq, Lk)` = `(1, 1, 1, 1)`, `(1, 2, 5, 7)`, `(2, 3, 17, 33)`, `(1, 8, 64, 64)`, `(1, 2, 9, 100)` and `(1, 2, 9, 1000)`
  - bias that is full, per key, per (query, key), per (batch, query) and a scalar, at `Lk` = 7, 100 and 1000, with every third bias element `-inf`
  - fully `-inf`-masked rows, which must be exactly zero
  - an explicit scale, large scores and a single key
  - bf16 only: the encoder shape `(1, 8, 5184, 5184)`, and the decoder cross-attention shape `(1, 8, 201, 5184)` with and without a full bias

## Hosts

- Python: `metal_kernels.flash_attention.flash_attention_hd32(q, k, v, bias=None, scale=None, *, backend)`, with `backend="mlx"` for `mlx.core.array` inputs or `backend="mps"` for `torch.Tensor` inputs on `mps`.
  - MLX passes this file as `header` to `mx.fast.metal_kernel`.
  - PyTorch MPS compiles this file with `torch.mps.compile_shader`.
- Swift: not started.
- Benchmark: `bench/flash_attention_hd32.py` against `mx.fast.scaled_dot_product_attention` (unfused at D = 32) and PyTorch MPS SDPA. `bench/run_mac.sh <version>` tests, benchmarks and records a version. See [bench/README.md](../../bench/README.md).

```python
import mlx.core as mx
from metal_kernels.flash_attention import flash_attention_hd32

q, k, v = (mx.random.normal((1, 8, 5184, 32)).astype(mx.bfloat16) for _ in range(3))
out = flash_attention_hd32(q, k, v, backend="mlx")
```
