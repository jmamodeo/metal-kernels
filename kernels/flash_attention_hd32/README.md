# flash_attention_hd32

Fused scaled dot-product attention for head dim 32, written for the SAM 3 detector encoder (`[1, 8, 5184, 32]` in bfloat16, 6 calls per image). The design is in [docs/flash-attention-hd32-design.md](../../docs/flash-attention-hd32-design.md).

## Status

Version 1: the tiled `simdgroup_matrix` kernel (32 query rows, 128 threads, 64-key K/V blocks loaded from device into registers, bf16 GEMM inputs, fp32 P). It keeps the v0 interface: the same Python API, buffers, `dims` layout, bias semantics and tolerances. On the M3 Max at `[1, 8, 5184, 32]` bf16 it runs at 4.12 ms (MLX path) against MLX SDPA's 8.41 ms in the same record (2.04×), and 4.17 ms (MPS path) against MPS SDPA's 3.91 ms, with max abs error equal to MPS SDPA's. `bench/run_mac.sh v1` tests, benchmarks and records it. Earlier versions and a recorded v2 trial are in [docs/ROADMAP.md](../../docs/ROADMAP.md).

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

- Block sizes: `FA_HD32_BLOCK_Q = 32` query rows per threadgroup, `FA_HD32_BLOCK_K = 64` keys per block, 8 rows per simdgroup. They are `constexpr` in the shader and mirrored by `QUERY_BLOCK`, `KEY_BLOCK` and `THREADGROUP_SIZE` in `metal_kernels.flash_attention`.
- Grid size: 1-D, `batch · heads · ceil(query_length / 32) · 128` threads. One threadgroup per (batch · head, 32-query block): 1296 threadgroups at `[1, 8, 5184, 32]`.
- Threadgroup size: exactly 128 threads (`FA_HD32_THREADS`), four simdgroups. Simdgroup `s` owns query rows `[8s, 8s + 8)` of the block. Simdgroups are independent; one whose rows are all past `query_length` returns at once.
- Threadgroup memory: none, and no barriers. Like metal-flash-attention on Apple9, each lane loads its K and V tile elements straight from device memory into registers. The simdgroups of a threadgroup read the same K/V block at about the same time, so the repeats hit the cache.
- Tiles: 8×8 `simdgroup_matrix`. For bf16 the GEMM inputs Q, K and V are `bfloat` tiles when the compiler defines `__HAVE_BFLOAT__` (Metal 3.1); otherwise bf16 is widened to fp32 tiles. fp32 inputs use fp32 tiles. S, P and the O accumulator are fp32, so P is never rounded. Q and O stay in registers as four tiles each across D = 32. Per 64-key block, `S = Q · Kᵀ` is eight tiles, then `O += P · V`. The row max and sum use `simd_shuffle_xor` with lane masks 1 and 8, which relies on the Apple 8×8 lane layout that metal-flash-attention and MLX also use (`fa_hd32_tile_coord`).
- Loads: the head dimension is split as `d = 4i + t` across the four tiles `t`. Both GEMMs use the same split, so the result is unchanged. Each lane's Q, V and O elements become 8 contiguous values and its Kᵀ elements become 4 contiguous values per key, so every load is a 4-element vector load and O is written directly to device memory.
- Layout: row-major contiguous `(…, sequence, 32)`. Each K/V block is one contiguous span, loaded by the 64 threads with consecutive addresses.
- Edge behavior: any query length and any key length work. Keys past `key_length` in the last block are not read (their K and V are zero in registers) and get `p = 0`; query rows past `query_length` in the last block compute on zeros, never read the bias and are not stored. Each tensor is limited to fewer than 2³¹ elements because offsets are 32-bit.
- Math: bf16 bits are reinterpreted as `bfloat`, or widened to fp32 by bit placement. The scale (applied to S in fp32) and bias are folded into `log2(e)` so softmax uses `exp2`. `-inf` is detected from the bits, so fast math cannot fold the test away, and no infinity is used in arithmetic: the running max starts at the finite `-FLT_MAX`, masked keys are excluded with a per-element flag rather than a sentinel score, and the rescale exponent is clamped to `-FLT_MAX`. O and the running sum are multiplied by `exp2(m_old − m_new)` only in blocks where some row of the simdgroup raised its max (tested with `simd_any`, so the branch is uniform across the simdgroup); the factor is exactly 1 for a row whose max did not rise, so skipping it leaves the result unchanged.
- Entry points: PyTorch MPS calls `flash_attention_hd32_bfloat` or `flash_attention_hd32_float`. MLX passes this file as `header` and calls `flash_attention_hd32_block` from its generated kernel body.

## Numerical check

- Reference: `reference/sdpa.py`, in fp32. bf16 cases use inputs already rounded to bf16.
- Tolerance: bf16 output `rtol = 2e-2`, `atol = 1e-2`; fp32 output `rtol = 1e-3`, `atol = 1e-4`. At the SAM 3 shapes the bf16 max abs error must also be no larger than PyTorch MPS bf16 SDPA's on the same inputs.
- `tests/test_flash_attention_hd32.py` runs every case on both hosts, in bf16 and fp32 unless noted:
  - `(B, H, Lq, Lk)` = `(1, 1, 1, 1)`, `(1, 2, 5, 7)`, `(2, 3, 17, 33)`, `(1, 8, 64, 64)`, `(1, 2, 9, 100)` and `(1, 2, 9, 1000)`
  - bias that is full, per key, per (query, key), per (batch, query) and a scalar, at `Lk` = 7, 100 and 1000, with every third bias element `-inf`
  - fully `-inf`-masked rows, which must be exactly zero
  - tail blocks: `Lq = 50` (not a multiple of 32) and `Lk = 5184 + 17` (not a multiple of 64), alone and together, with a full bias, and with fully `-inf`-masked rows inside the partial last query block
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
