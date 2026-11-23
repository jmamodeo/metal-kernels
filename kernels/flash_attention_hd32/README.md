# flash_attention_hd32

Fused scaled dot-product attention for head dim 32, written for the SAM 3 detector encoder (`[1, 8, 5184, 32]` in bfloat16, 6 calls per image). The design is in [docs/flash-attention-hd32-design.md](../../docs/flash-attention-hd32-design.md).

## Status

Version 6, the previous kernel, is the v1 tiled `simdgroup_matrix` kernel with each 64-key K/V block copied once per threadgroup into threadgroup memory (8 KB for bf16, 16 KB for fp32), shared by the four simdgroups. On the M3 Max at `[1, 8, 5184, 32]` bf16, same-record medians are 3.837 ms (MLX path) against v1's 4.330 ms and MLX SDPA's 8.366 ms, and 3.803 ms (MPS path) against v1's 4.106 ms and MPS SDPA's 3.833 ms. The paired candidate/v1 median-ratio 95% CIs are 0.8877 [0.8723, 0.9109] (MLX) and 0.9242 [0.9224, 0.9272] (MPS). The M3 target is met: the paired MPS kernel / MPS SDPA ratio is 0.9913 [0.9880, 0.9928]. Max abs error is 9.289e-4, equal to MPS SDPA's, mean 2.603e-5. It was recorded against the frozen v1. Earlier versions are in [docs/ROADMAP.md](../../docs/ROADMAP.md).

Version 7, the previous kernel, is v6 with the threadgroup K/V stage double buffered (16 KB for bf16, 32 KB for fp32). The next 64-key block's device loads are issued into registers before the current block's math and stored into the other buffer after it, so each block costs one barrier instead of two. On the M3 Max at `[1, 8, 5184, 32]` bf16, same-record medians are 3.812 ms (MLX path) against the frozen v6's 3.882 ms and MLX SDPA's 8.357 ms, and 3.824 ms (MPS path) against v6's 3.866 ms and MPS SDPA's 3.905 ms. The paired v7/v6 median-ratio 95% CIs are 0.9823 [0.9804, 0.9835] (MLX) and 0.9896 [0.9885, 0.9912] (MPS), and the paired MPS kernel / MPS SDPA ratio is 0.9798 [0.9788, 0.9807]. Max abs error is 9.289e-4, equal to MPS SDPA's. It was recorded against the frozen v6 (`bench/baselines/flash_attention_hd32_v6.metal`).

Version 7.1 is the current kernel. bf16 runs v7's K/V loop unchanged (double buffered, 16 KB, one barrier per block). fp32 runs v6's loop unchanged (one 16 KB buffer, each block copied straight into it, a barrier before and after the math), so fp32 no longer stages 32 KB, which was Apple's whole per-threadgroup limit. The loop is picked at compile time by element size in the one shared shader. v7.1 has not been measured yet. `bench/run_mac.sh v7.1` tests it, benchmarks and records bf16 against the frozen v7 (`bench/baselines/flash_attention_hd32_v7.metal`) and fp32 against the frozen v6; each passes when the paired candidate/baseline median-ratio 95% CI's upper bound is at most 1.01. It then runs the Swift build, tests and bf16/fp32 timing.

## Results

On one measured configuration — Apple M3 Max, macOS 26.7.3, torch 2.15.0, MLX 0.32.4, at `[1, 8, 5184, 32]` bf16 — the current kernel is about 2% faster than MPS SDPA (paired MPS kernel/SDPA median ratio 0.9798, 95% CI [0.9788, 0.9807]; 102 pairs, 10,000 resamples). The table is generated from `bench/results/flash_attention_hd32.jsonl` by `python tools/results_table.py flash_attention_hd32`, which also writes [bench/results/flash_attention_hd32.md](../../bench/results/flash_attention_hd32.md). Records are append-only and include trials that were not kept.

| Version | Date (UTC) | Commit | Design | MLX kernel ms | PyTorch kernel ms | Best TFLOP/s | Speedup vs built-in |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| v0 | 2026-11-22 | 6686822e7346530011f3d86ef60a188a2df5bf78 | per-row online softmax: one simdgroup per query row, one lane per head-dim column, simd_sum dot products, fp32 math, no threadgroup memory | 60.035 | 60.424 | 0.46 | mlx: 0.14x, mps: 0.06x |
| v1 | 2026-11-22 | 5f1f3e75b92a9ceea9a91d6d995d5766bd64f7a1 (dirty) | tiled simdgroup_matrix: one threadgroup of 2 simdgroups per 16 query rows, 8 rows per simdgroup, 64-key K/V blocks staged as fp32 in threadgroup memory (16 KB), fp32 simdgroup_float8x8 S=QK^T and O+=PV, exp2 online softmax with shuffle-xor row reductions | 39.998 | 40.174 | 0.69 | mlx: 0.21x, mps: 0.10x |
| v1 | 2026-11-22 | 696f2fdbd03ec819e5bb40acfcd4311a081bc169 (dirty) | tiled simdgroup_matrix: one threadgroup of 4 simdgroups per 32 query rows, 8 rows per simdgroup, 64-key K/V blocks loaded per lane from device into registers as vectors (no threadgroup memory, no barriers), bf16 simdgroup_matrix inputs with fp32 S and O accumulators (P stays fp32), exp2 online softmax with shuffle-xor row reductions | 4.123 | 4.168 | 6.68 | mlx: 2.04x, mps: 0.94x |
| v2 | 2026-11-23 | 329b94ed7fa50891eafa3be69af07ce46f5c63bc (dirty) | v1 tiling kept (32 query rows, 4 simdgroups, device-to-register K/V, bf16 GEMM inputs, fp32 P), key/value block raised from 64 to 128 | 4.312 | 4.293 | 6.41 | mlx: 1.95x, mps: 0.90x |
| v3 | 2026-11-23 | 270e26e0c0e5b6d53d2dead528e63699b04a9f84 (dirty) | v1 tiling kept (8 query rows per simdgroup, 64-key K/V blocks loaded per lane from device into registers, bf16 GEMM inputs, fp32 P), query block raised from 32 to 64 rows with 8 simdgroups (256 threads) per threadgroup | 4.263 | 4.270 | 6.46 | mlx: 1.97x, mps: 0.91x |
| v4 | 2026-11-23 | e083b7279739f3f01f147f2a9f00753a35515f25 (dirty) | v1 kernel kept (32 query rows, 4 simdgroups, 64-key K/V blocks loaded per lane from device into registers, bf16 GEMM inputs, fp32 P), O and running-sum rescale skipped when no row max in the simdgroup rises (simd_any) | 4.222 | 4.213 | 6.53 | mlx: 2.00x, mps: 0.93x |
| v1 | 2026-11-23 | 7394f28065c5bfccf67a3ac05195c1e73743c98c (dirty) | tiled simdgroup_matrix: one threadgroup of 4 simdgroups per 32 query rows, 8 rows per simdgroup, 64-key K/V blocks loaded per lane from device into registers as vectors (no threadgroup memory, no barriers), bf16 simdgroup_matrix inputs with fp32 S and O accumulators (P stays fp32), exp2 online softmax with shuffle-xor row reductions | 4.328 | 4.306 | 6.39 | mlx: 1.98x, mps: 0.91x |
| v1 | 2026-11-23 | 436f08433b13a89a41196bfc862895acaecf8471 (dirty) | tiled simdgroup_matrix: one threadgroup of 4 simdgroups per 32 query rows, 8 rows per simdgroup, 64-key K/V blocks loaded per lane from device into registers as vectors (no threadgroup memory, no barriers), bf16 simdgroup_matrix inputs with fp32 S and O accumulators (P stays fp32), exp2 online softmax with shuffle-xor row reductions | 4.423 | 4.183 | 6.58 | mlx: 1.89x, mps: 0.93x |
| v1 | 2026-11-23 | ed2166a880a4112c3c4eddf5ebb4826f9ed21e46 (dirty) | tiled simdgroup_matrix: one threadgroup of 4 simdgroups per 32 query rows, 8 rows per simdgroup, 64-key K/V blocks loaded per lane from device into registers as vectors (no threadgroup memory, no barriers), bf16 simdgroup_matrix inputs with fp32 S and O accumulators (P stays fp32), exp2 online softmax with shuffle-xor row reductions | 4.353 | 4.183 | 6.58 | mlx: 1.92x, mps: 0.93x |
| v1 | 2026-11-23 | 9121f2e3d9f1d87d72c89c46c4597530e629bd67 (dirty) | tiled simdgroup_matrix: one threadgroup of 4 simdgroups per 32 query rows, 8 rows per simdgroup, 64-key K/V blocks loaded per lane from device into registers as vectors (no threadgroup memory, no barriers), bf16 simdgroup_matrix inputs with fp32 S and O accumulators (P stays fp32), exp2 online softmax with shuffle-xor row reductions | 4.375 | 4.182 | 6.58 | mlx: 1.91x, mps: 0.93x |
| v1 | 2026-11-23 | 9121f2e3d9f1d87d72c89c46c4597530e629bd67 (dirty) | tiled simdgroup_matrix: one threadgroup of 4 simdgroups per 32 query rows, 8 rows per simdgroup, 64-key K/V blocks loaded per lane from device into registers as vectors (no threadgroup memory, no barriers), bf16 simdgroup_matrix inputs with fp32 S and O accumulators (P stays fp32), exp2 online softmax with shuffle-xor row reductions | 4.398 | 4.182 | 6.58 | mlx: 1.90x, mps: 0.93x |
| v1 | 2026-11-23 | 9121f2e3d9f1d87d72c89c46c4597530e629bd67 (dirty) | tiled simdgroup_matrix: one threadgroup of 4 simdgroups per 32 query rows, 8 rows per simdgroup, 64-key K/V blocks loaded per lane from device into registers as vectors (no threadgroup memory, no barriers), bf16 simdgroup_matrix inputs with fp32 S and O accumulators (P stays fp32), exp2 online softmax with shuffle-xor row reductions | 4.375 | 4.179 | 6.58 | mlx: 1.91x, mps: 0.93x |
| v6 | 2026-11-23 | 86d0895728026bea1c4a4e83436531fd4c5c570d (dirty) | v1 tiling (32 query rows, 4 simdgroups x 8 rows, 64-key blocks, bf16 simdgroup_matrix inputs, fp32 P and accumulators, always-applied exp2 correction) with each K/V block loaded once by all 128 threads into 8 KB of bf16 threadgroup memory shared by the 4 simdgroups, zero-filled past key_length, barrier after the load and before the next block | 3.837 | 3.803 | 7.24 | mlx: 2.18x, mps: 1.01x |
| v7 | 2026-11-23 | 633990b1fa26f9e60aa1c5d5d664b0901d43399b (dirty) | v6 with the K/V stage double buffered in threadgroup memory (16 KB for bf16, 32 KB for fp32): the next 64-key block&#x27;s device loads are issued into registers before the current block&#x27;s math and stored into the other buffer after it, one barrier per block instead of two | 3.812 | 3.824 | 7.22 | mlx: 2.19x, mps: 1.02x |


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
- Threadgroup size: exactly 128 threads (`FA_HD32_THREADS`), four simdgroups. Simdgroup `s` owns query rows `[8s, 8s + 8)` of the block. A simdgroup whose rows are all past `query_length` still helps copy K/V and waits at the barriers, but skips the math.
- Threadgroup memory (v7.1): `N` buffers, each one 64-key K block and one V block in the input dtype (`FA_HD32_BUFFER_ELEMENTS` = 4096 elements). `N = FaHd32Stage<T>::buffers` is 2 for bf16 and 1 for fp32, so the stage is `FA_HD32_KV_BYTES` = 16 KB for either dtype. It is declared at kernel scope (`FA_HD32_KV_STAGE` in the MLX body, a local array in the MPS entry point) and passed to `flash_attention_hd32_block`. All 128 threads copy with coalesced 4-element vector loads (zero past `key_length`). bf16 (v7's loop): block `b` lives in buffer `b % 2`; block 0 is copied, then a barrier; in iteration `b` each thread loads its share of block `b + 1` from device memory into registers, the lanes read their tile elements of block `b` for the math, each thread stores block `b + 1` into the other buffer, and one barrier follows (none after the last block). fp32 (v6's loop): each block is copied straight into the one buffer, then a barrier, the math, and a second barrier before the next copy. The shader declares the stage statically, so no host sets threadgroup memory.
- Tiles: 8×8 `simdgroup_matrix`. For bf16 the GEMM inputs Q, K and V are `bfloat` tiles when the compiler defines `__HAVE_BFLOAT__` (Metal 3.1); otherwise bf16 is widened to fp32 tiles. fp32 inputs use fp32 tiles. S, P and the O accumulator are fp32, so P is never rounded. Q and O stay in registers as four tiles each across D = 32. Per 64-key block, `S = Q · Kᵀ` is eight tiles, then `O += P · V`. The row max and sum use `simd_shuffle_xor` with lane masks 1 and 8, which relies on the Apple 8×8 lane layout that metal-flash-attention and MLX also use (`fa_hd32_tile_coord`).
- Loads: the head dimension is split as `d = 4i + t` across the four tiles `t`. Both GEMMs use the same split, so the result is unchanged. Each lane's Q, V and O elements become 8 contiguous values and its Kᵀ elements become 4 contiguous values per key, so every load is a 4-element vector load and O is written directly to device memory.
- Layout: row-major contiguous `(…, sequence, 32)`. Each K/V block is one contiguous span, copied by the 128 threads with consecutive addresses.
- Edge behavior: any query length and any key length work. Keys past `key_length` in the last block are not read (their K and V are zero in the stage) and get `p = 0`; query rows past `query_length` in the last block compute on zeros, never read the bias and are not stored. Each tensor is limited to fewer than 2³¹ elements because offsets are 32-bit.
- Math: bf16 bits are reinterpreted as `bfloat`, or widened to fp32 by bit placement. The scale (applied to S in fp32) and bias are folded into `log2(e)` so softmax uses `exp2`. `-inf` is detected from the bits, so fast math cannot fold the test away, and no infinity is used in arithmetic: the running max starts at the finite `-FLT_MAX`, masked keys are excluded with a per-element flag rather than a sentinel score, and the rescale exponent is clamped to `-FLT_MAX`. O and the running sum are multiplied by `exp2(m_old − m_new)` every block, which is exactly 1 when the max does not rise.
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
- Swift: the [Swift host](../../swift/README.md) compiles this file and launches `flash_attention_hd32` for bf16 and fp32. `bench/run_mac.sh swift` builds, tests, and times `[1, 8, 5184, 32]`.
- Benchmark: `bench/flash_attention_hd32.py` against `mx.fast.scaled_dot_product_attention` (unfused at D = 32) and PyTorch MPS SDPA. `bench/run_mac.sh <version>` tests, benchmarks and records a version. See [bench/README.md](../../bench/README.md).

```python
import mlx.core as mx
from metal_kernels.flash_attention import flash_attention_hd32

q, k, v = (mx.random.normal((1, 8, 5184, 32)).astype(mx.bfloat16) for _ in range(3))
out = flash_attention_hd32(q, k, v, backend="mlx")
```
