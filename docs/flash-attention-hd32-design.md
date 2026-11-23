# FlashAttention at head dim 32: design note

Linear: MTL-10 (parent MTL-8). This note started as the design before any implementation; [Our hd32 kernel plan](#our-hd32-kernel-plan) now records what shipped in v0 and v1.

## Why

The SAM 3 detector encoder calls SDPA at `[1, 8, 5184, 32]`. It runs 6 calls per pass, which is 2.6% of rank FLOPs ([sam3-op-inventory.md](sam3-op-inventory.md)). On the M3 Max, MLX takes 9.74 ms per call and PyTorch MPS takes 4.29 ms ([m3-max-2026-11-18.md](../bench/sam3_shapes/results/m3-max-2026-11-18.md)). MLX 0.32.4 has no fused kernel at D = 32. `has_fused_kernel` in `mlx/backend/metal/scaled_dot_product_attention.cpp` accepts head dims {64, 72, 80, 96, 128, 192, 256} (plus 512 on NAX) for query length > 8. Any other head dim takes the unfused matmul → softmax → matmul fallback, which writes the full `8 × 5184 × 5184` score matrix to memory.

Derived from those figures (arithmetic, not measurements): one call is `4·H·L²·D` ≈ 2.75e10 FLOPs, matching 1.651e11 / 6 in the inventory. That works out to about 6.4 TFLOP/s for MPS and 2.8 TFLOP/s for MLX. The score matrix the fallback writes is about 430 MB per call in bf16 (860 MB in fp32).

## What metal-flash-attention does

Source read: [philipturner/metal-flash-attention](https://github.com/philipturner/metal-flash-attention), commit `8671cdd` (README, `Sources/FlashAttention/Attention/**`, `GEMM/GEMMHeaders.swift`). MFA is single-head and JIT-generates the Metal source from Swift.

**Tiling and traversal.** Each threadgroup owns a block of rows along the parallelization dimension (queries, in the forward pass). It walks the traversal dimension (keys/values) in blocks. A third block dimension along D exists for large heads. The forward loop (`AttentionKernel+Source.swift`) runs `S = Q·Kᵀ`, masks the edge, updates the running max `m`, computes `correction = exp2(m_old − m_new)`, forms `P = exp2(S·scale − m)`, updates the running sum `l`, then does `O = O·correction + P·V`. After the last block it computes `O /= l` and writes `L = m + log2(l)`. Only one row block of S/P exists at a time. The full score matrix is never formed.

**Online softmax details.** The scale is folded into a single constant, `log2(e)/sqrt(D)`, so softmax uses `fast::exp2`. Row max and sum reductions inside an 8×8 tile use `simd_shuffle_xor` with lane masks 1 and 8. O is only rescaled when the max increases. Padding columns past the sequence end get a large negative *finite* value, `-(0.875/log2 e)·FLT_MAX`, not −inf, so that exp2 of 0 can't corrupt `l`.

**simdgroup_matrix.** All tiles are 8×8 `simdgroup_matrix_storage<T>`. Each lane holds 2 elements (`thread_elements()` is a `vec<T,2>`), placed by `morton_order(lane_id)`. The multiply is `__metal_simdgroup_matrix_8x8_multiply_accumulate`. Each simdgroup owns 8 query rows (`sidx * 8`), and a threadgroup has `32 · (parallelization / 8)` threads.

**Block sizes.** These come from a per-device table indexed by D (`AttentionDescriptor+Parameters.swift`). The forward mixed-precision row that covers D ≤ 32 is:

| Device | Parallelization | Traversal | Head block | Cached in registers |
| --- | ---: | ---: | ---: | --- |
| Apple9 (M3/M4) | 16 | 128 | 16 | Q, O |
| M1/M2 (≤ 96) | 32 | 128 | 32 | Q, O |

In MFA, this Apple9 row gives 64-thread threadgroups (2 simdgroups); our kernel uses 128-thread threadgroups instead (see the plan below). In MFA, threadgroup memory is the max over the staged operand blocks. For FP16 Q/K/V that is `128 × 16 × 2` = 4 KB, which I computed from `createThreadgroupMemoryAllocation`. On Apple9, MFA reads full K/V blocks straight from device memory into registers (`preferAsyncLoad = false`) and keeps threadgroup staging for edge blocks.

**Precision.** Mixed-precision forward keeps Q/K/V in FP16 in memory and registers, S in FP16, P in FP16, and the O accumulator in FP32. O and L are always written as FP32. MFA uses BF16 only for backward operands (dO, D, dS). On Apple9 it treats BF16→FP32 as native (`hasNativeBF16Casting = supportsFamily(.apple9)`). Its BF16 load puts the 16 bits in the high half of an FP32 word (`load_bfloat` in `GEMMHeaders.swift`), which makes the decode a bit placement, not arithmetic. The README notes that Metal's `bfloat` type rounds IEEE-style, which is costly on chips without hardware BF16.

## Our hd32 kernel plan

The target is Q, K, V `[1, 8, 5184, 32]` in bf16, output bf16, forward only, with no L output.

What shipped, by version:

- **v0:** per-row online softmax. One simdgroup per query row, one lane per head-dim column, `simd_sum` dot products, fp32 math, no threadgroup memory.
- **v1:** the tiled `simdgroup_matrix` kernel below with Bk = 64: 32 query rows and 128 threads per threadgroup, K/V loaded from device memory straight into registers, no threadgroup memory, bf16 GEMM inputs, fp32 P.
- **v2:** Bk = 128 was tried and was slower on the M3 Max (see [ROADMAP](ROADMAP.md)); it was not kept. The kernel is v1.
- **v6 (current kernel):** v1 with each 64-key K/V block copied once per threadgroup into 8 KB of bf16 threadgroup memory by all 128 threads, then shared by the four simdgroups, with two barriers per block. Measured on the M3 Max; the M3 target is met. Figures are in [docs/ROADMAP.md](ROADMAP.md).

1. **Grid.** There is one threadgroup per (batch · head, 32-query block): 5184/32 = 162 per head, 1296 in total, each with 128 threads (4 simdgroups × 8 rows). This is MFA's M1/M2 parallelization of 32 rather than its Apple9 value of 16. We add a head index instead of single-head dispatch.
2. **Traversal block.** The current kernel uses Bk = 64. 5184 = 81 × 64, so the target shape has no key-block tail. The edge-mask path is kept for other lengths, the cross-attention shape, and the tests. v2 tried Bk = 128 (the MFA Apple9 value) and was slower, likely from register pressure (16 S/P tiles instead of 8) and the 64-key tail that 5184 then required; that change was reverted.
3. **Registers.** Per thread, D = 32 is four 8×8 tiles across. That gives Q cached at 4 tiles × 2 = 8 values, O at 8 fp32, S/P for Bk = 64 at 8 tiles × 2 = 16 fp32, plus m and l. *Estimate:* about 35–40 live 32-bit values per thread before temporaries. MFA caches both Q and O at this D, so spilling is not expected at Bk = 64. v2's Bk = 128 raised S/P to 16 tiles (32 fp32) and was slower, consistent with register pressure.
4. **Threadgroup memory.** None through v5. As MFA does on Apple9, each lane loads its K and V tile elements straight from device memory into registers; the four simdgroups of a threadgroup read the same K/V block at about the same time, so the repeats hit the cache. No barriers. v6 instead stages each K/V block once per threadgroup in bf16 (8 KB, not the 16 KB fp32 stage of v1's slow first revision), so each block is read from device memory once per threadgroup rather than once per simdgroup, at the cost of two barriers per block.
5. **bf16 handling.** The GEMM inputs Q, K and V are `bfloat` tiles when the compiler defines `__HAVE_BFLOAT__` (Metal 3.1); otherwise bf16 is widened to fp32 by bit placement (`as_type<float>(uint(bits) << 16)`). S, P and the O accumulator are fp32, so P is never rounded. O is written as bf16 once at the end. Unlike MFA, we don't write O in FP32, because the SAM 3 caller wants bf16.
6. **Softmax.** Follow MFA: fold the scale into `log2(e)·scale`, use exp2, apply the shuffle-xor row reductions, rescale O only when the max rises, and mask with the finite-negative sentinel. A row whose bias is −inf everywhere must output 0, matching the MTL-9 reference, so guard `1/l` when `l == 0`.
7. **Hosts.** The single source file is `kernels/flash_attention_hd32/flash_attention_hd32.metal`.
   - MLX: `mx.fast.metal_kernel(header=<source>, source=<body calling the device function>)`, launched with `grid = (128 · 1296, 1, 1)` and `threadgroup = (128, 1, 1)`. MLX grids count threads. Check that MLX's bf16 binding type matches what the shader expects (`verbose=True` prints the generated source).
   - PyTorch MPS: `torch.mps.compile_shader(<source>)`, then `lib.flash_attention_hd32_bfloat(q, k, v, bias, out, dims, scale, threads=..., group_size=128)`, with contiguous bf16 tensors on `mps`.
   - Tests compare both against `reference/` from MTL-9 on small shapes, using the `metal` marker, and on the full shape.
8. **Stretch: decoder cross-attention.** Q is `[1, 8, 201, 32]`, K/V are `[1, 8, 5184, 32]`, and there is an additive bias of `[1, 8, 201, 5184]`.
   - Each lane loads its bias elements next to each K block and adds them to S before the max. The scale applies before the bias, as in `softmax(q·kᵀ·scale + bias)`.
   - Mask query rows past 201 on store.
   - The bias read (8·201·5184·2 B ≈ 16.7 MB in bf16) is bigger than K and V together (≈ 5.3 MB), so this shape is likely bandwidth-bound (estimate). Bias loads must be coalesced along the key axis.
   - With 201 queries there are only 7 × 8 = 56 threadgroups. If the GPU is underfilled, split the key axis into chunks that write partial (O, m, l), then run a small combine kernel. That is a design option, not yet measured.
9. **Benchmark.** Add the kernel to `bench/` against MLX SDPA (the fallback) and PyTorch MPS SDPA at the same shapes and dtype. Record results in `bench/results/` per the metrics rule. The goal is to beat 4.29 ms (the MPS figure) at the target shape.

## Open questions

- The shader uses `bfloat` simdgroup matrices when the compiler defines `__HAVE_BFLOAT__` and widens to fp32 tiles otherwise. Which path each host takes on the M3 Max is not yet confirmed (`verbose=True` on MLX, or a GPU capture).
- v1 uses 32 rows per threadgroup. 16 rows with direct device→register loads has not been measured.
