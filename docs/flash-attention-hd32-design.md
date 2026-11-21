# FlashAttention at head dim 32: design note

Linear: MTL-10 (parent MTL-8). This note covers design only. There is no implementation in this change.

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

On M3 this gives 64-thread threadgroups (2 simdgroups). Threadgroup memory is the max over the staged operand blocks. For FP16 Q/K/V that is `128 × 16 × 2` = 4 KB, which I computed from `createThreadgroupMemoryAllocation`. On Apple9, MFA reads full K/V blocks straight from device memory into registers (`preferAsyncLoad = false`) and keeps threadgroup staging for edge blocks.

**Precision.** Mixed-precision forward keeps Q/K/V in FP16 in memory and registers, S in FP16, P in FP16, and the O accumulator in FP32. O and L are always written as FP32. MFA uses BF16 only for backward operands (dO, D, dS). On Apple9 it treats BF16→FP32 as native (`hasNativeBF16Casting = supportsFamily(.apple9)`). Its BF16 load puts the 16 bits in the high half of an FP32 word (`load_bfloat` in `GEMMHeaders.swift`), which makes the decode a bit placement, not arithmetic. The README notes that Metal's `bfloat` type rounds IEEE-style, which is costly on chips without hardware BF16.

## Our hd32 kernel plan

The target is Q, K, V `[1, 8, 5184, 32]` in bf16, output bf16, forward only, with no L output.

1. **Grid.** There is one threadgroup per (head, 16-query block): 5184/16 = 324 per head, 2592 in total, each with 64 threads (2 simdgroups × 8 rows). This is the MFA Apple9 row. We add a head index instead of single-head dispatch.
2. **Traversal block.** Start at Bk = 64. 5184 = 81 × 64, so the target shape has no key edge and needs no edge mask. Bk = 128 (the MFA value) leaves a 64-key tail. Keep the edge-mask path anyway for the cross-attention shape and for tests. Sweep Bk ∈ {32, 64, 128} in the benchmark.
3. **Registers.** Per thread, D = 32 is four 8×8 tiles across. That gives Q cached at 4 tiles × 2 = 8 fp32, O at 8 fp32, S/P for Bk = 64 at 8 tiles × 2 = 16 fp32, plus m and l. *Estimate:* about 35–40 live 32-bit values per thread before temporaries. MFA caches both Q and O at this D, so spilling is not expected. Confirm with the Xcode GPU capture.
4. **Threadgroup memory.** One K block and one V block of `64 × 32 × 2 B` = 4 KB each, so 8 KB if both are staged. That is an estimate from the shape, not a measurement. Version 0 stages K/V through threadgroup memory so both host paths behave the same. A later version tries MFA's direct device→register reads on M3 and records the difference.
5. **bf16 handling.** Load bf16, widen to fp32 by bit placement (`as_type<float>(uint(bits) << 16)`), and run both GEMMs on `simdgroup_float8x8` with fp32 S, P and O. Write O as bf16 once at the end. Unlike MFA, we don't write O in FP32, because the SAM 3 caller wants bf16. Later work: a half-precision P·V with fp32 accumulate, gated by the NumPy tolerance.
6. **Softmax.** Follow MFA: fold the scale into `log2(e)·scale`, use exp2, apply the shuffle-xor row reductions, rescale O only when the max rises, and mask with the finite-negative sentinel. A row whose bias is −inf everywhere must output 0, matching the MTL-9 reference, so guard `1/l` when `l == 0`.
7. **Hosts.** The single source file is `kernels/flash_attention_hd32/flash_attention_hd32.metal`.
   - MLX: `mx.fast.metal_kernel(header=<source>, source=<body calling the device function>)`, launched with `grid = (64 · 2592, 1, 1)` and `threadgroup = (64, 1, 1)`. MLX grids count threads, as in `conv3x3.py`. Check that MLX's bf16 binding type matches what the shader expects (`verbose=True` prints the generated source).
   - PyTorch MPS: `torch.mps.compile_shader(<source>)`, then `lib.flash_attention_hd32(q, k, v, out, dims, threads=..., group_size=[64,1,1])`, with contiguous bf16 tensors on `mps`.
   - Tests compare both against `reference/` from MTL-9 on small shapes, using the `metal` marker, and on the full shape.
8. **Stretch: decoder cross-attention.** Q is `[1, 8, 201, 32]`, K/V are `[1, 8, 5184, 32]`, and there is an additive bias of `[1, 8, 201, 5184]`.
   - Load the bias tile `[16 × Bk]` next to each K block and add it to S before the max. The scale applies before the bias, as in `softmax(q·kᵀ·scale + bias)`.
   - Mask query rows past 201 on store.
   - The bias read (8·201·5184·2 B ≈ 16.7 MB in bf16) is bigger than K and V together (≈ 5.3 MB), so this shape is likely bandwidth-bound (estimate). Bias loads must be coalesced along the key axis.
   - With 201 queries there are only 13 × 8 = 104 threadgroups. If the GPU is underfilled, split the key axis into chunks that write partial (O, m, l), then run a small combine kernel. That is a design option, not yet measured.
9. **Benchmark.** Add the kernel to `bench/` against MLX SDPA (the fallback) and PyTorch MPS SDPA at the same shapes and dtype. Record results in `bench/results/` per the metrics rule. The goal is to beat 4.29 ms (the MPS figure) at the target shape.

## Open questions

- Does MLX's `metal_kernel` and `compile_shader` accept `bfloat` simdgroup matrices on our macOS/Xcode? Bf16 multiplies would skip the widen step.
- Is 16 rows per threadgroup still best for the 324-blocks-per-head grid on M3 Max, or does 32 rows (as on M1) win? Sweep both.
