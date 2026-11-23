# Roadmap

## Goal

Build a collection of fast, tested Metal kernels for Apple silicon that anyone can install from Python or Swift. Every kernel has a NumPy reference, a Metal implementation checked against it, a Swift host, and a benchmark against the built-in op.

Kernels are picked from real workloads. Profile a model, find where the built-in Metal ops in MPS or MLX fall back to the CPU or lose clearly, and write a kernel there. A kernel that cannot beat or fill in for a built-in op does not belong here.

## Per-kernel workflow

Start from `kernels/_template`. The `.metal` file is the only copy of the shader.

Before writing a kernel, check whether MPS or MLX already covers the op. The benchmark against that built-in op is how a custom kernel justifies itself.

### 1. NumPy reference

Put a NumPy implementation in `reference/<name>.py`. NumPy only: no MLX and no PyTorch. The function documents the shapes and dtypes it accepts and is the oracle the tests compare against.

### 2. Metal kernel from Python

Add `kernels/<name>/<name>.metal` and a `README.md` that states inputs, outputs, shapes, dtypes, and threadgroup assumptions.

Call that source from Python with MLX `mx.fast.metal_kernel` and/or PyTorch MPS `torch.mps.compile_shader`. Compare outputs to the NumPy reference on Apple silicon, and record the tolerance in the kernel README. The pytest that launches the kernel uses the `metal` marker so it skips when the host is not macOS.

### 3. Swift host

After the Python check matches the reference, add a Swift API on the `MetalKernels` package that loads the same `.metal` file. The Swift package does not wrap the Python package. Both hosts read the file in `kernels/`.

### 4. Benchmark

Add a script under `bench/` that times the kernel against the built-in framework implementation (MPS, MLX, or Metal Performance Shaders) when one exists. Publish the command, the shapes, the dtypes, and the machine. Leave the script out when there is no built-in op to compare, and say that in the kernel README.

## Kernels

The earlier conv3x3 kernel was dropped on Nov 22, 2026 by John's decision. It is no longer in this repository.

### FlashAttention hd32

Fused FlashAttention-style scaled dot-product attention for head dim 32. The SAM 3 detector encoder calls it at `[1, 8, 5184, 32]` in bfloat16, six times per image. This is the only kernel planned for the collection.

| Step | Status |
| --- | --- |
| 1. NumPy reference | done: `reference/sdpa.py` |
| 2. Metal kernel from Python | done: v7.1 is the current kernel. Tuning log below. |
| 3. Swift host | done: `FlashAttentionHD32Kernel` in the `MetalKernels` package compiles the shared `.metal` file at runtime and launches the MPS entry points. `swift test` checks it against a CPU reference on macOS |
| 4. Benchmark | `bench/flash_attention_hd32.py` and `bench/run_mac.sh` done. v0, v1, the v2–v4 trials, v6 and v7 have been recorded on the M3 Max; the JSONL rows stay uncommitted until M4. v5 did not produce a timing row because it failed the MPS-SDPA error bound before the bench ran. Through v6 the harness timed a frozen copy of v1 (`bench/baselines/flash_attention_hd32_v1.metal`); from v7 it times a frozen copy of v6 (`bench/baselines/flash_attention_hd32_v6.metal`, the kernel merged in `b2e6847`) interleaved with the candidate and the built-in on every iteration, in a position- and order-balanced rotation, at 102 timed iterations plus 12 warmup by default, and records a speed gate. Same-iteration samples are paired: the candidate passes only if the bootstrap 95% CI (10,000 resamples, fixed seed) of the median candidate/baseline time ratio lies entirely below 1. The earlier rule (median below v1's and outside v1's p10–p90 range) flagged a true 5% gain only about half the time, because v1's MLX p10–p90 is about ±4%. The same paired CI of the MPS candidate over MPS SDPA is the M3 target check and does not gate the version; v6 and v7 meet that target. |

Figures below are same-record medians on the M3 Max at `[1, 8, 5184, 32]` bf16 unless noted.

| Version | Change | Outcome | PR |
| --- | --- | --- | --- |
| v0 | Per-row online softmax, one simdgroup per query row, bias, bf16 and fp32, MLX and MPS launchers | Shipped; metal tests pass on the M3 Max | #30 |
| v1 | Tiled simdgroup, 32 query rows × 64 keys, 128 threads, device-to-register K/V, bf16 GEMM inputs, fp32 P | Shipped. First revision (fp32 K/V stage, 16 KB) ran at 40.0 ms. Merged revision: 4.12 ms (MLX) vs MLX SDPA 8.41 ms, 4.17 ms (MPS) vs MPS SDPA 3.91 ms; max abs error 9.29e-4, equal to MPS SDPA | #31 |
| v2 | BK=128 | Slower (4.31/4.29 ms vs v1 4.12/4.17); reverted. 32×64 is the best of the tile sizes tried | Recorded in #33 |
| v3 | BQ=64, 8 simdgroups | No measurable gain (4.26/4.27 ms vs v1 4.12/4.17 from a different run; 4.26 is close to v1's p10–p90 4.10–4.24). Kernel change not kept | #35 closed unmerged |
| v4 | Skip the O and running-sum rescale when no row max in the simdgroup rises | No measurable gain (4.22/4.21 ms; 4.22 sits inside v1's p10–p90 4.10–4.24 from a different run). Kernel change not kept | #39 closed unmerged |
| v5 | Fold scale·log2(e) into Q at load, exp2 throughout | Failed the error-no-worse-than-MPS-SDPA bound (rounding scaled Q back to bf16 before the MMA). Encoder 6.655e-4 vs 4.731e-4, cross 4.262e-4 vs 2.439e-4, seed 11. Reference tolerances held. Kernel change not kept; no timing row | #45 closed unmerged |
| v6 | Each 64-key K/V block loaded once into threadgroup memory (8 KB for bf16, 16 KB for fp32), shared by the 4 simdgroups | Shipped; superseded by v7. Max abs error 9.289e-4, mean 2.603e-5, equal to MPS SDPA. Medians: MLX 3.837 ms vs v1 4.330 and MLX SDPA 8.366; MPS 3.803 ms vs v1 4.106 and MPS SDPA 3.833. Paired candidate/v1: MLX 0.8877 [0.8723, 0.9109], MPS 0.9242 [0.9224, 0.9272]. M3 MPS kernel/SDPA 0.9913 [0.9880, 0.9928]; the M3 target is met | #48 |
| v7 | v6 with the K/V stage double buffered (16 KB for bf16, 32 KB for fp32): next block's device loads issued before the current block's math, one barrier per block | Shipped; superseded by v7.1. Timed against the frozen v6; the paired gate passed on both backends. Medians: MLX 3.812 ms vs v6 3.882 and MLX SDPA 8.357; MPS 3.824 ms vs v6 3.866 and MPS SDPA 3.905. Paired candidate/v6: MLX 0.9823 [0.9804, 0.9835], MPS 0.9896 [0.9885, 0.9912]. M3 MPS kernel/SDPA 0.9798 [0.9788, 0.9807]. Max abs error 9.289e-4, equal to MPS SDPA. | #50 |
| v7.1 | v7 with a single-buffer fp32 K/V stage (16 KB for both dtypes instead of 32 KB for fp32); bf16 unchanged; fp32 keeps the register prefetch with two barriers per block | Current kernel. Not yet measured: `bench/run_mac.sh v7.1` (bf16, same code as v7) and `bench/run_mac.sh swift` (bf16 and fp32) | MTL-32 |

## Workloads

### SAM 3

[SAM 3](https://github.com/facebookresearch/sam3) is Meta's promptable segmentation model for images and video. Its published implementation is PyTorch and targets CUDA. It is the first workload profiled here, and the aim for it is a full image and video pass on Metal with no CUDA dependency.

#### Op inventory

Traced in [sam3-op-inventory.md](sam3-op-inventory.md). One random-init forward pass at 1008px is 6.379e12 rank FLOPs: matmul 75.8%, attention 18.4%, convolution 5.7%. The vision encoder is 84.6% of that. The write-up has the shapes, the CUDA blockers, and the source review of PyTorch MPS 2.15.0 and MLX 0.32.4.

MPS and MLX already have matmul, SDPA, convolution, layer norm, and GELU. The M3 Max timings in [bench/sam3_shapes/results](../bench/sam3_shapes/results/m3-max-2026-11-18.md) show no CPU fallbacks. The only kernel being written is FlashAttention hd32: MLX is about 2x slower than PyTorch on the detector attention at head size 32, and MLX fused SDPA falls back at that head dim. A 3×3 convolution kernel was started earlier because PyTorch MPS is about 2x slower than MLX on the 3×3 and transposed 2×2 convolutions; John dropped that kernel on Nov 22, 2026. `aten._addmm_activation` is a call to split into linear plus GELU, not a new kernel, until a profile says otherwise.

#### MPS baseline

`metal_kernels.integrations.sam3.install()` patches the pinned SAM 3 checkout so the image path and the tracker run on CPU or MPS with built-in ops. It does not fork SAM 3. The Mac timings for the inventory's top shapes are [bench/sam3_shapes](../bench/sam3_shapes/README.md).

#### Later

Weight loading and a demo sit on top of the baseline above.
