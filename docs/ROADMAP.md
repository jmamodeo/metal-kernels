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
| 2. Metal kernel from Python | v0 (per-row online softmax, one simdgroup per query row, bias, bf16 and fp32, MLX and MPS launchers) done and passing the `metal` tests on the M3 Max. v1 (tiled simdgroup) merged: its first revision (16-row threadgroups, K/V staged as fp32 in 16 KB of threadgroup memory, fp32 GEMMs) passed the `metal` tests but ran at 40.0 ms against MLX SDPA's 8.40 ms; the merged revision (32 query rows, 128 threads, device-to-register K/V, bf16 GEMM inputs, fp32 P, 64-key blocks) passes them and runs at 4.12 ms (MLX path) against MLX SDPA's 8.41 ms, with error equal to MPS SDPA's. v2 (BK=128) was tried on the M3 Max and was slower, likely from register pressure (16 S/P tiles) and the 64-key tail; same-record medians were 4.31 ms (our MLX) and 4.29 ms (our MPS) against v1's 4.12 and 4.17, with MLX SDPA at 8.39 ms and MPS SDPA at 3.87 ms; max abs error was unchanged at 9.29e-4. The kernel was reverted to v1 (BK=64). v3 (BQ=64, 8 simdgroups) was tried on the M3 Max and was slower; same-record medians were 4.26 ms (our MLX) and 4.27 ms (our MPS) against v1's 4.12 and 4.17, with MLX SDPA at 8.42 ms and MPS SDPA at 3.91 ms; max abs error was unchanged at 9.29e-4. PR #35 was closed unmerged. With v2 also slower, the 32×64 tile sits at the optimum for this structure. |
| 3. Swift host | not started |
| 4. Benchmark | `bench/flash_attention_hd32.py` and `bench/run_mac.sh` done. v0, v1, and the v2 and v3 trials have been recorded on the M3 Max; the JSONL rows stay uncommitted until M4. |

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
