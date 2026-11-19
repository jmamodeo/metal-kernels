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

### conv3x3

A 3×3 convolution, first measured in the SAM 3 FPN neck and the mask-head pixel decoder. Input `[1, 256, 288, 288]`, weight `[256, 256, 3, 3]`, stride 1, padding 1, dilation 1, groups 1, with bias. The traced forward ran it in bfloat16. The reference computes float32. Rank FLOPs are 9.784e10 per call.

| Step | Status |
| --- | --- |
| 1. NumPy reference | done (`reference/conv3x3.py`) |
| 2. Metal kernel from Python | not started |
| 3. Swift host | not started |
| 4. Benchmark | not started |

## Workloads

### SAM 3

[SAM 3](https://github.com/facebookresearch/sam3) is Meta's promptable segmentation model for images and video. Its published implementation is PyTorch and targets CUDA. It is the first workload profiled here, and the aim for it is a full image and video pass on Metal with no CUDA dependency.

#### Op inventory

Traced in [sam3-op-inventory.md](sam3-op-inventory.md). One random-init forward pass at 1008px is 6.379e12 rank FLOPs: matmul 75.8%, attention 18.4%, convolution 5.7%. The vision encoder is 84.6% of that. The write-up has the shapes, the CUDA blockers, and the source review of PyTorch MPS 2.15.0 and MLX 0.32.4.

The first kernel is [conv3x3](../kernels/conv3x3/README.md). Step 1, the NumPy reference, is done. The Metal shader is not started. MPS and MLX already have matmul, SDPA, convolution, layer norm, and GELU. The M3 Max timings in [bench/sam3_shapes/results](../bench/sam3_shapes/results/m3-max-2026-11-18.md) show no CPU fallbacks. PyTorch MPS is about 2x slower than MLX on the 3×3 and transposed 2×2 convolutions, which is why conv3x3 comes first. MLX is about 2x slower than PyTorch on the detector attention at head size 32. `aten._addmm_activation` is a call to split into linear plus GELU, not a new kernel, until a profile says otherwise.

#### MPS baseline

`metal_kernels.integrations.sam3.install()` patches the pinned SAM 3 checkout so the image path and the tracker run on CPU or MPS with built-in ops. It does not fork SAM 3. The Mac timings for the inventory's top shapes are [bench/sam3_shapes](../bench/sam3_shapes/README.md).

#### Later

Weight loading and a demo sit on top of the baseline above.
