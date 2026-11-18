# Roadmap

## Goal

Run Meta's SAM 3 (Segment Anything 3) on Apple silicon with Metal, with no CUDA dependency.

SAM 3 is Meta's model for promptable concept segmentation in images and video. A short text phrase, image exemplars, or visual prompts (points, boxes, masks) produce instance masks. The official model pairs a detector with a memory-based video tracker that share a vision encoder, and a presence head that separates recognition from localization. The published implementation is PyTorch and targets CUDA.

This repository collects the custom Metal kernels that path still needs after the ops already covered by Metal Performance Shaders, MPS, and MLX. Kernels land one at a time. A full model runner comes after the kernels a forward pass actually needs.

Sources:

- Model card and code: <https://github.com/facebookresearch/sam3>
- Research overview: <https://ai.meta.com/research/sam3/>

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

## SAM 3 op inventory

Traced in [sam3-op-inventory.md](sam3-op-inventory.md). One random-init forward pass at 1008px is 6.379e12 rank FLOPs: matmul 75.8%, attention 18.4%, convolution 5.7%. The vision encoder is 84.6% of that. The write-up has the shapes, the CUDA blockers, and the source review of PyTorch MPS 2.15.0 and MLX 0.32.4.

No kernel is queued from this pass yet. MPS and MLX already have matmul, SDPA, convolution, layer norm, and GELU in the sources reviewed there, and none of that has been timed on a Mac. The first measurement is the vision-encoder SDPA at `[1, 16, 5184, 64]` and the MLP GEMMs at 5184 tokens. A custom kernel is justified only where that measurement shows a fallback or a large gap. The tracker memory attention (head dim 256, K length 5184 or 5188) is the most likely miss. `aten._addmm_activation` is a call to split into linear plus GELU, not a new kernel, until a profile says otherwise.

## MPS baseline

`metal_kernels.integrations.sam3.install()` patches the pinned SAM 3 checkout so the image path and the tracker run on CPU or MPS with built-in ops. It does not fork SAM 3. The Mac timings for the inventory's top shapes are [bench/sam3_shapes](../bench/sam3_shapes/README.md). A Mac run of that script is still what confirms MPS; this repository has not executed MPS.

## Later

Weight loading and a demo sit on top of the baseline above. Custom kernels wait on the Mac timings: a kernel is justified where a built-in falls back or is far from the shape in the inventory.
