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

Not started. No model graph has been traced in this repository, so there are no kernel tasks yet.

The table will be filled by walking the official SAM 3 PyTorch modules — the shared vision encoder, the detector and its presence head, and the memory-based tracker — and marking each op that still needs a kernel after MPS and MLX. Rows land here when that pass happens, each with a kernel name and a status.

| Op | Where it shows up | Framework coverage | Kernel in this repo |
| --- | --- | --- | --- |

## Later

Weight loading, a SAM 3 forward pass, and a demo sit on top of the kernels above. They wait until the inventory says which kernels a forward pass needs.
