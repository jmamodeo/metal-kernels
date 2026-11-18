# metal_kernels

Python package for the metal-kernels collection. See the [repository README](../README.md) for the goal, the per-kernel workflow, and install instructions.

Kernel sources live in `../kernels`. That `.metal` file is the only copy. This package locates the file and returns its text. It does not compile or launch kernels, and importing `metal_kernels` does not import MLX or PyTorch.

## SAM 3 on MPS

`metal_kernels.integrations.sam3` patches the pinned SAM 3 checkout (`31cb7e624e4fe95301b1be1d4e809111593d2e45`) at import time so the image model and the tracker can run without CUDA. Call it before importing SAM 3:

```python
from metal_kernels.integrations.sam3 import build_image_model, install

install()  # cuda if present, else mps, else cpu
model = build_image_model(load_from_HF=False)
```

The ViT MLP uses linear plus GELU on CPU and MPS. On CUDA the original fused op is left in place. MPS itself is not executed by the tests in this repository. `bench/sam3_shapes/README.md` is the command that times the built-in kernels on a Mac.

```bash
python3 -m pip install -e ".[dev]"
```
