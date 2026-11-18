# metal_kernels

Python package for the metal-kernels collection. See the [repository README](../README.md) for the goal, the per-kernel workflow, and install instructions.

Kernel sources live in `../kernels`. That `.metal` file is the only copy. This package locates the file and returns its text. It does not compile or launch kernels, and it does not import MLX or PyTorch.

```bash
python3 -m pip install -e ".[dev]"
```
