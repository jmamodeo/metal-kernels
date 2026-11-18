# Kernel template

Copy this directory to `kernels/<name>/` when adding a kernel. `_template` itself is not a kernel: loaders skip any directory whose name starts with `_`.

Expected files in `kernels/<name>/`:

| File | Role |
| --- | --- |
| `<name>.metal` | The shader. This is the only copy. Rename `kernel.metal` when you copy the template. |
| `README.md` | Inputs, outputs, shapes, dtypes, and threadgroup assumptions. |

The loader requires exactly one `*.metal` file in the directory. Other hosts live outside this directory:

| Piece | Path |
| --- | --- |
| NumPy oracle | `reference/<name>.py` (start from `reference/_template.py`) |
| Pytest | `tests/test_<name>.py`, Metal launches marked `metal` |
| Python host | MLX `mx.fast.metal_kernel` and/or PyTorch MPS `torch.mps.compile_shader`, calling `metal_kernels.metal_source` |
| Swift host | `MetalKernels` in `swift/Sources/MetalKernels` |
| Benchmark | `bench/<name>.py`, when a built-in framework op exists |

Use a short lowercase name. Match it across the directory, the `.metal` file, and the reference module.

## Status

Template only. No kernel function is defined.

## What it computes

Describe the op in a few sentences, including the formula the NumPy reference implements.

## Inputs

| Name | Shape | Dtype | Notes |
| --- | --- | --- | --- |
| | | | |

## Outputs

| Name | Shape | Dtype | Notes |
| --- | --- | --- | --- |
| | | | |

## Threadgroup assumptions

- Grid size:
- Threadgroup size:
- Threadgroup memory:
- Layout the kernel assumes (contiguous dimensions, vector width):
- Edge behavior (bounds checks, tail elements):

## Numerical check

- Reference: `reference/<name>.py`
- Tolerance, per dtype:
- Shapes covered by the test:

## Hosts

- Python (MLX): not started
- Python (PyTorch MPS): not started
- Swift: not started
- Benchmark against the built-in op: not started
