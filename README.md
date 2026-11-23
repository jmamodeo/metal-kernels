# metal-kernels

A community collection of Apple Metal GPU kernels for Apple silicon. Each kernel ships with a NumPy reference that defines the math, Python bindings through MLX and PyTorch MPS, a native Swift host, and benchmarks against the built-in framework op.

Kernels are chosen by profiling real models and writing a custom kernel only where the built-in Metal ops fall short. The first workload profiled is Meta's [SAM 3](https://github.com/facebookresearch/sam3) (Segment Anything 3). Its integration and shape benchmarks live in `metal_kernels.integrations.sam3` and `bench/sam3_shapes/`.

The registry holds one kernel, [`flash_attention_hd32`](kernels/flash_attention_hd32/README.md): Metal FlashAttention for head dim 32, the only kernel this collection will ship. Version 1 (the tiled `simdgroup_matrix` kernel) launches from MLX and PyTorch MPS:

```python
from metal_kernels.flash_attention import flash_attention_hd32

out = flash_attention_hd32(q, k, v, bias=None, scale=None, backend="mlx")  # or backend="mps"
```

The earlier conv3x3 kernel was dropped on Nov 22, 2026 by John's decision.

## Layout

```
Package.swift              Swift package MetalKernels (manifest at the repo root)
kernels/<name>/            one directory per kernel
kernels/_template/         files to copy for a new kernel; not a kernel
python/                    pip package metal_kernels
swift/                     Swift sources and tests
reference/                 NumPy oracles
tests/                     pytest
bench/                     benchmark scripts
docs/ROADMAP.md            goal, workflow, kernel status, workloads
```

The `.metal` file in `kernels/<name>/` is the only copy of that kernel. Python and Swift load that file. Kernel directories whose names start with `_` are ignored, so `_template` is never registered.

`Package.swift` lives at the repository root so the Swift package can bundle `kernels/`. SwiftPM copies resources only from inside the package directory. A manifest kept under `swift/` would leave `kernels/` outside the package. The sources and tests stay in `swift/`. The resource rule is `.copy`, which keeps the `.metal` files as source text instead of compiling them into a `default.metallib`.

## Workflow

Each kernel follows the same four steps. [docs/ROADMAP.md](docs/ROADMAP.md) has the longer version and the status of each kernel.

1. **NumPy reference** in `reference/<name>.py`. This is the math, and it is the oracle the tests check against.
2. **Metal kernel** in `kernels/<name>/<name>.metal`, called from Python through MLX (`mx.fast.metal_kernel`) and/or PyTorch MPS (`torch.mps.compile_shader`), and checked against the reference on Apple silicon.
3. **Swift host** in the `MetalKernels` package, over the same `.metal` file.
4. **Benchmark** in `bench/` against the built-in framework op, when one exists.

Write the reference and the Python check before the Swift host. The Swift package is a native API, not a wrapper around the Python package.

## Install

Installing these packages does not download any model weights. The commands below are the install paths now, and they stay the same once kernels exist.

### Python

From a checkout:

```bash
cd python
python3 -m pip install -e ".[dev]"
```

From the repository root the same install is `python3 -m pip install -e "./python[dev]"`.

The base package has no required dependencies. `pip install -e .` inside `python/` is enough to import the loader. The `dev` extra adds pytest and NumPy.

| Extra | What it installs | When it is used |
| --- | --- | --- |
| `dev` | pytest, NumPy | running the test suite |
| `reference` | NumPy | running oracle code |
| `mlx` | MLX | Apple silicon, once a kernel calls `mx.fast.metal_kernel` |
| `torch` | PyTorch | once a kernel calls `torch.mps.compile_shader` |

MLX wheels and the PyTorch MPS backend are macOS only. A kernel's Python launch path imports them when it runs. They are not installed by the base package or the `dev` extra.

Then, from the repository root:

```bash
pytest
python -m metal_kernels
```

`pytest` skips Metal execution tests when the host is not macOS. `python -m metal_kernels` prints the registry.

An editable install from a checkout is the supported way to find `kernels/`. The loader also honors `METAL_KERNELS_ROOT` when that variable points at the `kernels` directory. A non-editable wheel does not yet vendor the `.metal` sources.

### Swift

The package name is `MetalKernels`. It needs Swift 6.0 or newer. From the repository root:

```bash
swift test
```

As a local SwiftPM dependency:

```swift
.package(path: "/path/to/metal-kernels")
```

There is no version tag yet.

`swift test` builds the source loader and checks that the bundled `kernels/` copy matches the checkout. On macOS it also launches `flash_attention_hd32` on the GPU and checks it against a CPU reference. The Metal host is compiled only where `import Metal` is available, so the package still builds elsewhere; the launch tests skip there.

```swift
import MetalKernels

let attention = try FlashAttentionHD32Kernel()  // compiles kernels/flash_attention_hd32/flash_attention_hd32.metal
let out = try attention.run(
    q: q, k: k, v: v,
    qShape: [1, 8, 5184, 32], kShape: [1, 8, 5184, 32],
    dtype: .bfloat16)
```

SwiftPM prints `Found unhandled resource at <repo>/kernels` while doing this. The resource lives outside the Swift target directory, which is what that line reports, and the directory is still bundled.

## Contributing

Every merge into main needs John Amodeo's OK, and John is the sole commit author/committer (no co-author trailers).

Copy [kernels/_template](kernels/_template/README.md) to `kernels/<name>/` and add the NumPy reference, the test, and the hosts described there. Keep the shader in `kernels/` only.

Prefer one kernel per change so the reference, the shader, and the test can be reviewed together. Tests that launch Metal belong behind the `metal` marker so they skip on machines without it. Record benchmark numbers you measured; the `bench/` placeholder does not invent timings.

The project is released under the Apache License 2.0. See [LICENSE](LICENSE).

## Citation

If you use metal-kernels in your work, cite it with the metadata in [CITATION.cff](CITATION.cff). GitHub's "Cite this repository" button generates BibTeX and APA from that file.

```bibtex
@software{amodeo_metal_kernels,
  author = {Amodeo, John},
  title = {metal-kernels},
  url = {https://github.com/jmamodeo/metal-kernels},
  license = {Apache-2.0}
}
```
