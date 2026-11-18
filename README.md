# metal-kernels

Community collection of Apple Metal GPU kernels. The long-term goal is to run Meta's [SAM 3](https://github.com/facebookresearch/sam3) (Segment Anything 3) entirely on Metal on Apple silicon.

The kernel registry is empty. This repository is the skeleton: the shared layout, the per-kernel workflow, and Python and Swift packages that both read `kernels/`. Nothing here compiles a shader or runs SAM 3.

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
docs/ROADMAP.md            SAM 3 goal, workflow, op inventory
```

The `.metal` file in `kernels/<name>/` is the only copy of that kernel. Python and Swift load that file. Kernel directories whose names start with `_` are ignored, so `_template` is never registered.

`Package.swift` lives at the repository root so the Swift package can bundle `kernels/`. SwiftPM copies resources only from inside the package directory. A manifest kept under `swift/` would leave `kernels/` outside the package. The sources and tests stay in `swift/`. The resource rule is `.copy`, which keeps the `.metal` files as source text instead of compiling them into a `default.metallib`.

## Workflow

Each kernel follows the same four steps. [docs/ROADMAP.md](docs/ROADMAP.md) has the longer version and the SAM 3 notes.

1. **NumPy reference** in `reference/<name>.py`. This is the math, and it is the oracle the tests check against.
2. **Metal kernel** in `kernels/<name>/<name>.metal`, called from Python through MLX (`mx.fast.metal_kernel`) and/or PyTorch MPS (`torch.mps.compile_shader`), and checked against the reference on Apple silicon.
3. **Swift host** in the `MetalKernels` package, over the same `.metal` file.
4. **Benchmark** in `bench/` against the built-in framework op, when one exists.

Write the reference and the Python check before the Swift host. The Swift package is a native API, not a wrapper around the Python package.

## Install

Installing these packages does not download SAM 3 weights and does not run the model. The commands below are the install paths now, and they stay the same once kernels exist.

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

MLX wheels and the PyTorch MPS backend are macOS only. They are unused today, and they are not installed by the base package or the `dev` extra.

Then, from the repository root:

```bash
pytest
python -m metal_kernels
```

`pytest` skips Metal execution tests when the host is not macOS. `python -m metal_kernels` prints the registry, which is empty.

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

`swift test` builds the source loader and checks that the bundled `kernels/` copy matches the checkout. The package does not import Metal. Running a kernel still requires macOS.

SwiftPM prints `Found unhandled resource at <repo>/kernels` while doing this. The resource lives outside the Swift target directory, which is what that line reports, and the directory is still bundled.

## Contributing

Copy [kernels/_template](kernels/_template/README.md) to `kernels/<name>/` and add the NumPy reference, the test, and the hosts described there. Keep the shader in `kernels/` only.

Prefer one kernel per change so the reference, the shader, and the test can be reviewed together. Tests that launch Metal belong behind the `metal` marker so they skip on machines without it. Record benchmark numbers you measured; the `bench/` placeholder does not invent timings.

The project is released under the Apache License 2.0. See [LICENSE](LICENSE).
