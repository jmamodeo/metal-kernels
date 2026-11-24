# metal-kernels

[![CI](https://github.com/jmamodeo/metal-kernels/actions/workflows/ci.yml/badge.svg)](https://github.com/jmamodeo/metal-kernels/actions/workflows/ci.yml)

A community collection of Apple Metal GPU kernels for Apple silicon. Each kernel ships with a NumPy reference that defines the math, Python bindings through MLX and PyTorch MPS, a native Swift host, and benchmarks against the built-in framework op.

Kernels are chosen by profiling real models and writing a custom kernel only where the built-in Metal ops fall short. The first workload profiled is Meta's [SAM 3](https://github.com/facebookresearch/sam3) (Segment Anything 3). Its integration and shape benchmarks live in `metal_kernels.integrations.sam3` and `bench/sam3_shapes/`.

On the SAM 3 detector encoder attention at `[1, 8, 5184, 32]` bf16, MLX is about 2× slower than PyTorch: 9.74 ms versus 4.29 ms per call on an M3 Max (PyTorch 2.15.0, MLX 0.32.4). At head dim 32, MLX's fused SDPA is not used; `mx.fast.scaled_dot_product_attention` falls back to an unfused matmul → softmax → matmul path. Those figures are from the [shape timings](bench/sam3_shapes/results/m3-max-2026-11-18.md) and the [design note](docs/flash-attention-hd32-design.md).

The collection currently has one kernel, [`flash_attention_hd32`](kernels/flash_attention_hd32/README.md): Metal FlashAttention for head dim 32. More kernels get added when profiling finds a gap. Version 7.1 is the current kernel. It launches from MLX and PyTorch MPS, and from the [Swift host](swift/README.md). On one measured configuration — Apple M3 Max, macOS 26.7.3, torch 2.15.0, MLX 0.32.4, at `[1, 8, 5184, 32]` bf16 — it is about 2% faster than MPS SDPA (paired MPS kernel/SDPA median ratio 0.9828, 95% CI [0.9808, 0.9849]; 102 pairs, 10,000 resamples).

```python
from metal_kernels.flash_attention import flash_attention_hd32

out = flash_attention_hd32(q, k, v, bias=None, scale=None, backend="mlx")  # or backend="mps"
```

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

Installing these packages does not download any model weights.

### Python

The published package is on PyPI and requires Python 3.10 or newer. This is the primary way to install the collection:

```bash
python3 -m pip install metal-kernels
```

The base package has no required dependencies and is enough to import the loader. Optional extras add the hosts and tools used to launch or test a kernel:

| Extra | What it installs | When it is used |
| --- | --- | --- |
| `dev` | pytest, NumPy, build | running the test suite |
| `reference` | NumPy | running oracle code |
| `mlx` | MLX | Apple silicon, once a kernel calls `mx.fast.metal_kernel` |
| `torch` | PyTorch | once a kernel calls `torch.mps.compile_shader` |

```bash
python3 -m pip install "metal-kernels[mlx]"
python3 -m pip install "metal-kernels[torch]"
```

MLX wheels and the PyTorch MPS backend are macOS only. A kernel's Python launch path imports them when it runs. They are not installed by the base package.

A wheel or sdist install vendors the `.metal` sources inside the package. The loader also honors `METAL_KERNELS_ROOT` when that variable points at a `kernels` directory.

#### Development install from a checkout

Use an editable install only when you are changing the package or running the test suite from this repository:

```bash
cd python
python3 -m pip install -e ".[dev]"
```

From the repository root the same install is `python3 -m pip install -e "./python[dev]"`. An editable install from a checkout finds `kernels/` at the repository root.

Then, from the repository root:

```bash
pytest
python -m metal_kernels
```

`pytest` skips Metal execution tests when no Metal device is available. `python -m metal_kernels` prints the registry.

### Swift

The package name is `MetalKernels`. It needs Swift 6.0 or newer. See [swift/README.md](swift/README.md) for the host that compiles and launches `flash_attention_hd32` through the Metal API, plus the timing mode. From the repository root:

```bash
swift test
```

On a Mac, `bench/run_mac.sh swift` builds the package, runs the tests, and times `[1, 8, 5184, 32]`.

As a local SwiftPM dependency:

```swift
.package(path: "/path/to/metal-kernels")
```

There is no version tag yet.

`swift test` also checks that the bundled `kernels/` copy matches the checkout.

SwiftPM prints `Found unhandled resource at <repo>/kernels` while doing this. The resource lives outside the Swift target directory, which is what that line reports, and the directory is still bundled.

## Results

On one measured configuration — Apple M3 Max, macOS 26.7.3, torch 2.15.0, MLX 0.32.4, at `[1, 8, 5184, 32]` bf16 — the current kernel is about 2% faster than MPS SDPA (paired MPS kernel/SDPA median ratio 0.9828, 95% CI [0.9808, 0.9849]; 102 pairs, 10,000 resamples). Version 7.1 was recorded at 306bcea. bf16 passed the 1.01 gate versus the frozen v7 (MLX 1.0060 [1.0033, 1.0078], MPS 1.0072 [1.0045, 1.0094]). fp32 versus the frozen v6 passed on MLX (1.0036 [1.0022, 1.0071]) and did not pass on MPS (1.0081 [1.0046, 1.0101]; 1.0101 > 1.01). That MPS fp32 miss was accepted as a documented one-off waiver; the rationale is the source-identical v6 fp32 loop and restored Swift fp32 latency (3.990 ms versus v6's 3.985 ms), not proof of equivalence from the CI. The general 1.01 gate is unchanged. This is not a claim that fp32 outperforms v6.

Selected recorded runs at that same bf16 shape, taken from the committed JSONL. v1 is the merged tiled revision, not the earlier 40 ms staging trial. The paired MPS kernel/SDPA intervals are the same figures already recorded for v6, v7 and v7.1.

| Version | MLX kernel ms | PyTorch kernel ms | MPS kernel/SDPA (bf16) |
| --- | ---: | ---: | --- |
| v0 | 60.035 | 60.424 | — |
| v1 | 4.123 | 4.168 | — |
| v6 | 3.837 | 3.803 | 0.9913 [0.9880, 0.9928] |
| v7 | 3.812 | 3.824 | 0.9798 [0.9788, 0.9807] |
| v7.1 (current) | 3.797 | 3.794 | 0.9828 [0.9808, 0.9849] |

The full reviewable table is generated from `bench/results/flash_attention_hd32.jsonl` (append-only recorded runs, including trials that were not kept) by:

```bash
python tools/results_table.py flash_attention_hd32
```

That writes [bench/results/flash_attention_hd32.md](bench/results/flash_attention_hd32.md). The same table is in the [kernel README](kernels/flash_attention_hd32/README.md#results). The JSONL is committed with the table so the numbers can be regenerated.

## Contributing

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
