# SAM 3 shape timings

Times the built-in PyTorch and MLX ops at the shapes in [docs/sam3-op-inventory.md](../../docs/sam3-op-inventory.md). Run it on an Apple silicon Mac. A Linux machine has no MPS device, and a CPU timing of these shapes is not a stand-in.

The script does not set `PYTORCH_ENABLE_MPS_FALLBACK`. An op MPS does not implement raises, and the CSV records that as `error`. If the variable is already set in the environment, a CPU fallback is recorded as `fallback` together with the warning text. Leave it unset for the run that decides whether a kernel is required.

## Install and run

From the repository root, on the Mac:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e "python[mac]"
python bench/sam3_shapes/profile.py
```

`python[mac]` installs PyTorch, MLX, and NumPy. PyTorch warns `Failed to initialize NumPy` when NumPy is absent, and the profiler needs it. Use the default pip index. The CPU-only PyTorch index has no MPS backend.

That command writes:

- `bench/sam3_shapes/output/shapes.csv`
- `bench/sam3_shapes/output/summary.md`

`--list` prints the cases and does not time them.

## Optional full image forward

This also builds a random-init SAM 3 image model through `metal_kernels.integrations.sam3` and runs one text prompt. It needs the pinned checkout. Weights are not downloaded.

```bash
pip install torchvision
git clone https://github.com/facebookresearch/sam3 /tmp/sam3
git -C /tmp/sam3 checkout 31cb7e624e4fe95301b1be1d4e809111593d2e45
pip install -e /tmp/sam3
pip install einops==0.8.2 pycocotools==2.0.11 psutil==7.2.2 "numpy>=1.26,<2"
python bench/sam3_shapes/profile.py --full-forward
```

Triton is not required. The compatibility layer stubs it when `import triton` fails. `fill_hole_area` stays 0 on this path, so the distance-transform kernel is not called.

`--skip-mlx` or `--skip-pytorch` drops one backend. `--warmup` and `--iters` change the timed loop (defaults 3 and 10).
