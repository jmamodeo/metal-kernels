# metal-kernels

Python package for [metal-kernels](https://github.com/jmamodeo/metal-kernels), a collection of Apple Metal GPU kernels for Apple silicon. The collection currently has FlashAttention for head dim 32.

```bash
python3 -m pip install metal-kernels
```

The base package has no required dependencies. Optional extras (`mlx`, `torch`, `reference`, `dev`, `mac`) are listed in the [repository README](https://github.com/jmamodeo/metal-kernels#install). From a checkout, the development install is `python3 -m pip install -e "./python[dev]"` at the repository root.

The `.metal` file in [`kernels/`](https://github.com/jmamodeo/metal-kernels/tree/main/kernels) is the only copy of each shader. The package vendors that source and returns its text on import. It does not import MLX or PyTorch. A kernel's launch module compiles the source on first launch, through MLX or PyTorch MPS.

```python
from metal_kernels.flash_attention import flash_attention_hd32

out = flash_attention_hd32(q, k, v, bias=None, scale=None, backend="mlx")  # or backend="mps"
```

## SAM 3 on MPS

`metal_kernels.integrations.sam3` patches the pinned SAM 3 checkout (`31cb7e624e4fe95301b1be1d4e809111593d2e45`) at import time so the image model and the tracker can run without CUDA. Call it before importing SAM 3:

```python
from metal_kernels.integrations.sam3 import build_image_model, install

install()  # cuda if present, else mps, else cpu
model = build_image_model(load_from_HF=False)
```

The ViT MLP uses linear plus GELU on CPU and MPS. On CUDA the original fused op is left in place. MPS itself is not executed by the tests in this repository. The command that times the built-in kernels on a Mac is in [`bench/sam3_shapes/README.md`](https://github.com/jmamodeo/metal-kernels/blob/main/bench/sam3_shapes/README.md).
