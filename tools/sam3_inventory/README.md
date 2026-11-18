# SAM 3 op inventory

Traces one randomly initialized SAM 3 forward pass and writes the ranked aten ops under `output/`. Trained weights are not downloaded.

The checkout this inventory was generated from is `facebookresearch/sam3` commit `31cb7e624e4fe95301b1be1d4e809111593d2e45`. The script records the checkout it actually imported and warns if that SHA differs.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
git clone https://github.com/facebookresearch/sam3
git -C sam3 checkout 31cb7e624e4fe95301b1be1d4e809111593d2e45
pip install -e ./sam3
pip install -r tools/sam3_inventory/requirements.txt
```

`sam3/pyproject.toml` does not declare `einops`, `triton`, `pycocotools`, or `psutil`, but importing the package pulls them in. `triton` is imported at module level by `sam3/model/edt.py` even when the distance transform is not called. The requirements file pins the versions used for this inventory.

## Run

```bash
python tools/sam3_inventory/inventory.py --sam3-root /path/to/sam3
```

`--self-check` runs a tiny module through the same tracer and checks that matmul, attention, convolution, layer norm, and GELU show up, and that the shape formulas agree with `FlopCounterMode`.

The image path is `Sam3Processor`'s body at 1008px: vision backbone, text encoder on the prompt `a red cube`, then `forward_grounding` (geometry encoder, detector, segmentation head, processor postprocess). The tracker path builds `build_tracker(..., with_backbone=False)` so the ViT is not run a second time, feeds it the FPN maps from the image forward, and runs one prompted frame plus one memory-conditioned frame.

## What the numbers are

`rank_flops` is `FlopCounterMode` plus the matmul formula for `aten._addmm_activation`, which that counter does not register. A multiply-add counts as 2 flops. `extra_flops` is a separate estimate for ops with no matmul formula: 5 per norm element, 8 per GELU or SiLU element, 5 per softmax element, 1 per pointwise element, and the GELU inside the fused MLP (8 per output element). The fused matmul itself is in `rank_flops`, not in `extra_flops`.

`bytes` is the sum of distinct input and output storages at the aten call. It is not measured bandwidth. The CPU attention kernel does not expose its score matrix as a separate tensor, so attention traffic is under-counted.

Parameters stay float32. Each stage runs under `torch.autocast(device_type="cpu", dtype=torch.bfloat16)`, which is the CPU counterpart of the official CUDA bfloat16 autocast. `Mlp.forward` also forces its first linear and activation through `aten._addmm_activation` in bfloat16. Complex RoPE buffers are left complex. Nothing here was executed on Metal.
