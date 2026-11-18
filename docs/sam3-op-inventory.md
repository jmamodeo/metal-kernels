# SAM 3 op inventory

One randomly initialized SAM 3 forward pass, traced on CPU, ranked by estimated FLOPs. This is the list that decides which Metal kernels to write. Trained weights were not used.

The numbers below come from `tools/sam3_inventory/output/`. Reproduce them with `tools/sam3_inventory/README.md`. The checkout is [facebookresearch/sam3](https://github.com/facebookresearch/sam3) `31cb7e624e4fe95301b1be1d4e809111593d2e45`, imported as the installed `sam3` 0.1.0 package. The run used PyTorch `2.15.0+cpu` and torchvision `0.29.2+cpu`. The image model has 840,509,750 parameters. Nothing in this document was executed on Metal, MPS, or MLX.

## What was traced

Image size 1008, the builder default. Patch size 14 gives a 72×72 grid, 5184 tokens, width 1024, 32 ViT blocks, 16 heads, MLP hidden 4736. The text encoder is 24 layers at width 1024. The detector runs at width 256. The prompt was the string `a red cube`.

Parameters stay float32. Each stage runs under `torch.autocast(device_type="cpu", dtype=torch.bfloat16)`, the CPU counterpart of the official CUDA bfloat16 autocast. Complex RoPE buffers stay complex. The ViT MLP also calls `aten._addmm_activation` in bfloat16 directly.

| Stage | What ran | Rank FLOPs | Share |
| --- | --- | ---: | ---: |
| `image_backbone` | Vision trunk and FPN neck | 5.614e12 | 88.0% |
| `image_text` | Text encoder | 1.948e10 | 0.3% |
| `image_detector` | Geometry encoder, detector, mask head | 4.243e11 | 6.7% |
| `tracker_init` | Frame 0, one positive point, memory encoder on | 2.133e10 | 0.3% |
| `tracker_memory` | Frame 1, memory-conditioned, after the patch below | 3.007e11 | 4.7% |

The tracker is `build_tracker(apply_temporal_disambiguation=False, with_backbone=False)`. It consumes the three FPN maps from the image forward, projected the way `Sam3TrackerBase.forward_image` does (`conv_s0` 256→32, `conv_s1` 256→64). Those maps were `[1, 256, 288, 288]`, `[1, 256, 144, 144]`, `[1, 256, 72, 72]`. The ViT is not run again. A video frame that recomputes the backbone adds another copy of the vision encoder, which is 84.6% of this trace. The video builder also clones the FPN (`add_sam2_neck=True`); this image model does not.

## How the FLOPs are counted

`rank_flops` is `torch.utils.flop_counter.FlopCounterMode` plus a shape formula for `aten._addmm_activation`. A multiply-add counts as 2. The counter total is 4.770e12. The fused MLP matmul, which the counter does not register, adds 1.609e12. The rank total is **6.379e12**. The same shape formulas, applied to every matmul, attention, and convolution in the trace, add up to the same 6.379e12.

`extra_flops` (9.94e9, about 0.16% of the rank total) estimates ops the counter ignores: 5 per normalization element, 8 per GELU or SiLU element, 5 per softmax element, 1 per pointwise element, and the GELU fused inside `_addmm_activation`. It is not part of the rank.

`bytes` sums distinct input and output storages at the aten call. It is not measured bandwidth. Views are included, so the view family's byte share is not traffic. `aten._to_copy` is mostly the autocast cast from float32 weights to bfloat16. The CPU attention kernel does not expose its score matrix, so attention bytes are short by that matrix. For the global attention that matrix is `16 × 5184 × 5184` elements per call, about 0.86 GiB in bfloat16, and it is not in the byte column.

Dispatcher-inclusive wall time on this 4-thread CPU run was 3.13s for the backbone, 0.17s for text, 0.92s for the detector, 0.28s for tracker init, and 0.37s for the memory step. That includes the Python dispatcher. It is not a benchmark.

## Ranked families

| Family | Calls | Rank FLOPs | Share | Extra FLOPs | Byte share | What dominates it |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| matmul / linear | 660 | 4.837e12 | 75.8% | 6.29e9 | 13.9% | ViT MLP and QKV projections |
| attention | 103 | 1.176e12 | 18.4% | 0 | 2.7% | SDPA, score matrix not counted in bytes |
| conv | 41 | 3.661e11 | 5.7% | 0 | 1.3% | 3×3 at 288² and a 2×2 transposed conv |
| norm | 224 | 0 | 0 | 2.14e9 | 4.8% | layer norm and group norm |
| elementwise | 1044 | 0 | 0 | 8.94e8 | 13.6% | residual add, mul, RoPE |
| activation | 154 | 0 | 0 | 6.10e8 | 1.5% | GELU outside the fused MLP, ReLU, SiLU |
| softmax | 12 | 0 | 0 | 1.13e7 | 0.0% | the few softmaxes that are not inside SDPA |
| view | 3763 | 0 | 0 | 0 | 32.6% | reshape and window partition, not real traffic |
| memory | 2135 | 0 | 0 | 0 | 22.9% | mostly `aten._to_copy` autocast casts |
| other | 1149 | 0 | 0 | 0 | 5.6% | transpose, compare, `mean`, `arange` |

Matmul, attention, and convolution are 99.9% of rank FLOPs.

## Top shapes

Grouped across the 32 blocks. Share is of the 6.379e12 rank total.

| Rank FLOPs | Share | Calls | Where | Op and shape |
| ---: | ---: | ---: | --- | --- |
| 1.609e12 | 25.2% | 32 | ViT MLP fc1 | `_addmm_activation`, `[5184, 1024] × [1024, 4736]`, `use_gelu=True` |
| 1.609e12 | 25.2% | 32 | ViT MLP fc2 | `addmm`, `[5184, 4736] × [4736, 1024]` |
| 1.044e12 | 16.4% | 32 | ViT QKV | `addmm`, `[5184, 1024] × [1024, 3072]` |
| 4.403e11 | 6.9% | 4 | ViT global attention, blocks 7, 15, 23, 31 | SDPA `[1, 16, 5184, 64]` |
| 3.479e11 | 5.5% | 32 | ViT attention projection | `addmm`, `[5184, 1024] × [1024, 1024]` |
| 3.425e11 | 5.4% | 28 | ViT window attention | SDPA `[9, 16, 576, 64]` (9 windows of 24×24) |
| 1.651e11 | 2.6% | 6 | Detector encoder | SDPA `[1, 8, 5184, 32]` |
| 1.102e11 | 1.7% | 4 | Tracker memory attention | SDPA Q `[1, 1, 5184, 256]`, K/V `[1, 1, 5188, 256]` |
| 1.101e11 | 1.7% | 4 | Tracker memory attention | SDPA `[1, 1, 5184, 256]` square |
| 9.784e10 | 1.5% | 1 | FPN neck | conv `[1, 256, 288, 288]`, weight `[256, 256, 3, 3]` |
| 9.784e10 | 1.5% | 1 | Mask head pixel decoder | same 3×3 |
| 4.349e10 | 0.7% | 2 | FPN neck | transposed conv `[1, 1024, 72, 72]`, weight `[1024, 512, 2, 2]` |

The two MLP linears plus QKV plus the attention projection are 72.3% of the whole forward pass.

## Where the FLOPs sit

| Region | Rank FLOPs | Share |
| --- | ---: | ---: |
| Vision encoder | 5.399e12 | 84.6% |
| Tracker memory attention (frame 1 only, no second ViT) | 2.814e11 | 4.4% |
| Detector encoder | 2.559e11 | 4.0% |
| Vision neck | 2.149e11 | 3.4% |
| Mask head | 1.433e11 | 2.2% |
| Detector decoder | 2.068e10 | 0.3% |
| Text encoder | 1.948e10 | 0.3% |
| Tracker memory encoder, frame 0 and frame 1 together | 2.938e10 | 0.5% |
| Tracker mask decoder, both frames | 1.123e10 | 0.2% |
| Geometry encoder | 4.102e9 | 0.1% |

## Apple silicon coverage

This is a source review, not a measurement. The MPS notes are the PyTorch tag `v2.15.0` tree, the same version as the trace. The MLX notes are the 0.32.4 docs. Neither stack was executed. Every row that says an op exists still needs a Mac run at the shapes in the table above before it counts as covered for this model.

PyTorch's MPS fallback list at that tag (`aten/src/ATen/mps/MPSFallback.mm`) registers an unconditional CPU fallback only for `embedding_renorm_` and a `_slow_conv2d_forward` stub. Any other op without an MPS kernel errors unless `PYTORCH_ENABLE_MPS_FALLBACK=1`.

| Family | PyTorch MPS at v2.15.0 | MLX 0.32.4 docs | Confirm on a Mac |
| --- | --- | --- | --- |
| matmul / linear | `Linear.mm` routes through `addmm` / `mm` | `mlx.nn.Linear`, `mx.matmul` | The four GEMM shapes above, in bfloat16, batch 5184 |
| attention | `Attention.mm`: `scaled_dot_product_attention` for floating dtypes, dropout must be 0, with vector, prefill, and general paths | `mlx.nn.MultiHeadAttention`, `mx.fast.scaled_dot_product_attention` | Seq 5184 with 16 heads of dim 64; window seq 576; tracker head dim 256 with K length 5184 and 5188 |
| fused MLP | No `_addmm_activation` kernel found in the files reviewed. Linear and GELU exist separately | Linear and `mlx.nn.GELU` exist separately | Whether `aten._addmm_activation` is implemented or errors |
| conv, including transposed | `Convolution.mm`: `_mps_convolution` and `_mps_convolution_transpose` | `mlx.nn.Conv2d`, `mlx.nn.ConvTranspose2d` (NHWC) | 3×3 at `[1, 256, 288, 288]` and the 2×2 transposed conv |
| layer norm | `Normalization.mm`: `layer_norm_mps` | `mlx.nn.LayerNorm`, `mx.fast.layer_norm` | Width 1024 and 256 |
| group norm | `GroupNorm.mm`: `group_norm` Metal kernels | `mlx.nn.GroupNorm` (`pytorch_compatible=False` by default) | The mask-head groups |
| GELU | `Activation.mm`: `gelu_out_mps` | `mlx.nn.GELU` | The unfused GELU calls |
| softmax | `SoftMax.mm`: `softmax_mps_out` | inside SDPA; `mlx.nn.Softmax` | Only the 12 standalone softmaxes |
| upsample | `UpSample.mm`: nearest, bilinear, and bilinear antialias | `mlx.nn.Upsample` (nearest, linear, cubic) | The mask upsample `[1, 1, 1008, 1008]` |
| max pool | `Pooling.mm`: `max_pool2d_with_indices` | `mlx.nn.MaxPool2d` | The one neck pool on `[1, 1024, 72, 72]` |
| grid sample | `GridSampler.mm`: `grid_sampler_2d` | No documented op. mlx-explore/mlx#1213 tracks a custom kernel, not an `mlx` API | Do not use this trace to size it. See below |

The one `aten.grid_sampler_2d` in the trace has grid shape `[1, 0, 1, 2]`. Random-init scores kept no boxes after the processor threshold, so the geometric sample is empty. That call says the op is reachable. It does not say what a real box costs.

## Proposed kernel order

The rank says where the time would go. The write order also asks whether MPS and MLX already have the op. Per the roadmap, a custom kernel has to beat the built-in on the shape that shows up here.

1. **Measure the vision-encoder attention and the four GEMMs on a Mac before writing anything.** Those two families are 94.3% of rank FLOPs, and both stacks already have an implementation in the sources above. The shapes to time are SDPA `[1, 16, 5184, 64]` (4 calls, 6.9%), SDPA `[9, 16, 576, 64]` (28 calls, 5.4%), and the MLP pair `[5184, 1024] × [1024, 4736]` and its transpose (32 calls each, 25.2% each). If the 5184-length SDPA stays on the GPU and is close to a hand-written kernel, it is not a kernel task.

2. **First kernel to write, only if that measurement falls back or is far from the built-in: global attention at `[1, 16, 5184, 64]`.** It is the largest attention shape, and the score matrix is the memory risk the byte column hides. Window attention at seq 576 is the follow-up (5.4%), then the detector encoder SDPA at `[1, 8, 5184, 32]` (2.6%).

3. **Tracker memory attention is the next candidate, and the one most likely to miss a fast path.** Q is `[1, 1, 5184, 256]` and one of the two K/V shapes is `[1, 1, 5188, 256]`: one head, head dim 256, and the lengths differ. Four calls, 1.7% each, 4.4% of a trace that does not rerun the ViT. Confirm the unequal-length path on a Mac. A video frame that also reruns the backbone makes this a smaller share of the frame, not a smaller kernel.

4. **Treat `aten._addmm_activation` as a port, not a kernel.** The ViT MLP calls it directly, 32 times, 25.2% of rank FLOPs, in bfloat16. MPS linear and GELU exist as separate ops, and so do MLX linear and GELU. Splitting the call uses those. A fused GEMM+GELU kernel waits on a Mac profile that shows the split is the bottleneck.

5. **Benchmark the neck and mask-head convolutions, and write one only if the built-in loses.** The 3×3 at 288² is 1.5% in the neck and 1.5% in the mask head. The transposed 2×2 is 0.7%. MPS and MLX both have convolution and transposed convolution in the sources above.

6. **Leave layer norm, GELU, softmax, upsample, group norm, and max pool off the write list.** Together their extra-FLOP estimate is about 0.16% of rank, and each has a kernel file at the v2.15.0 tag and an MLX API. RoPE in this forward is a handful of real multiplies on a complex view, not a missing op.

7. **`grid_sampler_2d` is an MLX gap, not a compute leader.** MPS has the op. MLX does not document one. Re-trace with a prompt that keeps a box before sizing a kernel. The empty grid in this run is not that measurement.

## What blocks a Metal run

These are in the SAM 3 source at the pinned commit, or they were raised by this trace. The scan is `output/cuda_scan.json` (15 expected snippets, all present).

- **CUDA device strings at init.** `PositionEmbeddingSine` precomputes with `torch.zeros(..., device="cuda")`, and the decoder caches box-RPB coordinates with `torch.arange(..., device="cuda")`. Building the image model and the tracker on this CPU PyTorch required redirecting those factories. The run recorded 16 `zeros` redirects and 2 `arange` redirects. The cache is then reused, so the forward itself does not allocate them again.
- **Unconditional bfloat16 fused MLP.** `sam3/perflib/fused.py` casts the ViT MLP fc1 operands to bfloat16 and calls `aten._addmm_activation`. With grad disabled this happens even outside autocast. A float32-only forward dies on the next linear.
- **`Tensor.pin_memory()`.** The geometry encoder and the tracker relative position code call it. On this CPU build it raises `Cannot access accelerator device when none is available`. The trace patched it to identity and counted 2 calls.
- **`Tensor.cuda()` in the tracker.** Frame 1, unpatched, raises `AssertionError: Torch not compiled with CUDA enabled` at `sam3_tracker_base.py` line 656, `prev["maskmem_features"].cuda(non_blocking=True)`. A second `.cuda()` sits on the positional encoding a few lines later. The memory-conditioned stage in this inventory was re-run with `.cuda()` as identity so the attention could be counted. That patch is not how the model runs.
- **CUDA autocast entered at construction.** `Sam3TrackerPredictor.__init__` and the video inference helpers enter `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`. On this CPU build the import warns `CUDA is not available ... Disabling autocast.` The decoder FFN is also wrapped in `torch.amp.autocast(device_type="cuda", enabled=False)`.
- **Triton imported even when unused.** `sam3/model/edt.py` imports `triton` at import time, and `sam3_tracker_utils` imports that module. With `fill_hole_area=0` the kernel is not called. Triton NMS (`sam3/perflib/triton/nms.py`), connected components (`connected_components.py`, `cc_torch`), and the training sigmoid focal loss are also Triton. None of them ran in this forward.
- **FlashAttention 3 and FP8 are optional and default off for this path.** `sam3/perflib/fa3.py` imports `flash_attn_interface` and casts to `float8_e4m3fn`. The image builder and `build_tracker` leave `use_fa3=False`, so this trace used `F.scaled_dot_product_attention`, which on CPU is `aten._scaled_dot_product_flash_attention_for_cpu`. `build_sam3_multiplex_video_predictor` defaults `use_fa3=True`. That predictor was not traced.
- **`torch.compile`.** `sam3/perflib/compile.py` compiles modules and looks for CUDA graphs. The builders were called with `compile=False`.

## Files

| Path | Contents |
| --- | --- |
| `tools/sam3_inventory/inventory.py` | Build, trace, rank, and the static CUDA scan |
| `tools/sam3_inventory/output/summary.json` | Metadata, family totals, regions, grouped shapes |
| `tools/sam3_inventory/output/families.csv` | The family table |
| `tools/sam3_inventory/output/shapes.csv` | Grouped shapes |
| `tools/sam3_inventory/output/regions.csv` | Region rollup |
| `tools/sam3_inventory/output/calls.json` | Every aggregated aten call |
| `tools/sam3_inventory/output/flops_by_stage.json` | `FlopCounterMode` per module |
| `tools/sam3_inventory/output/cuda_scan.json` | CUDA-only snippets and the frame-1 exception |
