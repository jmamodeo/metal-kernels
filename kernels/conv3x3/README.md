# conv3x3

3×3 convolution used by the SAM 3 FPN neck and the mask-head pixel decoder. Step 1, the NumPy reference, is done. Step 2, the Metal kernel called from Python, is in progress: the shader and both hosts are written, and the Mac timings are not filled in yet.

SAM 3 commit `31cb7e624e4fe95301b1be1d4e809111593d2e45`:

- Neck: `Sam3DualViTDetNeck` in `sam3/model/necks.py` builds `conv_3x3` as `nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=True)`. `model_builder._create_vit_neck` sets `d_model=256`. Stride, dilation, and groups are omitted, so they are the PyTorch defaults: stride 1, dilation 1, groups 1. Padding mode is the default, zeros.
- Mask head: `PixelDecoder` in `sam3/model/maskformer_segmentation.py` builds `nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1)`. That positional form is kernel 3, stride 1, padding 1. Dilation, groups, and bias stay at the defaults (1, 1, and True). `model_builder._create_segmentation_head` sets `hidden_dim=256`.

The traced image forward ran both calls in bfloat16, on input `[1, 256, 288, 288]` with weight `[256, 256, 3, 3]` and bias `[256]`. This reference accepts float32 and computes in float32. The 288×288 map is the finest FPN level (two stride-2 transposed convolutions from the 72×72 trunk feature, then this 3×3).

## What it computes

Each output pixel is one dot product of the 3×3 neighborhood across every input channel, plus the bias for that output channel:

```
out[n, oc, y, x] = bias[oc]
  + sum_{ic, ky, kx} weight[oc, ic, ky, kx] * padded[n, ic, y+ky, x+kx]
```

Padding is 1 on each spatial edge, so the output height and width match the input. At 256 input channels the dot product has `256 * 9 = 2304` inputs. Multiply-adds are `2 * Cout * Cin * 9 * H * W`. For the SAM 3 shape that is 97844723712, the inventory's 9.784e10. `reference/conv3x3.py` has two implementations of this formula: an explicit loop, and the same math lowered to one matmul through im2col.

## Inputs

| Name | Shape | Dtype | Notes |
| --- | --- | --- | --- |
| `x` | `[N, Cin, H, W]` | float32 | NCHW. The SAM 3 call is `[1, 256, 288, 288]`. |
| `weight` | `[Cout, Cin, 3, 3]` | float32 | OIHW, PyTorch `Conv2d` layout. The SAM 3 call is `[256, 256, 3, 3]`. |
| `bias` | `[Cout]` | float32 | One value per output channel. Both SAM 3 call sites pass bias. |

## Outputs

| Name | Shape | Dtype | Notes |
| --- | --- | --- | --- |
| `out` | `[N, Cout, H, W]` | float32 | Same spatial size as the input. Stride 1, padding 1. |

## Threadgroup assumptions

The launch is `dispatchThreads`. `threads` is the total number of threads, not the number of groups.

- Grid size: `(ceil(W / 8) * 8, ceil(H / 8) * 8, N * ceil(Cout / 4))`
- Threadgroup size: `(8, 8, 1)`. One thread per output pixel in an 8×8 tile. Each thread accumulates 4 output channels.
- Threadgroup memory: one `float` tile of `(8 + 2) * (8 + 2) = 100` values, 400 bytes. That is the input window for the current input channel, including the 1-pixel halo. It is overwritten for every input channel. Nothing like the 729 MiB im2col matrix is allocated.
- Layout the kernel assumes: C-contiguous NCHW input, OIHW weight, a contiguous bias vector, all float32. No vector loads.
- Edge behavior: the grid is rounded up to a whole 8×8 tile. Threads past `H` or `W` still load and hit both barriers, and they do not write. Samples outside the image are zeros.

At `[1, 256, 288, 288]` the grid is `(288, 288, 64)` because 288 is divisible by 8 and 256 output channels are 64 blocks of 4.

## Numerical check

- Reference: `reference/conv3x3.py` (`conv3x3_naive` and `conv3x3_im2col`)
- Tolerance, per dtype: float32, `rtol=1e-4`, `atol=1e-4` for the shapes in `tests/test_conv3x3_metal.py`. The NumPy tests also use that tolerance against CPU `conv2d`, including the full `[1, 256, 288, 288]` im2col path.
- Shapes covered by the Metal test: `H=W=1`, `5`, `7`, and `8`; `9×10`; `Cin != Cout`; batch 2. The full SAM 3 shape is what `bench/conv3x3.py` times, not what the pytest launches.
- Measured on John's M3 Max: not yet. Run `python bench/conv3x3.py` and record the milliseconds here.

## Run on a Mac

From the repository root, with the `mac` extra installed:

```bash
pytest tests/test_conv3x3_metal.py
python bench/conv3x3.py
```

`pytest` without a Mac still runs the launch-geometry checks in that file. The tests that launch the shader are marked `metal` and skip.

## Hosts

- Python (MLX): `metal_kernels.conv3x3.conv3x3` via `mx.fast.metal_kernel`. The `.metal` file is the header; the body calls `conv3x3_thread`.
- Python (PyTorch MPS): the same function, via `torch.mps.compile_shader` on `kernel void conv3x3`.
- Swift: not started
- Benchmark against the built-in op: `bench/conv3x3.py` is written. Numbers are not recorded yet.
