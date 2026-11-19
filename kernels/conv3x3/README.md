# conv3x3

3×3 convolution used by the SAM 3 FPN neck and the mask-head pixel decoder. Step 1 is the NumPy reference. The shader in this directory is a comment header only.

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

- Grid size: not started
- Threadgroup size: not started
- Threadgroup memory: not started
- Layout the kernel assumes (contiguous dimensions, vector width): not started
- Edge behavior (bounds checks, tail elements): not started

## Numerical check

- Reference: `reference/conv3x3.py` (`conv3x3_naive` and `conv3x3_im2col`)
- Tolerance, per dtype: float32, `rtol=1e-4`, `atol=1e-4` against the other NumPy implementation and against `torch.nn.functional.conv2d` on CPU when PyTorch is installed
- Shapes covered by the test: `H=W=1`, `5`, and `7`, including `Cin != Cout`; the im2col path also runs at `[1, 256, 288, 288]`

## Hosts

- Python (MLX): not started
- Python (PyTorch MPS): not started
- Swift: not started
- Benchmark against the built-in op: not started
