// SPDX-License-Identifier: Apache-2.0
// conv3x3
//
// 3x3 convolution, stride 1, padding 1, dilation 1, groups 1, with bias.
// Input is NCHW. Weight is OIHW, [Cout, Cin, 3, 3]. Bias is [Cout].
//
//   out[n, oc, y, x] = bias[oc]
//     + sum over ic, ky, kx of
//         weight[oc, ic, ky, kx] * padded[n, ic, y + ky, x + kx]
//
// SAM 3 at facebookresearch/sam3 31cb7e62 uses this for the FPN neck
// (Sam3DualViTDetNeck.conv_3x3) and the mask-head pixel decoder
// (PixelDecoder, Conv2d(hidden_dim, hidden_dim, 3, 1, 1)). Both are
// 256 channels. The traced forward ran the tensors in bfloat16.
// The NumPy reference for this step computes float32.
//
// No kernel function yet. See reference/conv3x3.py.
