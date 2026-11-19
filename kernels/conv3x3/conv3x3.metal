// SPDX-License-Identifier: Apache-2.0
// conv3x3
//
// 3x3 convolution, stride 1, padding 1, dilation 1, groups 1, with bias.
// Input is NCHW. Weight is OIHW, [Cout, Cin, 3, 3]. Bias is [Cout].
// Output is NCHW and has the same height and width as the input.
//
//   out[n, oc, y, x] = bias[oc]
//     + sum over ic, ky, kx of
//         weight[oc, ic, ky, kx] * padded[n, ic, y + ky, x + kx]
//
// This is the direct form of that sum. It does not build the im2col matrix
// (729 MiB at the SAM 3 shape). A threadgroup owns an 8x8 patch of output
// pixels and, one input channel at a time, loads the 10x10 window those
// pixels need into threadgroup memory. The 10 counts the 8 outputs plus the
// one-pixel halo on every side. Each thread then dots its own 3x3.
//
// PyTorch MPS and the later Swift host call `conv3x3`. MLX generates its
// own kernel signature and calls `conv3x3_thread` with the same buffers.
// The math is only here.

#include <metal_stdlib>
using namespace metal;

// Must match TILE and OC_PER_THREAD in metal_kernels/conv3x3.py.
enum {
    CONV3X3_TILE = 8,
    CONV3X3_OC_PER_THREAD = 4,
    CONV3X3_TILE_PAD = CONV3X3_TILE + 2
};

// One thread of the launch. Every thread in the group must enter this
// function: the barriers below are only legal if the whole group hits them.
//
// thread_in_group is this thread's (x, y) inside the 8x8 group.
// group_in_grid is which 8x8 patch, and which (batch, output-channel block).
inline void conv3x3_thread(
    device const float* x,
    device const float* weight,
    device const float* bias,
    device float* out,
    threadgroup float* tile,
    int n,
    int cin,
    int height,
    int width,
    int cout,
    uint3 thread_in_group,
    uint3 group_in_grid) {
    // Output pixel this thread writes. x grows across a row, y down a column,
    // the same order as the NCHW index below.
    int x0 = int(group_in_grid.x) * CONV3X3_TILE;
    int y0 = int(group_in_grid.y) * CONV3X3_TILE;
    int tx = int(thread_in_group.x);
    int ty = int(thread_in_group.y);
    int pixel_x = x0 + tx;
    int pixel_y = y0 + ty;

    // grid z walks batch first, then blocks of 4 output channels.
    int oc_blocks = (cout + CONV3X3_OC_PER_THREAD - 1) / CONV3X3_OC_PER_THREAD;
    int z = int(group_in_grid.z);
    int n_idx = z / oc_blocks;
    int oc_base = (z - n_idx * oc_blocks) * CONV3X3_OC_PER_THREAD;

    // Threads past the image still load the halo and hit the barriers.
    // They just do not write.
    bool inside = n_idx < n && pixel_x < width && pixel_y < height;

    // Linear id 0..63. Used only to share the load of the 10x10 tile.
    uint tid = uint(ty * CONV3X3_TILE + tx);
    uint threads_in_group = uint(CONV3X3_TILE * CONV3X3_TILE);
    int tile_elems = CONV3X3_TILE_PAD * CONV3X3_TILE_PAD;

    float acc[CONV3X3_OC_PER_THREAD];
    for (int k = 0; k < CONV3X3_OC_PER_THREAD; ++k) {
        int oc = oc_base + k;
        acc[k] = (oc < cout) ? bias[oc] : 0.0f;
    }

    for (int ic = 0; ic < cin; ++ic) {
        // tile[ly, lx] holds input channel ic at image coordinate
        // (y0 - 1 + ly, x0 - 1 + lx). The -1 is the padding halo.
        // Coordinates outside the image are zeros, which is padding mode.
        for (int i = int(tid); i < tile_elems; i += int(threads_in_group)) {
            int ly = i / CONV3X3_TILE_PAD;
            int lx = i - ly * CONV3X3_TILE_PAD;
            int iy = y0 - 1 + ly;
            int ix = x0 - 1 + lx;
            float value = 0.0f;
            if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
                // NCHW: ((n * Cin + ic) * H + y) * W + x
                int x_index = ((n_idx * cin + ic) * height + iy) * width + ix;
                value = x[x_index];
            }
            tile[i] = value;
        }
        // Every thread must see the full window before anyone reads it.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (inside) {
            for (int k = 0; k < CONV3X3_OC_PER_THREAD; ++k) {
                int oc = oc_base + k;
                if (oc >= cout) {
                    continue;
                }
                // This thread's output sits one pixel in from the tile origin
                // because of the halo, so its 3x3 starts at (ty, tx).
                float sum = 0.0f;
                for (int ky = 0; ky < 3; ++ky) {
                    for (int kx = 0; kx < 3; ++kx) {
                        int tile_index = (ty + ky) * CONV3X3_TILE_PAD + (tx + kx);
                        // OIHW: ((oc * Cin + ic) * 3 + ky) * 3 + kx
                        int w_index = ((oc * cin + ic) * 3 + ky) * 3 + kx;
                        sum += tile[tile_index] * weight[w_index];
                    }
                }
                acc[k] += sum;
            }
        }
        // The next channel overwrites the tile. Wait until every thread
        // has finished reading this one.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (inside) {
        for (int k = 0; k < CONV3X3_OC_PER_THREAD; ++k) {
            int oc = oc_base + k;
            if (oc >= cout) {
                continue;
            }
            int out_index = ((n_idx * cout + oc) * height + pixel_y) * width + pixel_x;
            out[out_index] = acc[k];
        }
    }
}

// PyTorch MPS (`torch.mps.compile_shader`) and the Swift host.
// dims is five int32s: N, Cin, H, W, Cout.
kernel void conv3x3(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant int* dims [[buffer(4)]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
    threadgroup float tile[CONV3X3_TILE_PAD * CONV3X3_TILE_PAD];
    conv3x3_thread(
        x,
        weight,
        bias,
        out,
        tile,
        dims[0],
        dims[1],
        dims[2],
        dims[3],
        dims[4],
        thread_position_in_threadgroup,
        threadgroup_position_in_grid);
}
