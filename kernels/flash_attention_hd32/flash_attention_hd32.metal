// SPDX-License-Identifier: Apache-2.0
// FlashAttention for head dim 32, version 0.
//
// softmax(Q Kᵀ * scale) V with an online softmax, so the score matrix is
// never written to memory. One simdgroup owns one query row and each of its
// 32 lanes owns one of the 32 head-dim columns. Q, K, V and O are bf16 in
// memory; all arithmetic is fp32 and O is rounded to bf16 once at the end.
//
// This file is the only copy of the shader. Both hosts compile this text:
// - PyTorch MPS (torch.mps.compile_shader) calls the `flash_attention_hd32`
//   kernel below.
// - MLX (mx.fast.metal_kernel) passes this file as `header` and calls
//   `flash_attention_hd32_row` from the generated kernel body.
//
// See kernels/flash_attention_hd32/README.md for shapes and launch geometry.

#include <metal_stdlib>
using namespace metal;

constant constexpr uint FA_HD32_HEAD_DIM = 32;

// bf16 -> fp32 is a bit placement: the bf16 bits are the high half.
inline float fa_hd32_load_bf16(const device ushort* data, uint index) {
    return as_type<float>(uint(data[index]) << 16);
}

// fp32 -> bf16, round to nearest even. Uses integer ops only so it holds
// under fast math. NaN stays NaN.
inline ushort fa_hd32_to_bf16(float value) {
    uint bits = as_type<uint>(value);
    if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0u) {
        return ushort((bits >> 16) | 0x0040u);
    }
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return ushort(bits >> 16);
}

// Computes one output element: row `thread_id / 32`, column `thread_id % 32`.
//
// q, out: (rows, 32) with rows = batch * heads * query_length
// k, v:   (batch * heads, key_length, 32)
//
// The launch must use a 1-D grid of rows * 32 threads and a threadgroup
// size that is a multiple of 32, so every simdgroup maps to exactly one row.
// key_length must be at least 1.
inline void flash_attention_hd32_row(
    const device ushort* q,
    const device ushort* k,
    const device ushort* v,
    device ushort* out,
    uint rows,
    uint query_length,
    uint key_length,
    float scale,
    uint thread_id) {
    const uint row = thread_id / FA_HD32_HEAD_DIM;
    const uint lane = thread_id % FA_HD32_HEAD_DIM;
    // All 32 lanes of a simdgroup share `row`, so they exit together and
    // simd_sum below always sees a full simdgroup.
    if (row >= rows) {
        return;
    }
    const uint batch_head = row / query_length;
    const device ushort* k_head = k + batch_head * key_length * FA_HD32_HEAD_DIM;
    const device ushort* v_head = v + batch_head * key_length * FA_HD32_HEAD_DIM;

    // Fold the softmax scale into log2(e) so the exponentials are exp2.
    const float q_scaled =
        fa_hd32_load_bf16(q, row * FA_HD32_HEAD_DIM + lane) * (scale * M_LOG2E_F);

    // Seed the running max, sum and output with key 0. This avoids -inf,
    // which fast math does not promise to handle.
    float running_max = simd_sum(q_scaled * fa_hd32_load_bf16(k_head, lane));
    float running_sum = 1.0f;
    float acc = fa_hd32_load_bf16(v_head, lane);

    for (uint j = 1; j < key_length; ++j) {
        const uint offset = j * FA_HD32_HEAD_DIM + lane;
        // simd_sum makes `score` uniform across the simdgroup, so the
        // branch below never diverges.
        const float score = simd_sum(q_scaled * fa_hd32_load_bf16(k_head, offset));
        const float value = fa_hd32_load_bf16(v_head, offset);
        if (score > running_max) {
            // Rescale the running sum and output only when the max rises.
            const float correction = exp2(running_max - score);
            running_max = score;
            running_sum = running_sum * correction + 1.0f;
            acc = acc * correction + value;
        } else {
            const float p = exp2(score - running_max);
            running_sum += p;
            acc += p * value;
        }
    }

    out[row * FA_HD32_HEAD_DIM + lane] = fa_hd32_to_bf16(acc / running_sum);
}

// PyTorch MPS entry point. dims = (rows, query_length, key_length) as int32,
// scale is a one-element float32 buffer.
kernel void flash_attention_hd32(
    const device ushort* q [[buffer(0)]],
    const device ushort* k [[buffer(1)]],
    const device ushort* v [[buffer(2)]],
    device ushort* out [[buffer(3)]],
    const device int* dims [[buffer(4)]],
    const device float* scale [[buffer(5)]],
    uint thread_id [[thread_position_in_grid]]) {
    flash_attention_hd32_row(
        q, k, v, out, uint(dims[0]), uint(dims[1]), uint(dims[2]), scale[0], thread_id);
}
