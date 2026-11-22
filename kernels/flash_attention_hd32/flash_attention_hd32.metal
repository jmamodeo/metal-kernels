// SPDX-License-Identifier: Apache-2.0
// FlashAttention for head dim 32, version 0.
//
// softmax(Q Kᵀ * scale + bias) V with an online softmax, so the score matrix
// is never written to memory. One simdgroup owns one query row and each of
// its 32 lanes owns one of the 32 head-dim columns. Q, K, V, bias and O share
// one dtype, bf16 or fp32. All arithmetic is fp32; a bf16 O is rounded once
// at the end.
//
// This file is the only copy of the shader. Both hosts compile this text:
// - PyTorch MPS (torch.mps.compile_shader) calls `flash_attention_hd32_bfloat`
//   or `flash_attention_hd32_float` below.
// - MLX (mx.fast.metal_kernel) passes this file as `header` and calls
//   `flash_attention_hd32_row` from the generated kernel body.
//
// See kernels/flash_attention_hd32/README.md for shapes and launch geometry.

#include <metal_stdlib>
using namespace metal;

constant constexpr uint FA_HD32_HEAD_DIM = 32;
constant constexpr uint FA_HD32_NEG_INF_BITS = 0xFF800000u;
constant constexpr uint FA_HD32_NEG_FLT_MAX_BITS = 0xFF7FFFFFu;

// Layout of the int32 `dims` buffer.
enum FaHd32Dim : uint {
    FA_HD32_ROWS = 0,          // batch * heads * query_length
    FA_HD32_HEADS = 1,
    FA_HD32_QUERY_LENGTH = 2,
    FA_HD32_KEY_LENGTH = 3,
    FA_HD32_HAS_BIAS = 4,      // 0 or 1
    FA_HD32_BIAS_STRIDE_B = 5, // bias strides in elements; 0 broadcasts
    FA_HD32_BIAS_STRIDE_H = 6,
    FA_HD32_BIAS_STRIDE_Q = 7,
    FA_HD32_BIAS_STRIDE_K = 8,
    FA_HD32_DIM_COUNT = 9,
};

struct FaHd32Params {
    uint rows;
    uint heads;
    uint query_length;
    uint key_length;
    bool has_bias;
    uint bias_stride_b;
    uint bias_stride_h;
    uint bias_stride_q;
    uint bias_stride_k;
    float scale;
};

// bf16 is stored as ushort bits. Widening is a bit placement: the bf16 bits
// are the high half of the fp32 word.
inline float fa_hd32_load(const device ushort* data, uint index) {
    return as_type<float>(uint(data[index]) << 16);
}
inline float fa_hd32_load(const constant ushort* data, uint index) {
    return as_type<float>(uint(data[index]) << 16);
}
inline float fa_hd32_load(const device float* data, uint index) {
    return data[index];
}
inline float fa_hd32_load(const constant float* data, uint index) {
    return data[index];
}

// fp32 -> bf16, round to nearest even. Integer ops only, so it holds under
// fast math. NaN stays NaN.
inline void fa_hd32_store(device ushort* data, uint index, float value) {
    uint bits = as_type<uint>(value);
    if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0u) {
        data[index] = ushort((bits >> 16) | 0x0040u);
        return;
    }
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    data[index] = ushort(bits >> 16);
}
inline void fa_hd32_store(device float* data, uint index, float value) {
    data[index] = value;
}

// -inf test on the bits, so fast math cannot fold it away.
inline bool fa_hd32_is_neg_inf(float value) {
    return as_type<uint>(value) == FA_HD32_NEG_INF_BITS;
}

// Maps a -inf produced by overflow (for example a finfo.min bias times
// log2(e)) to -FLT_MAX, so finite inputs keep finite scores.
inline float fa_hd32_finite(float value) {
    return fa_hd32_is_neg_inf(value) ? as_type<float>(FA_HD32_NEG_FLT_MAX_BITS) : value;
}

// View an MLX buffer as the storage type above, keeping its address space.
// Any 2-byte element type (MLX's `bfloat16_t`) becomes ushort bits; float
// stays float. MLX passes inputs with fewer than 8 elements in the constant
// address space, which is why both address spaces are covered.
template <typename U>
inline const device ushort* fa_hd32_bits(const device U* data) {
    static_assert(sizeof(U) == 2, "expected a bf16 buffer");
    return reinterpret_cast<const device ushort*>(data);
}
template <typename U>
inline const constant ushort* fa_hd32_bits(const constant U* data) {
    static_assert(sizeof(U) == 2, "expected a bf16 buffer");
    return reinterpret_cast<const constant ushort*>(data);
}
template <typename U>
inline device ushort* fa_hd32_bits(device U* data) {
    static_assert(sizeof(U) == 2, "expected a bf16 buffer");
    return reinterpret_cast<device ushort*>(data);
}
inline const device float* fa_hd32_bits(const device float* data) {
    return data;
}
inline const constant float* fa_hd32_bits(const constant float* data) {
    return data;
}
inline device float* fa_hd32_bits(device float* data) {
    return data;
}

template <typename Dims>
inline FaHd32Params fa_hd32_params(Dims dims, float scale) {
    FaHd32Params params;
    params.rows = uint(dims[FA_HD32_ROWS]);
    params.heads = uint(dims[FA_HD32_HEADS]);
    params.query_length = uint(dims[FA_HD32_QUERY_LENGTH]);
    params.key_length = uint(dims[FA_HD32_KEY_LENGTH]);
    params.has_bias = dims[FA_HD32_HAS_BIAS] != 0;
    params.bias_stride_b = uint(dims[FA_HD32_BIAS_STRIDE_B]);
    params.bias_stride_h = uint(dims[FA_HD32_BIAS_STRIDE_H]);
    params.bias_stride_q = uint(dims[FA_HD32_BIAS_STRIDE_Q]);
    params.bias_stride_k = uint(dims[FA_HD32_BIAS_STRIDE_K]);
    params.scale = scale;
    return params;
}

// Computes one output element: row `thread_id / 32`, column `thread_id % 32`.
//
// q, out: (rows, 32) with rows = batch * heads * query_length
// k, v:   (batch * heads, key_length, 32)
// bias:   indexed with the strides in `params`; ignored unless has_bias
//
// The launch must use a 1-D grid of rows * 32 threads and a threadgroup
// size that is a multiple of 32, so every simdgroup maps to exactly one row.
//
// A key whose bias is -inf is skipped. A row where every key is skipped
// has an empty running sum (l == 0) and writes zeros. A finite bias, even
// finfo.min, is never treated as a mask.
template <typename T, typename Bias>
inline void flash_attention_hd32_row(
    const device T* q,
    const device T* k,
    const device T* v,
    Bias bias,
    device T* out,
    FaHd32Params params,
    uint thread_id) {
    const uint row = thread_id / FA_HD32_HEAD_DIM;
    const uint lane = thread_id % FA_HD32_HEAD_DIM;
    // All 32 lanes of a simdgroup share `row`, so they exit together and
    // simd_sum below always sees a full simdgroup.
    if (row >= params.rows) {
        return;
    }
    const uint batch_head = row / params.query_length;
    const uint query = row % params.query_length;
    const uint head_offset = batch_head * params.key_length * FA_HD32_HEAD_DIM;
    const device T* k_head = k + head_offset;
    const device T* v_head = v + head_offset;
    const uint bias_row = (batch_head / params.heads) * params.bias_stride_b +
                          (batch_head % params.heads) * params.bias_stride_h +
                          query * params.bias_stride_q;

    // Fold the softmax scale into log2(e) so the exponentials are exp2.
    const float q_scaled =
        fa_hd32_load(q, row * FA_HD32_HEAD_DIM + lane) * (params.scale * M_LOG2E_F);

    // Running max, sum and output. running_sum == 0 means no key has been
    // accepted yet; running_max is only read after the first accepted key,
    // so no -inf is ever used in arithmetic.
    float running_max = 0.0f;
    float running_sum = 0.0f;
    float acc = 0.0f;

    for (uint j = 0; j < params.key_length; ++j) {
        // The bias and the score depend only on (row, j), so every branch
        // below is uniform across the simdgroup.
        float bias_log2 = 0.0f;
        if (params.has_bias) {
            const float b = fa_hd32_load(bias, bias_row + j * params.bias_stride_k);
            if (fa_hd32_is_neg_inf(b)) {
                continue;
            }
            bias_log2 = fa_hd32_finite(b * M_LOG2E_F);
        }
        const uint offset = j * FA_HD32_HEAD_DIM + lane;
        const float score =
            fa_hd32_finite(simd_sum(q_scaled * fa_hd32_load(k_head, offset)) + bias_log2);
        const float value = fa_hd32_load(v_head, offset);
        if (running_sum == 0.0f) {
            running_max = score;
            running_sum = 1.0f;
            acc = value;
        } else if (score > running_max) {
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

    const float result = running_sum == 0.0f ? 0.0f : acc / running_sum;
    fa_hd32_store(out, row * FA_HD32_HEAD_DIM + lane, result);
}

// PyTorch MPS entry points. `dims` holds FA_HD32_DIM_COUNT int32 values in
// the FaHd32Dim layout; `scale` is a one-element float32 buffer. `bias` must
// be a valid buffer even when has_bias is 0.
template <typename T>
kernel void flash_attention_hd32(
    const device T* q [[buffer(0)]],
    const device T* k [[buffer(1)]],
    const device T* v [[buffer(2)]],
    const device T* bias [[buffer(3)]],
    device T* out [[buffer(4)]],
    const device int* dims [[buffer(5)]],
    const device float* scale [[buffer(6)]],
    uint thread_id [[thread_position_in_grid]]) {
    flash_attention_hd32_row(q, k, v, bias, out, fa_hd32_params(dims, scale[0]), thread_id);
}

#define FA_HD32_INSTANTIATE(NAME, T)                                     \
    template [[host_name("flash_attention_hd32_" #NAME)]] kernel void    \
    flash_attention_hd32<T>(                                             \
        const device T* q [[buffer(0)]],                                 \
        const device T* k [[buffer(1)]],                                 \
        const device T* v [[buffer(2)]],                                 \
        const device T* bias [[buffer(3)]],                              \
        device T* out [[buffer(4)]],                                     \
        const device int* dims [[buffer(5)]],                            \
        const device float* scale [[buffer(6)]],                         \
        uint thread_id [[thread_position_in_grid]]);

FA_HD32_INSTANTIATE(bfloat, ushort)
FA_HD32_INSTANTIATE(float, float)
