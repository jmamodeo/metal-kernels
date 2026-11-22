// SPDX-License-Identifier: Apache-2.0
// FlashAttention for head dim 32, version 1.
//
// softmax(Q Kᵀ * scale + bias) V with an online softmax, so the score matrix
// is never written to memory. One threadgroup of two simdgroups owns 16
// query rows of one (batch, head); each simdgroup owns 8 rows and runs both
// GEMMs on 8x8 simdgroup_float8x8 tiles. K and V are walked in blocks of 64
// keys staged through threadgroup memory. Q, K, V, bias and O share one
// dtype, bf16 or fp32. All arithmetic is fp32; a bf16 O is rounded once at
// the end.
//
// This file is the only copy of the shader. Both hosts compile this text:
// - PyTorch MPS (torch.mps.compile_shader) calls `flash_attention_hd32_bfloat`
//   or `flash_attention_hd32_float` below.
// - MLX (mx.fast.metal_kernel) passes this file as `header` and calls
//   `flash_attention_hd32_block` from the generated kernel body.
//
// See kernels/flash_attention_hd32/README.md for shapes and launch geometry.

#include <metal_stdlib>
using namespace metal;

constant constexpr uint FA_HD32_HEAD_DIM = 32;
constant constexpr uint FA_HD32_SIMD_WIDTH = 32;
constant constexpr uint FA_HD32_TILE = 8;      // simdgroup_matrix side
constant constexpr uint FA_HD32_BLOCK_Q = 16;  // query rows per threadgroup
constant constexpr uint FA_HD32_BLOCK_K = 64;  // keys per K/V block
constant constexpr uint FA_HD32_D_TILES = FA_HD32_HEAD_DIM / FA_HD32_TILE;
constant constexpr uint FA_HD32_K_TILES = FA_HD32_BLOCK_K / FA_HD32_TILE;
constant constexpr uint FA_HD32_THREADS = FA_HD32_BLOCK_Q / FA_HD32_TILE * FA_HD32_SIMD_WIDTH;
constant constexpr uint FA_HD32_TILE_ELEMENTS = FA_HD32_BLOCK_K * FA_HD32_HEAD_DIM;
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

// (column, row) of the two elements a lane holds in an 8x8 simdgroup
// matrix: thread_elements()[e] is element (row, column + e). This is the
// Apple GPU layout that metal-flash-attention (`morton_order`) and MLX
// (steel `BaseMMAFrag`) also rely on. The four lanes that share a row differ
// in lane bits 0 and 3, so row reductions use simd_shuffle_xor with 1 and 8.
inline ushort2 fa_hd32_tile_coord(ushort lane) {
    const ushort quad = lane / 4;
    const ushort row = (quad & 4) + (lane / 2) % 4;
    const ushort column = (quad & 2) * 2 + (lane % 2) * 2;
    return ushort2(column, row);
}

inline float fa_hd32_row_max(float value) {
    value = max(value, simd_shuffle_xor(value, ushort(1)));
    return max(value, simd_shuffle_xor(value, ushort(8)));
}

inline float fa_hd32_row_sum(float value) {
    value += simd_shuffle_xor(value, ushort(1));
    return value + simd_shuffle_xor(value, ushort(8));
}

// Computes a block of FA_HD32_BLOCK_Q query rows of one (batch, head).
//
// q, out: (rows, 32) with rows = batch * heads * query_length
// k, v:   (batch * heads, key_length, 32)
// bias:   indexed with the strides in `params`; ignored unless has_bias
// k_tile, v_tile: threadgroup arrays of FA_HD32_TILE_ELEMENTS floats each
//
// The launch must use FA_HD32_THREADS threads per threadgroup and one
// threadgroup per (batch * head, query block): batch * heads *
// ceil(query_length / FA_HD32_BLOCK_Q) threadgroups. `group` is the
// threadgroup index and `thread_index` the thread index inside it.
//
// Simdgroup s owns query rows [8 s, 8 s + 8) of the block. Q (pre-scaled by
// scale * log2(e)) and the O accumulator stay in registers as four 8x8 fp32
// tiles each. Keys are walked in blocks of FA_HD32_BLOCK_K: the threadgroup
// stages the K and V block in threadgroup memory as fp32 (zero past the last
// key), then each simdgroup computes S = Q Kᵀ, updates the online softmax
// and accumulates O += P V, all on simdgroup_float8x8.
//
// A key whose bias is -inf, or past key_length, gets p = 0 and never moves
// the running max. A row where every key is masked has an empty running
// sum (l == 0) and writes zeros. A finite bias, even finfo.min, is never
// treated as a mask. Query rows past query_length compute on zeros and are
// not stored.
template <typename T, typename Bias>
inline void flash_attention_hd32_block(
    const device T* q,
    const device T* k,
    const device T* v,
    Bias bias,
    device T* out,
    FaHd32Params params,
    threadgroup float* k_tile,
    threadgroup float* v_tile,
    uint group,
    uint thread_index) {
    const uint query_blocks = (params.query_length + FA_HD32_BLOCK_Q - 1) / FA_HD32_BLOCK_Q;
    const uint batch_head = group / query_blocks;
    // Uniform across the threadgroup, so no barrier below is skipped by
    // only some threads.
    if (batch_head * params.query_length >= params.rows) {
        return;
    }
    const uint simdgroup = thread_index / FA_HD32_SIMD_WIDTH;
    const ushort2 coord = fa_hd32_tile_coord(ushort(thread_index % FA_HD32_SIMD_WIDTH));
    const uint query = (group % query_blocks) * FA_HD32_BLOCK_Q +
                       simdgroup * FA_HD32_TILE + coord.y;
    const bool query_valid = query < params.query_length;
    const uint row = batch_head * params.query_length + query;
    const uint head_offset = batch_head * params.key_length * FA_HD32_HEAD_DIM;
    const uint bias_row = (batch_head / params.heads) * params.bias_stride_b +
                          (batch_head % params.heads) * params.bias_stride_h +
                          query * params.bias_stride_q;
    const bool load_bias = params.has_bias && query_valid;
    const float neg_flt_max = as_type<float>(FA_HD32_NEG_FLT_MAX_BITS);

    // Fold the softmax scale into log2(e) so the exponentials are exp2.
    const float q_scale = params.scale * M_LOG2E_F;
    simdgroup_float8x8 q_tiles[FA_HD32_D_TILES];
    simdgroup_float8x8 o_tiles[FA_HD32_D_TILES];
    for (uint d = 0; d < FA_HD32_D_TILES; ++d) {
        float2 values = float2(0.0f);
        if (query_valid) {
            const uint offset = row * FA_HD32_HEAD_DIM + d * FA_HD32_TILE + coord.x;
            values = float2(fa_hd32_load(q, offset), fa_hd32_load(q, offset + 1)) * q_scale;
        }
        q_tiles[d].thread_elements()[0] = values.x;
        q_tiles[d].thread_elements()[1] = values.y;
        o_tiles[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    // Running max and sum of this lane's row. running_max starts at the
    // finite -FLT_MAX and only takes finite scores, so no -inf is ever used
    // in arithmetic. running_sum == 0 means no key has been accepted yet.
    float running_max = neg_flt_max;
    float running_sum = 0.0f;

    for (uint key_start = 0; key_start < params.key_length; key_start += FA_HD32_BLOCK_K) {
        // Stage the K and V block. The first barrier keeps the previous
        // block's tiles alive until every simdgroup is done with them.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint block_keys = params.key_length - key_start;
        const uint block_elements =
            (block_keys < FA_HD32_BLOCK_K ? block_keys : FA_HD32_BLOCK_K) * FA_HD32_HEAD_DIM;
        const uint block_offset = head_offset + key_start * FA_HD32_HEAD_DIM;
        for (uint e = thread_index; e < FA_HD32_TILE_ELEMENTS; e += FA_HD32_THREADS) {
            float k_value = 0.0f;
            float v_value = 0.0f;
            if (e < block_elements) {
                k_value = fa_hd32_load(k, block_offset + e);
                v_value = fa_hd32_load(v, block_offset + e);
            }
            k_tile[e] = k_value;
            v_tile[e] = v_value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // S = Q Kᵀ, one 8x8 tile per 8 keys. Kᵀ tiles are transposed loads
        // of the (key, d) K block.
        simdgroup_float8x8 s_tiles[FA_HD32_K_TILES];
        for (uint j = 0; j < FA_HD32_K_TILES; ++j) {
            s_tiles[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint d = 0; d < FA_HD32_D_TILES; ++d) {
                simdgroup_float8x8 k_t;
                simdgroup_load(
                    k_t,
                    k_tile + j * FA_HD32_TILE * FA_HD32_HEAD_DIM + d * FA_HD32_TILE,
                    FA_HD32_HEAD_DIM,
                    ulong2(0, 0),
                    true);
                simdgroup_multiply_accumulate(s_tiles[j], q_tiles[d], k_t, s_tiles[j]);
            }
        }

        // Add the bias, mask, and take the block's row max over valid keys.
        uint valid_mask = 0;
        float block_max = neg_flt_max;
        for (uint j = 0; j < FA_HD32_K_TILES; ++j) {
            for (uint e = 0; e < 2; ++e) {
                const uint key = key_start + j * FA_HD32_TILE + coord.x + e;
                float score = s_tiles[j].thread_elements()[e];
                bool valid = key < params.key_length;
                if (valid && load_bias) {
                    const float b = fa_hd32_load(bias, bias_row + key * params.bias_stride_k);
                    if (fa_hd32_is_neg_inf(b)) {
                        valid = false;
                    } else {
                        score += fa_hd32_finite(b * M_LOG2E_F);
                    }
                }
                score = fa_hd32_finite(score);
                s_tiles[j].thread_elements()[e] = score;
                if (valid) {
                    valid_mask |= 1u << (2 * j + e);
                    block_max = max(block_max, score);
                }
            }
        }
        block_max = fa_hd32_row_max(block_max);

        // Online softmax. correction is exp2(0) = 1 when the max does not
        // rise, and the difference is clamped so it never becomes -inf.
        const float new_max = max(running_max, block_max);
        const float correction = exp2(fa_hd32_finite(running_max - new_max));
        running_max = new_max;

        float block_sum = 0.0f;
        for (uint j = 0; j < FA_HD32_K_TILES; ++j) {
            for (uint e = 0; e < 2; ++e) {
                const bool valid = (valid_mask >> (2 * j + e)) & 1u;
                const float p = valid ? exp2(s_tiles[j].thread_elements()[e] - running_max) : 0.0f;
                s_tiles[j].thread_elements()[e] = p;
                block_sum += p;
            }
        }
        running_sum = running_sum * correction + fa_hd32_row_sum(block_sum);

        // O = O * correction + P V.
        for (uint d = 0; d < FA_HD32_D_TILES; ++d) {
            o_tiles[d].thread_elements() *= correction;
        }
        for (uint j = 0; j < FA_HD32_K_TILES; ++j) {
            for (uint d = 0; d < FA_HD32_D_TILES; ++d) {
                simdgroup_float8x8 v_t;
                simdgroup_load(
                    v_t,
                    v_tile + j * FA_HD32_TILE * FA_HD32_HEAD_DIM + d * FA_HD32_TILE,
                    FA_HD32_HEAD_DIM);
                simdgroup_multiply_accumulate(o_tiles[d], s_tiles[j], v_t, o_tiles[d]);
            }
        }
    }

    if (!query_valid) {
        return;
    }
    for (uint d = 0; d < FA_HD32_D_TILES; ++d) {
        const uint offset = row * FA_HD32_HEAD_DIM + d * FA_HD32_TILE + coord.x;
        for (uint e = 0; e < 2; ++e) {
            const float value = o_tiles[d].thread_elements()[e];
            fa_hd32_store(out, offset + e, running_sum == 0.0f ? 0.0f : value / running_sum);
        }
    }
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
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float k_tile[FA_HD32_TILE_ELEMENTS];
    threadgroup float v_tile[FA_HD32_TILE_ELEMENTS];
    flash_attention_hd32_block(
        q, k, v, bias, out, fa_hd32_params(dims, scale[0]), k_tile, v_tile, group, thread_index);
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
        uint group [[threadgroup_position_in_grid]],                     \
        uint thread_index [[thread_index_in_threadgroup]]);

FA_HD32_INSTANTIATE(bfloat, ushort)
FA_HD32_INSTANTIATE(float, float)
