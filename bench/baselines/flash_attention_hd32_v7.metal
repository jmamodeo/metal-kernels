// SPDX-License-Identifier: Apache-2.0
// FlashAttention for head dim 32, version 7.
//
// softmax(Q Kᵀ * scale + bias) V with an online softmax, so the score matrix
// is never written to memory. One threadgroup of four simdgroups owns 32
// query rows of one (batch, head); each simdgroup owns 8 rows and runs both
// GEMMs on 8x8 simdgroup_matrix tiles with fp32 accumulators. K and V are
// walked in blocks of 64 keys. All 128 threads copy each block once from
// device memory into threadgroup memory, in the input dtype, and the four
// simdgroups read their tiles from that shared copy. The stage is double
// buffered (16 KB for bf16): the next block's device loads are issued before
// the current block's math and stored into the other buffer after it, so
// there is one barrier per block. Q, K, V, bias
// and O share one dtype, bf16 or fp32. Softmax and accumulation are fp32; a
// bf16 O is rounded once at the end.
//
// This file is the only copy of the shader. Both hosts compile this text:
// - PyTorch MPS (torch.mps.compile_shader) calls `flash_attention_hd32_bfloat`
//   or `flash_attention_hd32_float` below.
// - MLX (mx.fast.metal_kernel) passes this file as `header` and calls
//   `flash_attention_hd32_block` from the generated kernel body, passing a
//   threadgroup buffer it declares with FA_HD32_KV_STAGE.
//
// See kernels/flash_attention_hd32/README.md for shapes and launch geometry.

#include <metal_stdlib>
using namespace metal;

#define FA_HD32_UNROLL _Pragma("clang loop unroll(full)")

constant constexpr uint FA_HD32_HEAD_DIM = 32;
constant constexpr uint FA_HD32_SIMD_WIDTH = 32;
constant constexpr uint FA_HD32_TILE = 8;      // simdgroup_matrix side
constant constexpr uint FA_HD32_BLOCK_Q = 32;  // query rows per threadgroup
constant constexpr uint FA_HD32_BLOCK_K = 64;  // keys per K/V block
constant constexpr uint FA_HD32_D_TILES = FA_HD32_HEAD_DIM / FA_HD32_TILE;
constant constexpr uint FA_HD32_K_TILES = FA_HD32_BLOCK_K / FA_HD32_TILE;
constant constexpr uint FA_HD32_THREADS = FA_HD32_BLOCK_Q / FA_HD32_TILE * FA_HD32_SIMD_WIDTH;
// Elements of one staged K or V block, of one K/V buffer (K then V), and of
// the whole double-buffered K/V stage.
constant constexpr uint FA_HD32_BLOCK_ELEMENTS = FA_HD32_BLOCK_K * FA_HD32_HEAD_DIM;
constant constexpr uint FA_HD32_BUFFER_ELEMENTS = 2 * FA_HD32_BLOCK_ELEMENTS;
constant constexpr uint FA_HD32_KV_ELEMENTS = 2 * FA_HD32_BUFFER_ELEMENTS;
// 4-element vectors each thread copies per K or V block.
constant constexpr uint FA_HD32_COPY_VECTORS = FA_HD32_BLOCK_ELEMENTS / 4 / FA_HD32_THREADS;
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

// Storage type of a buffer element by size: ushort bits for bf16, float for
// fp32. Lets the MLX kernel body, which only sees its buffer types, declare
// the K/V stage in the same type as the block function's T.
template <uint Bytes>
struct FaHd32Storage;
template <>
struct FaHd32Storage<2> {
    typedef ushort type;
};
template <>
struct FaHd32Storage<4> {
    typedef float type;
};

// Declares the threadgroup K/V stage NAME for a kernel whose K buffer is
// PTR. Threadgroup variables are declared at kernel scope and passed in.
#define FA_HD32_KV_STAGE(NAME, PTR) \
    threadgroup FaHd32Storage<sizeof((PTR)[0])>::type NAME[FA_HD32_KV_ELEMENTS]

// Per-storage-type load helpers and the element type fed to the GEMMs. bf16
// tiles are simdgroup_bfloat8x8 when the compiler has `bfloat` (Metal 3.1);
// otherwise bf16 is widened to fp32 tiles. fp32 inputs use fp32 tiles.
// Accumulators (S and O) are always fp32.
template <typename T>
struct FaHd32Io;

template <>
struct FaHd32Io<float> {
    typedef float mma_type;
    typedef float4 vec4;
    static inline vec4 load4(const device float* data, uint index) {
        return *reinterpret_cast<const device float4*>(data + index);
    }
    static inline vec4 load4(const threadgroup float* data, uint index) {
        return *reinterpret_cast<const threadgroup float4*>(data + index);
    }
    static inline void store4(threadgroup float* data, uint index, vec4 value) {
        *reinterpret_cast<threadgroup float4*>(data + index) = value;
    }
    static inline mma_type element(float value) {
        return value;
    }
};

template <>
struct FaHd32Io<ushort> {
#if defined(__HAVE_BFLOAT__)
    typedef bfloat mma_type;
    static inline mma_type element(ushort bits) {
        return as_type<bfloat>(bits);
    }
#else
    typedef float mma_type;
    static inline mma_type element(ushort bits) {
        return as_type<float>(uint(bits) << 16);
    }
#endif
    typedef ushort4 vec4;
    static inline vec4 load4(const device ushort* data, uint index) {
        return *reinterpret_cast<const device ushort4*>(data + index);
    }
    static inline vec4 load4(const threadgroup ushort* data, uint index) {
        return *reinterpret_cast<const threadgroup ushort4*>(data + index);
    }
    static inline void store4(threadgroup ushort* data, uint index, vec4 value) {
        *reinterpret_cast<threadgroup ushort4*>(data + index) = value;
    }
};

// One thread's share of a K and V block, held in registers between the
// device load and the threadgroup store.
template <typename T>
struct FaHd32Copy {
    typename FaHd32Io<T>::vec4 k[FA_HD32_COPY_VECTORS];
    typename FaHd32Io<T>::vec4 v[FA_HD32_COPY_VECTORS];
};

// Loads this thread's share of keys [key_start, key_start + FA_HD32_BLOCK_K)
// of one head's K and V into registers, zero past key_length. The
// threadgroup's FA_HD32_THREADS threads each take every FA_HD32_THREADS-th
// 4-element vector, so consecutive threads read consecutive addresses.
template <typename T>
inline FaHd32Copy<T> fa_hd32_load_block(
    const device T* k_head,
    const device T* v_head,
    uint key_start,
    uint key_length,
    uint thread_index) {
    typedef FaHd32Io<T> Io;
    typedef typename Io::vec4 V4;
    const uint block = key_start * FA_HD32_HEAD_DIM;
    FaHd32Copy<T> copy;
    FA_HD32_UNROLL
    for (uint n = 0; n < FA_HD32_COPY_VECTORS; ++n) {
        const uint index = 4 * (thread_index + n * FA_HD32_THREADS);
        const bool valid = key_start + index / FA_HD32_HEAD_DIM < key_length;
        copy.k[n] = valid ? Io::load4(k_head, block + index) : V4(0);
        copy.v[n] = valid ? Io::load4(v_head, block + index) : V4(0);
    }
    return copy;
}

// Stores a share loaded by fa_hd32_load_block into one K/V buffer: K at
// [0, FA_HD32_BLOCK_ELEMENTS), V after it.
template <typename T>
inline void fa_hd32_store_block(
    threadgroup T* dst,
    thread const FaHd32Copy<T>& copy,
    uint thread_index) {
    typedef FaHd32Io<T> Io;
    FA_HD32_UNROLL
    for (uint n = 0; n < FA_HD32_COPY_VECTORS; ++n) {
        const uint index = 4 * (thread_index + n * FA_HD32_THREADS);
        Io::store4(dst, index, copy.k[n]);
        Io::store4(dst + FA_HD32_BLOCK_ELEMENTS, index, copy.v[n]);
    }
}

// Computes a block of FA_HD32_BLOCK_Q query rows of one (batch, head).
//
// q, out: (rows, 32) with rows = batch * heads * query_length
// k, v:   (batch * heads, key_length, 32)
// bias:   indexed with the strides in `params`; ignored unless has_bias
//
// The launch must use FA_HD32_THREADS threads per threadgroup and one
// threadgroup per (batch * head, query block): batch * heads *
// ceil(query_length / FA_HD32_BLOCK_Q) threadgroups. `group` is the
// threadgroup index and `thread_index` the thread index inside it. `kv` is
// the threadgroup's FA_HD32_KV_ELEMENTS-element stage: two buffers of
// FA_HD32_BUFFER_ELEMENTS, each K at [0, FA_HD32_BLOCK_ELEMENTS) and V after
// it. Every thread of the threadgroup must call this function, because the
// K/V copy uses barriers.
//
// Simdgroup s owns query rows [8 s, 8 s + 8) of the block. Q and the fp32 O
// accumulator stay in registers as four 8x8 tiles each. Keys are walked in
// blocks of FA_HD32_BLOCK_K; block b lives in buffer b % 2 (zero past the
// last key). The whole threadgroup copies block 0 into buffer 0 before the
// loop, then a barrier. In iteration b each thread first issues its device
// loads of block b + 1 into registers, then each lane reads its K and V tile
// elements of block b from buffer b % 2 and the simdgroup computes
// S = Q Kᵀ, updates the online softmax and accumulates O += P V; then the
// thread stores block b + 1 into buffer (b + 1) % 2 and the threadgroup
// meets at one barrier. That barrier both publishes block b + 1 and keeps
// iteration b + 1's store into buffer b % 2 from overwriting block b while
// another simdgroup still reads it. A simdgroup whose rows are all past
// query_length still copies and waits at the barriers, but skips the math.
//
// The head dimension is split as d = 4 i + t across the four tiles t, with
// i the index inside a tile. Both GEMMs sum or index over d consistently, so
// the result is unchanged, and every lane's Q, K, V and O elements become
// contiguous runs of 4 or 8 values that load and store as vectors.
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
    threadgroup T* kv,
    FaHd32Params params,
    uint group,
    uint thread_index) {
    typedef FaHd32Io<T> Io;
    typedef typename Io::mma_type M;
    typedef typename Io::vec4 V4;

    const uint query_blocks = (params.query_length + FA_HD32_BLOCK_Q - 1) / FA_HD32_BLOCK_Q;
    const uint batch_head = group / query_blocks;
    const uint simdgroup = thread_index / FA_HD32_SIMD_WIDTH;
    const uint first_query = (group % query_blocks) * FA_HD32_BLOCK_Q + simdgroup * FA_HD32_TILE;
    // Uniform across the threadgroup, so no thread is left at a barrier.
    if (batch_head * params.query_length >= params.rows) {
        return;
    }
    // Uniform across the simdgroup, so every shuffle and matrix operation
    // below still sees all 32 lanes.
    const bool simdgroup_active = first_query < params.query_length;
    const ushort2 coord = fa_hd32_tile_coord(ushort(thread_index % FA_HD32_SIMD_WIDTH));
    const uint query = first_query + coord.y;
    const bool query_valid = query < params.query_length;
    const uint row = batch_head * params.query_length + query;
    const uint head_offset = batch_head * params.key_length * FA_HD32_HEAD_DIM;
    const device T* k_head = k + head_offset;
    const device T* v_head = v + head_offset;
    const uint bias_row = (batch_head / params.heads) * params.bias_stride_b +
                          (batch_head % params.heads) * params.bias_stride_h +
                          query * params.bias_stride_q;
    const bool load_bias = params.has_bias && query_valid;
    const float neg_flt_max = as_type<float>(FA_HD32_NEG_FLT_MAX_BITS);
    // Fold the softmax scale into log2(e) so the exponentials are exp2.
    const float qk_scale = params.scale * M_LOG2E_F;

    // Q tile t, element (r, c + e) is Q[r][4 (c + e) + t]: the lane's eight
    // elements are Q[r][4 c .. 4 c + 8).
    simdgroup_matrix<M, 8, 8> q_tiles[FA_HD32_D_TILES];
    simdgroup_float8x8 o_tiles[FA_HD32_D_TILES];
    {
        V4 q_lo = V4(0);
        V4 q_hi = V4(0);
        if (query_valid) {
            const uint offset = row * FA_HD32_HEAD_DIM + 4 * coord.x;
            q_lo = Io::load4(q, offset);
            q_hi = Io::load4(q, offset + 4);
        }
        FA_HD32_UNROLL
        for (uint t = 0; t < FA_HD32_D_TILES; ++t) {
            q_tiles[t].thread_elements()[0] = Io::element(q_lo[t]);
            q_tiles[t].thread_elements()[1] = Io::element(q_hi[t]);
            o_tiles[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    // Running max and sum of this lane's row. running_max starts at the
    // finite -FLT_MAX and only takes finite scores, so no -inf is ever used
    // in arithmetic. running_sum == 0 means no key has been accepted yet.
    float running_max = neg_flt_max;
    float running_sum = 0.0f;

    {
        const FaHd32Copy<T> first =
            fa_hd32_load_block(k_head, v_head, 0, params.key_length, thread_index);
        fa_hd32_store_block(kv, first, thread_index);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint stage = 0;
    for (uint key_start = 0; key_start < params.key_length; key_start += FA_HD32_BLOCK_K) {
        const bool full_block = params.key_length - key_start >= FA_HD32_BLOCK_K;
        // Uniform across the threadgroup, like the trip count, so every
        // thread reaches the same barriers.
        const bool has_next = params.key_length - key_start > FA_HD32_BLOCK_K;
        const threadgroup T* k_stage = kv + stage * FA_HD32_BUFFER_ELEMENTS;
        const threadgroup T* v_stage = k_stage + FA_HD32_BLOCK_ELEMENTS;

        // Issue the next block's device loads before the math so their
        // latency hides behind it. Past key_length they load nothing.
        const FaHd32Copy<T> next = fa_hd32_load_block(
            k_head, v_head, key_start + FA_HD32_BLOCK_K, has_next ? params.key_length : 0u,
            thread_index);
        if (simdgroup_active) {
            // S = Q Kᵀ. Kᵀ tile t, element (r, c + e) is K[key c + e][4 r + t]:
            // one 4-element load per key. Keys past key_length are zero in the
            // stage.
            simdgroup_float8x8 s_tiles[FA_HD32_K_TILES];
            FA_HD32_UNROLL
            for (uint j = 0; j < FA_HD32_K_TILES; ++j) {
                const uint offset = (j * FA_HD32_TILE + coord.x) * FA_HD32_HEAD_DIM + 4 * coord.y;
                const V4 k_lo = Io::load4(k_stage, offset);
                const V4 k_hi = Io::load4(k_stage, offset + FA_HD32_HEAD_DIM);
                s_tiles[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                FA_HD32_UNROLL
                for (uint t = 0; t < FA_HD32_D_TILES; ++t) {
                    simdgroup_matrix<M, 8, 8> k_t;
                    k_t.thread_elements()[0] = Io::element(k_lo[t]);
                    k_t.thread_elements()[1] = Io::element(k_hi[t]);
                    simdgroup_multiply_accumulate(s_tiles[j], q_tiles[t], k_t, s_tiles[j]);
                }
            }

            // Scale, add the bias, mask, and take the block's row max over
            // valid keys.
            bool valid[FA_HD32_K_TILES][2];
            float block_max = neg_flt_max;
            FA_HD32_UNROLL
            for (uint j = 0; j < FA_HD32_K_TILES; ++j) {
                FA_HD32_UNROLL
                for (uint e = 0; e < 2; ++e) {
                    const uint key = key_start + j * FA_HD32_TILE + coord.x + e;
                    float score = s_tiles[j].thread_elements()[e] * qk_scale;
                    bool ok = full_block || key < params.key_length;
                    if (ok && load_bias) {
                        const float b = fa_hd32_load(bias, bias_row + key * params.bias_stride_k);
                        if (fa_hd32_is_neg_inf(b)) {
                            ok = false;
                        } else {
                            score += fa_hd32_finite(b * M_LOG2E_F);
                        }
                    }
                    score = fa_hd32_finite(score);
                    s_tiles[j].thread_elements()[e] = score;
                    valid[j][e] = ok;
                    block_max = ok ? max(block_max, score) : block_max;
                }
            }
            block_max = fa_hd32_row_max(block_max);

            // Online softmax. correction is exp2(0) = 1 when the max does not
            // rise, and the difference is clamped so it never becomes -inf.
            const float new_max = max(running_max, block_max);
            const float correction = exp2(fa_hd32_finite(running_max - new_max));
            running_max = new_max;

            float block_sum = 0.0f;
            FA_HD32_UNROLL
            for (uint j = 0; j < FA_HD32_K_TILES; ++j) {
                FA_HD32_UNROLL
                for (uint e = 0; e < 2; ++e) {
                    const float p =
                        valid[j][e] ? exp2(s_tiles[j].thread_elements()[e] - running_max) : 0.0f;
                    s_tiles[j].thread_elements()[e] = p;
                    block_sum += p;
                }
            }
            running_sum = running_sum * correction + fa_hd32_row_sum(block_sum);

            // O = O * correction + P V. P stays fp32. V tile t, element
            // (r, c + e) is V[key r][4 (c + e) + t]: the lane's eight elements
            // are V[key r][4 c .. 4 c + 8).
            FA_HD32_UNROLL
            for (uint t = 0; t < FA_HD32_D_TILES; ++t) {
                o_tiles[t].thread_elements() *= correction;
            }
            FA_HD32_UNROLL
            for (uint j = 0; j < FA_HD32_K_TILES; ++j) {
                const uint offset = (j * FA_HD32_TILE + coord.y) * FA_HD32_HEAD_DIM + 4 * coord.x;
                const V4 v_lo = Io::load4(v_stage, offset);
                const V4 v_hi = Io::load4(v_stage, offset + 4);
                FA_HD32_UNROLL
                for (uint t = 0; t < FA_HD32_D_TILES; ++t) {
                    simdgroup_matrix<M, 8, 8> v_t;
                    v_t.thread_elements()[0] = Io::element(v_lo[t]);
                    v_t.thread_elements()[1] = Io::element(v_hi[t]);
                    simdgroup_multiply_accumulate(o_tiles[t], s_tiles[j], v_t, o_tiles[t]);
                }
            }
        }
        if (has_next) {
            stage ^= 1u;
            fa_hd32_store_block(kv + stage * FA_HD32_BUFFER_ELEMENTS, next, thread_index);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (!query_valid) {
        return;
    }
    // O tile t, element (r, c + e) is O[r][4 (c + e) + t], so the lane
    // writes O[r][4 c .. 4 c + 8) directly to device memory.
    const float inv_sum = running_sum == 0.0f ? 0.0f : 1.0f / running_sum;
    const uint offset = row * FA_HD32_HEAD_DIM + 4 * coord.x;
    FA_HD32_UNROLL
    for (uint e = 0; e < 2; ++e) {
        FA_HD32_UNROLL
        for (uint t = 0; t < FA_HD32_D_TILES; ++t) {
            fa_hd32_store(out, offset + 4 * e + t, o_tiles[t].thread_elements()[e] * inv_sum);
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
    threadgroup T kv[FA_HD32_KV_ELEMENTS];
    flash_attention_hd32_block(
        q, k, v, bias, out, kv, fa_hd32_params(dims, scale[0]), group, thread_index);
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
