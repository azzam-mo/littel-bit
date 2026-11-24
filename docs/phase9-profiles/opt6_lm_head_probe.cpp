// Phase 9 Optimization 6 investigation probe.
//
// Measures the current AVX2 LM-head kernel and candidate access patterns on the
// real Qwen3/LittleBit BF16 lm_head table. Candidates are compared bit-for-bit
// against the production scalar LM-head path (and the current AVX2 kernel).
// Nothing here is part of the engine build.

#include "littlebit/common.h"
#include "littlebit/safetensors.h"
#include "lm_head_avx2.h"

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

uint32_t bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

float from_bits(uint32_t u) {
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// Same loop shape as src/lm_head_avx2.cpp scalar_lm_head. Used as the fallback
// for odd shapes and as an independent local oracle.
__attribute__((noinline))
void scalar_lm_head(
    const uint16_t* W, const float* h, float* out, size_t V, size_t H
) {
    for (size_t v = 0; v < V; ++v) {
        const uint16_t* row = W + v * H;
        float dot = 0.0f;
        for (size_t i = 0; i < H; ++i) {
            dot += h[i] * littlebit::bf16_to_fp32(row[i]);
        }
        out[v] = dot;
    }
}

__attribute__((target("avx2"), always_inline))
static inline __m256 add_ordered(__m256 src1, __m256 src2) {
    __m256 result;
    __asm__ volatile("vaddps %2, %1, %0" : "=x"(result) : "x"(src1), "x"(src2));
    return result;
}

__attribute__((target("avx2"), always_inline))
static inline __m128 addss_ordered(__m128 src1, __m128 src2) {
    __m128 result;
    __asm__ volatile("vaddss %2, %1, %0" : "=x"(result) : "x"(src1), "x"(src2));
    return result;
}

__attribute__((target("avx2"), always_inline))
static inline __m256 bf16_lane_to_float(__m128i bits16) {
    const __m256i w32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(bits16), 16);
    return _mm256_castsi256_ps(w32);
}

__attribute__((target("avx2"), always_inline))
static inline __m256 mul_weight_hidden(__m256 w, __m256 hv) {
    __m256 p;
    __asm__ volatile("vmulps %2, %1, %0" : "=x"(p) : "x"(w), "x"(hv));
    return p;
}

// Exact 8-step accumulation pattern used by the compiled scalar oracle.
__attribute__((target("avx2"), always_inline))
static inline __m256 step8_from_columns(
    const __m128i* c, const float* h, size_t j, __m256 sum
) {
    sum = add_ordered(mul_weight_hidden(bf16_lane_to_float(c[0]), _mm256_set1_ps(h[j + 0])), sum);
    sum = add_ordered(mul_weight_hidden(bf16_lane_to_float(c[1]), _mm256_set1_ps(h[j + 1])), sum);
    sum = add_ordered(sum, mul_weight_hidden(bf16_lane_to_float(c[2]), _mm256_set1_ps(h[j + 2])));
    sum = add_ordered(sum, mul_weight_hidden(bf16_lane_to_float(c[3]), _mm256_set1_ps(h[j + 3])));
    sum = add_ordered(sum, mul_weight_hidden(bf16_lane_to_float(c[4]), _mm256_set1_ps(h[j + 4])));
    sum = add_ordered(sum, mul_weight_hidden(bf16_lane_to_float(c[5]), _mm256_set1_ps(h[j + 5])));
    sum = add_ordered(sum, mul_weight_hidden(bf16_lane_to_float(c[6]), _mm256_set1_ps(h[j + 6])));
    sum = add_ordered(sum, mul_weight_hidden(bf16_lane_to_float(c[7]), _mm256_set1_ps(h[j + 7])));
    return sum;
}

// 8x8 transpose of packed uint16 rows. Each input register holds eight
// consecutive hidden elements of one vocabulary row; each output register holds
// one hidden element from eight consecutive vocabulary rows.
__attribute__((target("avx2"), always_inline))
static inline void transpose8x8_u16(const __m128i* in, __m128i* out) {
    const __m128i t0 = _mm_unpacklo_epi16(in[0], in[1]);
    const __m128i t1 = _mm_unpackhi_epi16(in[0], in[1]);
    const __m128i t2 = _mm_unpacklo_epi16(in[2], in[3]);
    const __m128i t3 = _mm_unpackhi_epi16(in[2], in[3]);
    const __m128i t4 = _mm_unpacklo_epi16(in[4], in[5]);
    const __m128i t5 = _mm_unpackhi_epi16(in[4], in[5]);
    const __m128i t6 = _mm_unpacklo_epi16(in[6], in[7]);
    const __m128i t7 = _mm_unpackhi_epi16(in[6], in[7]);

    const __m128i s0 = _mm_unpacklo_epi32(t0, t2);
    const __m128i s1 = _mm_unpackhi_epi32(t0, t2);
    const __m128i s2 = _mm_unpacklo_epi32(t1, t3);
    const __m128i s3 = _mm_unpackhi_epi32(t1, t3);
    const __m128i s4 = _mm_unpacklo_epi32(t4, t6);
    const __m128i s5 = _mm_unpackhi_epi32(t4, t6);
    const __m128i s6 = _mm_unpacklo_epi32(t5, t7);
    const __m128i s7 = _mm_unpackhi_epi32(t5, t7);

    out[0] = _mm_unpacklo_epi64(s0, s4);
    out[1] = _mm_unpackhi_epi64(s0, s4);
    out[2] = _mm_unpacklo_epi64(s1, s5);
    out[3] = _mm_unpackhi_epi64(s1, s5);
    out[4] = _mm_unpacklo_epi64(s2, s6);
    out[5] = _mm_unpackhi_epi64(s2, s6);
    out[6] = _mm_unpacklo_epi64(s3, s7);
    out[7] = _mm_unpackhi_epi64(s3, s7);
}

// Candidate A: contiguous 16-bit row loads plus an 8x8 register transpose.
// Eight vocabulary rows stay in eight AVX2 lanes, so the accumulation order is
// identical to the compiled scalar oracle while the gathers disappear.
__attribute__((target("avx2")))
void avx2_transpose8(
    const uint16_t* W, const float* h, float* out, size_t V, size_t H
) {
    if (H % 8 != 0 || V < 8) {
        scalar_lm_head(W, h, out, V, H);
        return;
    }
    size_t v = 0;
    for (; v + 8 <= V; v += 8) {
        const uint16_t* rows[8];
        for (int r = 0; r < 8; ++r) rows[r] = W + (v + r) * H;
        __m256 sum = _mm256_setzero_ps();
        for (size_t j = 0; j < H; j += 8) {
            __m128i in[8];
            for (int r = 0; r < 8; ++r) {
                in[r] = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(rows[r] + j)
                );
            }
            __m128i c[8];
            transpose8x8_u16(in, c);
            sum = step8_from_columns(c, h, j, sum);
        }
        _mm256_storeu_ps(out + v, sum);
    }
    if (v < V) scalar_lm_head(W + v * H, h, out + v, V - v, H);
}

// Candidate B2: two 8x8 transposes at once in 256-bit registers. Each ymm input
// holds 16 consecutive hidden elements of one row, and each ymm output holds
// column t (lane 0) and column t+8 (lane 1) for eight rows. Halves the
// load/transpose instruction count per element versus the 128-bit variant.
__attribute__((target("avx2"), always_inline))
static inline void transpose8x8x2_u16(const __m256i* in, __m256i* out) {
    const __m256i t0 = _mm256_unpacklo_epi16(in[0], in[1]);
    const __m256i t1 = _mm256_unpackhi_epi16(in[0], in[1]);
    const __m256i t2 = _mm256_unpacklo_epi16(in[2], in[3]);
    const __m256i t3 = _mm256_unpackhi_epi16(in[2], in[3]);
    const __m256i t4 = _mm256_unpacklo_epi16(in[4], in[5]);
    const __m256i t5 = _mm256_unpackhi_epi16(in[4], in[5]);
    const __m256i t6 = _mm256_unpacklo_epi16(in[6], in[7]);
    const __m256i t7 = _mm256_unpackhi_epi16(in[6], in[7]);

    const __m256i s0 = _mm256_unpacklo_epi32(t0, t2);
    const __m256i s1 = _mm256_unpackhi_epi32(t0, t2);
    const __m256i s2 = _mm256_unpacklo_epi32(t1, t3);
    const __m256i s3 = _mm256_unpackhi_epi32(t1, t3);
    const __m256i s4 = _mm256_unpacklo_epi32(t4, t6);
    const __m256i s5 = _mm256_unpackhi_epi32(t4, t6);
    const __m256i s6 = _mm256_unpacklo_epi32(t5, t7);
    const __m256i s7 = _mm256_unpackhi_epi32(t5, t7);

    out[0] = _mm256_unpacklo_epi64(s0, s4);
    out[1] = _mm256_unpackhi_epi64(s0, s4);
    out[2] = _mm256_unpacklo_epi64(s1, s5);
    out[3] = _mm256_unpackhi_epi64(s1, s5);
    out[4] = _mm256_unpacklo_epi64(s2, s6);
    out[5] = _mm256_unpackhi_epi64(s2, s6);
    out[6] = _mm256_unpacklo_epi64(s3, s7);
    out[7] = _mm256_unpackhi_epi64(s3, s7);
}

template <int T>
__attribute__((target("avx2"), always_inline))
static inline __m256 step_column(__m128i c, float hv, __m256 sum) {
    const __m256 p = mul_weight_hidden(bf16_lane_to_float(c), _mm256_set1_ps(hv));
    if constexpr (T < 2) {
        return add_ordered(p, sum);
    } else {
        return add_ordered(sum, p);
    }
}

__attribute__((target("avx2")))
void avx2_transpose16wide(
    const uint16_t* W, const float* h, float* out, size_t V, size_t H
) {
    if (H % 16 != 0 || V < 8) {
        avx2_transpose8(W, h, out, V, H);
        return;
    }
    size_t v = 0;
    for (; v + 8 <= V; v += 8) {
        const uint16_t* rows[8];
        for (int r = 0; r < 8; ++r) rows[r] = W + (v + r) * H;
        __m256 sum = _mm256_setzero_ps();
        for (size_t j = 0; j < H; j += 16) {
            __m256i in[8];
            __m256i o[8];
            for (int r = 0; r < 8; ++r) {
                in[r] = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(rows[r] + j)
                );
            }
            transpose8x8x2_u16(in, o);
            sum = step_column<0>(_mm256_castsi256_si128(o[0]), h[j + 0], sum);
            sum = step_column<1>(_mm256_castsi256_si128(o[1]), h[j + 1], sum);
            sum = step_column<2>(_mm256_castsi256_si128(o[2]), h[j + 2], sum);
            sum = step_column<3>(_mm256_castsi256_si128(o[3]), h[j + 3], sum);
            sum = step_column<4>(_mm256_castsi256_si128(o[4]), h[j + 4], sum);
            sum = step_column<5>(_mm256_castsi256_si128(o[5]), h[j + 5], sum);
            sum = step_column<6>(_mm256_castsi256_si128(o[6]), h[j + 6], sum);
            sum = step_column<7>(_mm256_castsi256_si128(o[7]), h[j + 7], sum);
            sum = step_column<0>(_mm256_extracti128_si256(o[0], 1), h[j + 8], sum);
            sum = step_column<1>(_mm256_extracti128_si256(o[1], 1), h[j + 9], sum);
            sum = step_column<2>(_mm256_extracti128_si256(o[2], 1), h[j + 10], sum);
            sum = step_column<3>(_mm256_extracti128_si256(o[3], 1), h[j + 11], sum);
            sum = step_column<4>(_mm256_extracti128_si256(o[4], 1), h[j + 12], sum);
            sum = step_column<5>(_mm256_extracti128_si256(o[5], 1), h[j + 13], sum);
            sum = step_column<6>(_mm256_extracti128_si256(o[6], 1), h[j + 14], sum);
            sum = step_column<7>(_mm256_extracti128_si256(o[7], 1), h[j + 15], sum);
        }
        _mm256_storeu_ps(out + v, sum);
    }
    if (v < V) avx2_transpose8(W + v * H, h, out + v, V - v, H);
}

// Candidate B: two independent 8-row groups per iteration. The second group's
// transposes can overlap the first group's dependent accumulation chain.
__attribute__((target("avx2")))
void avx2_transpose16(
    const uint16_t* W, const float* h, float* out, size_t V, size_t H
) {
    if (H % 8 != 0 || V < 16) {
        if (H % 8 != 0 || V < 8) {
            scalar_lm_head(W, h, out, V, H);
            return;
        }
        avx2_transpose8(W, h, out, V, H);
        return;
    }
    size_t v = 0;
    for (; v + 16 <= V; v += 16) {
        const uint16_t* rows[16];
        for (int r = 0; r < 16; ++r) rows[r] = W + (v + r) * H;
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        for (size_t j = 0; j < H; j += 8) {
            __m128i in[8];
            __m128i c[8];
            for (int r = 0; r < 8; ++r) {
                in[r] = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(rows[r] + j)
                );
            }
            transpose8x8_u16(in, c);
            sum0 = step8_from_columns(c, h, j, sum0);
            for (int r = 0; r < 8; ++r) {
                in[r] = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(rows[8 + r] + j)
                );
            }
            transpose8x8_u16(in, c);
            sum1 = step8_from_columns(c, h, j, sum1);
        }
        _mm256_storeu_ps(out + v, sum0);
        _mm256_storeu_ps(out + v + 8, sum1);
    }
    if (v < V) {
        if (V - v >= 8) avx2_transpose8(W + v * H, h, out + v, V - v, H);
        else scalar_lm_head(W + v * H, h, out + v, V - v, H);
    }
}

// Candidate C: two independent 8-row groups with the per-column accumulation
// explicitly interleaved, so the second group's dependency chain covers the
// first group's add latency.
__attribute__((target("avx2")))
void avx2_transpose16_pipe(
    const uint16_t* W, const float* h, float* out, size_t V, size_t H
) {
    if (H % 8 != 0 || V < 16) {
        avx2_transpose8(W, h, out, V, H);
        return;
    }
    size_t v = 0;
    for (; v + 16 <= V; v += 16) {
        const uint16_t* rows[16];
        for (int r = 0; r < 16; ++r) rows[r] = W + (v + r) * H;
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        for (size_t j = 0; j < H; j += 8) {
            __m128i in[8];
            __m128i c0[8];
            __m128i c1[8];
            for (int r = 0; r < 8; ++r) {
                in[r] = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(rows[r] + j)
                );
            }
            transpose8x8_u16(in, c0);
            for (int r = 0; r < 8; ++r) {
                in[r] = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(rows[8 + r] + j)
                );
            }
            transpose8x8_u16(in, c1);

#define OPT6_STEP(T, C, S)                                                    \
    S = (T) < 2                                                               \
        ? add_ordered(                                                        \
              mul_weight_hidden(                                              \
                  bf16_lane_to_float(C), _mm256_set1_ps(h[j + (T)])),         \
              S)                                                              \
        : add_ordered(                                                        \
              S,                                                              \
              mul_weight_hidden(                                              \
                  bf16_lane_to_float(C), _mm256_set1_ps(h[j + (T)])))
            OPT6_STEP(0, c0[0], sum0); OPT6_STEP(0, c1[0], sum1);
            OPT6_STEP(1, c0[1], sum0); OPT6_STEP(1, c1[1], sum1);
            OPT6_STEP(2, c0[2], sum0); OPT6_STEP(2, c1[2], sum1);
            OPT6_STEP(3, c0[3], sum0); OPT6_STEP(3, c1[3], sum1);
            OPT6_STEP(4, c0[4], sum0); OPT6_STEP(4, c1[4], sum1);
            OPT6_STEP(5, c0[5], sum0); OPT6_STEP(5, c1[5], sum1);
            OPT6_STEP(6, c0[6], sum0); OPT6_STEP(6, c1[6], sum1);
            OPT6_STEP(7, c0[7], sum0); OPT6_STEP(7, c1[7], sum1);
#undef OPT6_STEP
        }
        _mm256_storeu_ps(out + v, sum0);
        _mm256_storeu_ps(out + v + 8, sum1);
    }
    if (v < V) avx2_transpose8(W + v * H, h, out + v, V - v, H);
}

// Candidate D: no transpose. Eight rows are walked with contiguous SSE loads
// and eight independent scalar accumulation chains.
__attribute__((target("avx2")))
void avx2_independent_rows(
    const uint16_t* W, const float* h, float* out, size_t V, size_t H
) {
    if (H % 8 != 0 || V < 8) {
        scalar_lm_head(W, h, out, V, H);
        return;
    }
    const __m128i zero = _mm_setzero_si128();
    size_t v = 0;
    for (; v + 8 <= V; v += 8) {
        const uint16_t* rows[8];
        for (int r = 0; r < 8; ++r) rows[r] = W + (v + r) * H;
        __m128 acc[8];
        for (int r = 0; r < 8; ++r) acc[r] = _mm_setzero_ps();
        for (size_t j = 0; j < H; j += 8) {
            const __m128 h_lo = _mm_loadu_ps(h + j);
            const __m128 h_hi = _mm_loadu_ps(h + j + 4);
            for (int r = 0; r < 8; ++r) {
                const __m128i raw = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(rows[r] + j)
                );
                __m128i hi = _mm_unpackhi_epi16(raw, zero);
                __m128i lo = _mm_unpacklo_epi16(raw, zero);
                hi = _mm_slli_epi32(hi, 16);
                lo = _mm_slli_epi32(lo, 16);
                const __m128 p_lo = _mm_mul_ps(_mm_castsi128_ps(lo), h_lo);
                const __m128 p_hi = _mm_mul_ps(_mm_castsi128_ps(hi), h_hi);

                __m128 s = acc[r];
                __m128 tmp = p_lo;
                s = addss_ordered(tmp, s);
                tmp = _mm_shuffle_ps(p_lo, p_lo, 0x55);
                s = addss_ordered(tmp, s);
                tmp = _mm_unpackhi_ps(p_lo, p_lo);
                s = addss_ordered(s, tmp);
                tmp = _mm_shuffle_ps(p_lo, p_lo, 0xff);
                s = addss_ordered(s, tmp);
                tmp = p_hi;
                s = addss_ordered(s, tmp);
                tmp = _mm_shuffle_ps(p_hi, p_hi, 0x55);
                s = addss_ordered(s, tmp);
                tmp = _mm_unpackhi_ps(p_hi, p_hi);
                s = addss_ordered(s, tmp);
                tmp = _mm_shuffle_ps(p_hi, p_hi, 0xff);
                s = addss_ordered(s, tmp);
                acc[r] = s;
            }
        }
        for (int r = 0; r < 8; ++r) out[v + r] = _mm_cvtss_f32(acc[r]);
    }
    if (v < V) scalar_lm_head(W + v * H, h, out + v, V - v, H);
}

// Streaming diagnostics: read only, read+BF16 convert, read+convert+multiply.
__attribute__((target("avx2")))
float diag_memory_only(const uint16_t* W, size_t n) {
    __m256i acc = _mm256_setzero_si256();
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        acc = _mm256_xor_si256(acc, _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(W + i)
        ));
    }
    __m128i lo = _mm256_castsi256_si128(acc);
    __m128i hi = _mm256_extracti128_si256(acc, 1);
    lo = _mm_xor_si128(lo, hi);
    alignas(16) uint32_t tmp[4];
    _mm_storeu_si128(reinterpret_cast<__m128i*>(tmp), lo);
    float s = 0.0f;
    for (int k = 0; k < 4; ++k) s += static_cast<float>(tmp[k]);
    for (; i < n; ++i) s += static_cast<float>(W[i]);
    return s;
}

__attribute__((target("avx2")))
float diag_convert_only(const uint16_t* W, size_t n) {
    __m256i acc = _mm256_setzero_si256();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m128i raw = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(W + i)
        );
        acc = _mm256_add_epi32(acc, _mm256_cvtepu16_epi32(raw));
    }
    alignas(32) uint32_t tmp[8];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), acc);
    float s = 0.0f;
    for (int k = 0; k < 8; ++k) s += static_cast<float>(tmp[k]);
    for (; i < n; ++i) s += static_cast<float>(W[i]);
    return s;
}

__attribute__((target("avx2")))
float diag_multiply_only(const uint16_t* W, const float* h, size_t n, size_t H) {
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m128i raw = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(W + i)
        );
        const __m256 w = bf16_lane_to_float(raw);
        __m256 p;
        __asm__ volatile("vmulps %2, %1, %0" : "=x"(p) : "x"(w), "x"(_mm256_set1_ps(h[i % H])));
        acc = _mm256_add_ps(acc, p);
    }
    alignas(32) float tmp[8];
    _mm256_storeu_ps(tmp, acc);
    float s = 0.0f;
    for (int k = 0; k < 8; ++k) s += tmp[k];
    for (; i < n; ++i) s += littlebit::bf16_to_fp32(W[i]) * h[i % H];
    return s;
}

using Kernel = void (*)(const uint16_t*, const float*, float*, size_t, size_t);

struct Timing {
    std::string name;
    double ms = 0.0;
    double gbs = 0.0;
};

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

std::vector<float> make_hidden(size_t H, uint32_t seed) {
    std::vector<float> h(H);
    for (size_t i = 0; i < H; ++i) {
        // Deterministic, realistic-magnitude hidden state.
        const uint32_t u = 0x3f000000u + (0x3f000000u * static_cast<uint32_t>(i + seed)) % 0x00800000u;
        h[i] = from_bits(u | 0x80000000u * ((i & 1) == 0 ? 0u : 1u));
    }
    return h;
}

bool compare_to_ref(
    const char* label, const std::vector<float>& ref,
    const std::vector<float>& got
) {
    for (size_t i = 0; i < ref.size(); ++i) {
        if (bits(ref[i]) != bits(got[i])) {
            std::cerr << "MISMATCH " << label << " v=" << i << " ref="
                      << std::hex << bits(ref[i]) << " got=" << bits(got[i])
                      << std::dec << '\n';
            return false;
        }
    }
    return true;
}

void run_engine(
    const uint16_t* W, const float* h, float* out, size_t V, size_t H, bool avx2
) {
    if (avx2) unsetenv("LITTLEBIT_DISABLE_AVX2");
    else setenv("LITTLEBIT_DISABLE_AVX2", "1", 1);
    littlebit::lm_head_logits(W, h, out, V, H);
}

// Small-shape special sweep against the local scalar oracle.
bool special_sweep(const Kernel* kernels, const char* const* names, size_t nk) {
    const uint16_t w_patterns[] = {
        0x0000, 0x8000, 0x007f, 0x807f, 0x0080, 0x8080,
        0x3f80, 0xbf80, 0x7f80, 0xff80, 0x7fa1, 0xffa1,
        0x7fc0, 0xffc0
    };
    const uint32_t h_patterns[] = {
        0x00000000, 0x80000000, 0x00000001, 0x80000001,
        0x007fffff, 0x807fffff, 0x00800000, 0x80800000,
        0x3f800000, 0xbf800000, 0x7f7fffff, 0xff7fffff,
        0x7f800000, 0xff800000,
        0x7fa12345, 0xffa12345, 0x7fc00000, 0xffc00000
    };
    std::vector<size_t> failures(nk, 0);
    std::vector<bool> first_reported(nk, false);
    size_t cases = 0;
    for (size_t V : {size_t(1), size_t(7), size_t(8), size_t(9), size_t(15),
                     size_t(16), size_t(17), size_t(31), size_t(32), size_t(33)}) {
        for (size_t H : {size_t(1), size_t(7), size_t(8), size_t(9), size_t(15),
                         size_t(16), size_t(33), size_t(64)}) {
            std::vector<uint16_t> W(V * H);
            std::vector<float> h(H);
            for (size_t v = 0; v < V; ++v) {
                for (size_t i = 0; i < H; ++i) {
                    W[v * H + i] = w_patterns[(v * 3 + i * 5) % 14];
                }
            }
            for (size_t i = 0; i < H; ++i) {
                h[i] = from_bits(h_patterns[(i * 7) % 18]);
            }
            std::vector<float> ref(V);
            scalar_lm_head(W.data(), h.data(), ref.data(), V, H);
            for (size_t k = 0; k < nk; ++k) {
                std::vector<float> got(V);
                kernels[k](W.data(), h.data(), got.data(), V, H);
                for (size_t v = 0; v < V; ++v) {
                    if (bits(ref[v]) != bits(got[v])) {
                        ++failures[k];
                        if (!first_reported[k]) {
                            first_reported[k] = true;
                            std::cerr << "MISMATCH " << names[k] << " V=" << V
                                      << " H=" << H << " v=" << v << " ref="
                                      << std::hex << bits(ref[v]) << " got="
                                      << bits(got[v]) << std::dec << '\n';
                        }
                        break;
                    }
                }
            }
            ++cases;
        }
    }
    bool all = true;
    for (size_t k = 0; k < nk; ++k) {
        std::cout << "special_sweep " << names[k] << " cases=" << cases
                  << " failed_cases=" << failures[k]
                  << " result=" << (failures[k] == 0 ? "PASS" : "FAIL") << '\n';
        all &= failures[k] == 0;
    }
    return all;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: opt6_lm_head_probe <model_dir> <json_out> [rounds]\n";
        return 2;
    }
    const std::string model_dir = argv[1];
    const std::string json_out = argv[2];
    const size_t rounds = argc > 3 ? static_cast<size_t>(std::stoul(argv[3])) : 6;

    littlebit::SafeTensorsFile st;
    if (!st.open(model_dir + "/model.safetensors")) return 3;
    if (!st.has_tensor("lm_head.weight")) {
        std::cerr << "no lm_head.weight\n";
        return 4;
    }
    const auto& info = st.get_tensor_info("lm_head.weight");
    const uint16_t* W = reinterpret_cast<const uint16_t*>(info.data_ptr);
    const size_t V = static_cast<size_t>(info.shape[0]);
    const size_t H = static_cast<size_t>(info.shape[1]);
    const size_t n = V * H;
    const double bytes = static_cast<double>(n) * 2.0;
    std::cout << "lm_head V=" << V << " H=" << H << " bytes=" << n * 2 << '\n';

    const Kernel kernels[] = {
        avx2_transpose8, avx2_transpose16wide, avx2_transpose16,
        avx2_transpose16_pipe, avx2_independent_rows
    };
    const char* names[] = {
        "transpose8", "transpose16wide", "transpose16", "transpose16_pipe",
        "independent_rows"
    };
    const bool special_ok = special_sweep(kernels, names, 5);

    // Real weights, deterministic hidden vector.
    std::vector<float> h = make_hidden(H, 262);
    std::vector<float> ref(V), engine_avx(V), engine_scalar(V);
    std::vector<std::vector<float>> got(5, std::vector<float>(V));

    run_engine(W, h.data(), engine_avx.data(), V, H, true);
    run_engine(W, h.data(), engine_scalar.data(), V, H, false);
    std::vector<float> local_ref(V);
    scalar_lm_head(W, h.data(), local_ref.data(), V, H);
    const bool engine_matches_scalar = compare_to_ref("engine_avx2_vs_engine_scalar", engine_scalar, engine_avx);
    const bool local_matches_scalar = compare_to_ref("local_oracle_vs_engine_scalar", engine_scalar, local_ref);
    bool candidates_exact = true;
    for (size_t k = 0; k < 5; ++k) {
        kernels[k](W, h.data(), got[k].data(), V, H);
        candidates_exact &= compare_to_ref(names[k], engine_scalar, got[k]);
    }
    std::cout << "real_exact engine_avx2=" << engine_matches_scalar
              << " local_oracle=" << local_matches_scalar
              << " candidates=" << candidates_exact << '\n';
    ref = engine_scalar;

    // Timed kernels: current engine AVX2, scalar, and candidates. The engine is
    // timed through the public entry point so the real production path is used.
    // Interleave rounds to reduce ordering/frequency bias.
    const int ntimed = 9;
    const char* tnames[ntimed] = {
        "engine_avx2", "scalar", "transpose8", "transpose16wide",
        "transpose16", "transpose16_pipe", "independent_rows",
        "memory_only", "convert_only"
    };
    std::vector<std::vector<double>> samples(ntimed);
    std::vector<float> sinkbuf(V);
    volatile float sink = 0.0f;

    auto timed_call = [&](int which) {
        auto a = Clock::now();
        switch (which) {
            case 0: run_engine(W, h.data(), sinkbuf.data(), V, H, true); break;
            case 1: run_engine(W, h.data(), sinkbuf.data(), V, H, false); break;
            case 2: avx2_transpose8(W, h.data(), sinkbuf.data(), V, H); break;
            case 3: avx2_transpose16wide(W, h.data(), sinkbuf.data(), V, H); break;
            case 4: avx2_transpose16(W, h.data(), sinkbuf.data(), V, H); break;
            case 5: avx2_transpose16_pipe(W, h.data(), sinkbuf.data(), V, H); break;
            case 6: avx2_independent_rows(W, h.data(), sinkbuf.data(), V, H); break;
            case 7: sink += diag_memory_only(W, n); break;
            default: sink += diag_convert_only(W, n); break;
        }
        auto b = Clock::now();
        sink += sinkbuf[0];
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    // Warm up every path.
    for (int w = 0; w < 2; ++w) {
        for (int k = 0; k < ntimed; ++k) timed_call(k);
    }
    for (size_t round = 0; round < rounds; ++round) {
        for (int step = 0; step < ntimed; ++step) {
            const int k = static_cast<int>((round + static_cast<size_t>(step)) % ntimed);
            samples[k].push_back(timed_call(k));
        }
    }

    std::vector<Timing> timings;
    for (int k = 0; k < ntimed; ++k) {
        Timing t;
        t.name = tnames[k];
        t.ms = median(samples[k]);
        t.gbs = bytes / (t.ms * 1e6);
        timings.push_back(t);
        std::cout << t.name << " median_ms=" << t.ms << " GB/s=" << t.gbs << '\n';
    }

    // Convert-only and multiply-only diagnostics (not exact comparisons).
    auto diag_bench = [&](const char* name, auto fn) {
        for (int w = 0; w < 2; ++w) sink += fn(W, h.data(), n, H);
        std::vector<double> s;
        for (size_t r = 0; r < rounds; ++r) {
            auto a = Clock::now();
            sink += fn(W, h.data(), n, H);
            auto b = Clock::now();
            s.push_back(std::chrono::duration<double, std::milli>(b - a).count());
        }
        const double ms = median(s);
        std::cout << name << " median_ms=" << ms << " GB/s=" << bytes / (ms * 1e6) << '\n';
        return std::make_pair(ms, bytes / (ms * 1e6));
    };
    auto conv = diag_bench("convert_only", [](const uint16_t* w, const float*, size_t nn, size_t) {
        return diag_convert_only(w, nn);
    });
    auto mul = diag_bench("multiply_only", [](const uint16_t* w, const float* hh, size_t nn, size_t HH) {
        return diag_multiply_only(w, hh, nn, HH);
    });
    (void)mul;

    // JSON artifact.
    FILE* f = std::fopen(json_out.c_str(), "w");
    if (!f) return 6;
    std::fprintf(f, "{\n  \"model_dir\": \"%s\",\n  \"V\": %zu,\n  \"H\": %zu,\n",
                 model_dir.c_str(), V, H);
    std::fprintf(f, "  \"bytes_per_call\": %zu,\n  \"rounds\": %zu,\n", n * 2, rounds);
    std::fprintf(f, "  \"special_sweep\": %s,\n  \"real_exact_engine_avx2\": %s,\n",
                 special_ok ? "true" : "false",
                 engine_matches_scalar ? "true" : "false");
    std::fprintf(f, "  \"real_exact_local_oracle\": %s,\n  \"real_exact_candidates\": %s,\n",
                 local_matches_scalar ? "true" : "false",
                 candidates_exact ? "true" : "false");
    std::fprintf(f, "  \"timings\": [\n");
    for (size_t i = 0; i < timings.size(); ++i) {
        std::fprintf(f,
            "    {\"name\": \"%s\", \"median_ms\": %.6f, \"GB_per_s\": %.6f}%s\n",
            timings[i].name.c_str(), timings[i].ms, timings[i].gbs,
            i + 1 == timings.size() ? "" : ",");
    }
    std::fprintf(f, "  ],\n  \"convert_only_ms\": %.6f,\n  \"multiply_only_ms\": %.6f,\n",
                 conv.first, mul.first);
    std::fprintf(f, "  \"sink\": %f\n}\n", static_cast<double>(sink));
    std::fclose(f);

    return (special_ok && engine_matches_scalar && local_matches_scalar && candidates_exact) ? 0 : 7;
}
