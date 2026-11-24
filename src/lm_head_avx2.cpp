#include "littlebit/binary_packer.h"
#include "littlebit/common.h"
#include "lm_head_avx2.h"

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

namespace littlebit {
namespace {

// Exact copy of the engine's scalar lm_head loop at commit 9605266. It is
// deliberately compiled at the project default -O3 without an AVX2 target so
// the frozen auto-vectorized SSE reduction order stays available as the oracle
// and as the scalar fallback.
__attribute__((noinline))
void scalar_lm_head(
    const uint16_t* W, const float* h, float* out, size_t V, size_t H
) {
    for (size_t v = 0; v < V; ++v) {
        const uint16_t* row = W + v * H;
        float dot = 0.0f;
        for (size_t i = 0; i < H; ++i) {
            dot += h[i] * bf16_to_fp32(row[i]);
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
static inline __m256 bf16_lane_to_float(__m128i bits16) {
    const __m256i w32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(bits16), 16);
    return _mm256_castsi256_ps(w32);
}

// GCC's packed multiply uses src1=weight, src2=hidden. Keeping the operand
// order explicit preserves NaN payload/sign selection.
__attribute__((target("avx2"), always_inline))
static inline __m256 mul_weight_hidden(__m256 w, __m256 hv) {
    __m256 p;
    __asm__ volatile("vmulps %2, %1, %0" : "=x"(p) : "x"(w), "x"(hv));
    return p;
}

// Transpose eight packed uint16 rows. Each input register holds eight
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

// Exact compiled-oracle reduction order for one 8-element hidden group. One
// vocabulary row is accumulated per AVX2 lane, so the sequence of additions is
// identical to the scalar oracle while the gathers are gone.
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

// Eight vocabulary rows are accumulated in eight AVX2 lanes. Each row is read
// with contiguous 16-byte loads and an 8x8 register transpose replaces the
// previous per-element 32-bit gathers. The reduction order is still exactly the
// frozen scalar order, including alternating vaddps operand order.
__attribute__((target("avx2")))
void avx2_row_lanes(
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
            __m128i columns[8];
            for (int r = 0; r < 8; ++r) {
                in[r] = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(rows[r] + j)
                );
            }
            transpose8x8_u16(in, columns);
            sum = step8_from_columns(columns, h, j, sum);
        }
        _mm256_storeu_ps(out + v, sum);
    }

    if (v < V) scalar_lm_head(W + v * H, h, out + v, V - v, H);
}

} // namespace

void lm_head_logits(
    const uint16_t* W, const float* h, float* out, size_t V, size_t H
) {
    if (avx2_available()) {
        avx2_row_lanes(W, h, out, V, H);
        return;
    }
    scalar_lm_head(W, h, out, V, H);
}

} // namespace littlebit
