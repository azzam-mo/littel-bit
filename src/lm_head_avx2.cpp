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

// Eight vocabulary rows are accumulated in eight AVX2 lanes. For each 8-element
// hidden group the reduction reproduces GCC's auto-vectorized scalar order
// exactly, including the alternating vaddps operand order that controls NaN
// payload/sign selection.
__attribute__((target("avx2")))
void avx2_row_lanes(
    const uint16_t* W, const float* h, float* out, size_t V, size_t H
) {
    if (H % 8 != 0 || V < 8) {
        scalar_lm_head(W, h, out, V, H);
        return;
    }

    const int Hi = static_cast<int>(H);
    const __m256i lane = _mm256_set_epi32(
        7 * Hi, 6 * Hi, 5 * Hi, 4 * Hi, 3 * Hi, 2 * Hi, 1 * Hi, 0
    );
    const __m256i mask16 = _mm256_set1_epi32(0xffff);
    const int* base = reinterpret_cast<const int*>(W);

    size_t v = 0;
    for (; v + 8 <= V; v += 8) {
        __m256 sum = _mm256_setzero_ps();
        const __m256i vbase = _mm256_set1_epi32(static_cast<int>(v * H));

        // A 32-bit gather from a BF16 address also reads the adjacent BF16.
        // Avoid that read at the final hidden element by loading the last
        // 8-element group directly with in-bounds 16-bit loads.
        for (size_t j = 0; j + 8 < H; j += 8) {
            __m256 P[8];
            for (int t = 0; t < 8; ++t) {
                const __m256i idx = _mm256_add_epi32(
                    _mm256_add_epi32(vbase, lane),
                    _mm256_set1_epi32(static_cast<int>(j + t))
                );
                const __m256i raw = _mm256_i32gather_epi32(base, idx, 2);
                const __m256i bf_bits = _mm256_slli_epi32(
                    _mm256_and_si256(raw, mask16), 16
                );
                const __m256 w = _mm256_castsi256_ps(bf_bits);
                const __m256 hv = _mm256_set1_ps(h[j + t]);
                __m256 p;
                // GCC's packed multiply uses src1=weight, src2=hidden.
                __asm__ volatile("vmulps %2, %1, %0" : "=x"(p) : "x"(w), "x"(hv));
                P[t] = p;
            }

            // Exact compiled-oracle reduction order for one 8-element group.
            sum = add_ordered(P[0], sum); // p0 + dot  (src1=p0, src2=dot)
            sum = add_ordered(P[1], sum); // p1 + acc  (src1=p1, src2=acc)
            sum = add_ordered(sum, P[2]);
            sum = add_ordered(sum, P[3]);
            sum = add_ordered(sum, P[4]);
            sum = add_ordered(sum, P[5]);
            sum = add_ordered(sum, P[6]);
            sum = add_ordered(sum, P[7]);
        }
        {
            const size_t j = H - 8;
            const uint16_t* r0 = W + v * H;
            __m256 P[8];
            for (int t = 0; t < 8; ++t) {
                const __m256i raw = _mm256_setr_epi32(
                    r0[0 * H + j + t], r0[1 * H + j + t],
                    r0[2 * H + j + t], r0[3 * H + j + t],
                    r0[4 * H + j + t], r0[5 * H + j + t],
                    r0[6 * H + j + t], r0[7 * H + j + t]
                );
                const __m256i bf_bits = _mm256_slli_epi32(
                    _mm256_and_si256(raw, mask16), 16
                );
                const __m256 w = _mm256_castsi256_ps(bf_bits);
                const __m256 hv = _mm256_set1_ps(h[j + t]);
                __m256 p;
                __asm__ volatile("vmulps %2, %1, %0" : "=x"(p) : "x"(w), "x"(hv));
                P[t] = p;
            }
            sum = add_ordered(P[0], sum);
            sum = add_ordered(P[1], sum);
            sum = add_ordered(sum, P[2]);
            sum = add_ordered(sum, P[3]);
            sum = add_ordered(sum, P[4]);
            sum = add_ordered(sum, P[5]);
            sum = add_ordered(sum, P[6]);
            sum = add_ordered(sum, P[7]);
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
