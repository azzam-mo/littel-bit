#include "littlebit/binary_packer.h"

#include <immintrin.h>

#include <cstdlib>
#include <cstring>

namespace littlebit {
namespace {

// Exact copy of the branchy scalar kernel at commit 87c2048. Used when AVX2 is
// unavailable or explicitly disabled, so this file never calls back into the
// default engine (which may later dispatch to the AVX2 kernel).
void scalar_fallback(
    const int32_t* packed_A, const float* x, float* y, size_t M, size_t K
) {
    size_t words_per_row = (K + 31) / 32;

    for (size_t i = 0; i < M; ++i) {
        float sum = 0.0f;
        const int32_t* row_words = packed_A + i * words_per_row;

        size_t k = 0;
        for (size_t w = 0; w < words_per_row; ++w) {
            uint32_t word = static_cast<uint32_t>(row_words[w]);
            size_t bits_to_process = (K - k >= 32) ? 32 : (K - k);

            for (size_t b = 0; b < bits_to_process; ++b) {
                float val = x[k++];
                if ((word & (1u << b)) == 0u) {
                    sum += val; // bit 0 = +1
                } else {
                    sum -= val; // bit 1 = -1
                }
            }
        }
        y[i] = sum;
    }
}

bool cpu_supports_avx2() {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") != 0;
}

// Eight output rows are accumulated in eight independent AVX2 lanes. Each lane
// walks K in exactly the same k/b order as the scalar kernel, so per-row FP
// operation order (and NaN/Inf/signed-zero behaviour) is unchanged.
__attribute__((target("avx2")))
void avx2_rows(
    const int32_t* packed_A, const float* x, float* y, size_t M, size_t K
) {
    size_t words_per_row = (K + 31) / 32;
    size_t i = 0;
    const __m256i all_ones = _mm256_set1_epi32(-1);
    const __m256i zero = _mm256_setzero_si256();

    for (; i + 8 <= M; i += 8) {
        __m256 sum = _mm256_setzero_ps();
        const int32_t* base = packed_A + i * words_per_row;
        const int w_int = static_cast<int>(words_per_row);

        for (size_t w = 0; w < words_per_row; ++w) {
            // Gather the same word index from eight consecutive rows.
            __m256i offsets = _mm256_set_epi32(
                7 * w_int, 6 * w_int, 5 * w_int, 4 * w_int,
                3 * w_int, 2 * w_int, 1 * w_int, 0
            );
            __m256i words = _mm256_i32gather_epi32(
                reinterpret_cast<const int*>(base + w), offsets, 4
            );

            size_t k0 = w * 32;
            size_t bits_to_process = (K - k0 >= 32) ? 32 : (K - k0);

            for (size_t b = 0; b < bits_to_process; ++b) {
                __m256 vx = _mm256_set1_ps(x[k0 + b]);

                // Force src1=sum, src2=vx, exactly as the scalar `sum += val`
                // and `sum -= val` instructions. The compiler is free to swap
                // operands for the commutative add, which changes which NaN is
                // propagated when both operands are NaNs.
                __m256 add, sub;
                __asm__ volatile("vaddps %2, %1, %0" : "=x"(add) : "x"(sum), "x"(vx));
                __asm__ volatile("vsubps %2, %1, %0" : "=x"(sub) : "x"(sum), "x"(vx));

                __m256i bit = _mm256_and_si256(words, _mm256_set1_epi32(1 << b));
                __m256i is_one = _mm256_xor_si256(_mm256_cmpeq_epi32(bit, zero), all_ones);
                sum = _mm256_blendv_ps(add, sub, _mm256_castsi256_ps(is_one));
            }
        }
        _mm256_storeu_ps(y + i, sum);
    }

    // Rows that do not fill a complete AVX2 vector use the exact scalar order.
    for (; i < M; ++i) {
        float sum = 0.0f;
        const int32_t* row_words = packed_A + i * words_per_row;
        size_t k = 0;
        for (size_t w = 0; w < words_per_row; ++w) {
            uint32_t word = static_cast<uint32_t>(row_words[w]);
            size_t bits_to_process = (K - k >= 32) ? 32 : (K - k);
            for (size_t b = 0; b < bits_to_process; ++b) {
                float val = x[k++];
                if ((word & (1u << b)) == 0u) {
                    sum += val;
                } else {
                    sum -= val;
                }
            }
        }
        y[i] = sum;
    }
}

} // namespace

bool avx2_available() {
    static const bool supported = cpu_supports_avx2();
    const char* env = std::getenv("LITTLEBIT_DISABLE_AVX2");
    const bool disabled = env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
    return supported && !disabled;
}

void binary_matvec_mult_avx2(
    const int32_t* packed_A, const float* x, float* y, size_t M, size_t K
) {
    if (!avx2_available()) {
        scalar_fallback(packed_A, x, y, M, K);
        return;
    }
    avx2_rows(packed_A, x, y, M, K);
}

} // namespace littlebit
