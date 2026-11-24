#include "littlebit/binary_packer.h"
#include "littlebit/scalar_kernels.h"

#include <immintrin.h>

#include <cstdlib>
#include <cstring>

namespace littlebit {
namespace {

// Uses the shared frozen scalar kernel when AVX2 is unavailable or explicitly
// disabled, so this file never calls back into the default engine (which may
// dispatch to AVX2 or NEON).
void scalar_fallback(
    const int32_t* packed_A, const float* x, float* y, size_t M, size_t K
) {
    detail::scalar_binary_matvec(packed_A, x, y, M, K);
}

bool cpu_supports_avx2() {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") != 0;
}

template <int G>
__attribute__((target("avx2")))
void avx2_rows_group(
    const int32_t* packed_A, const float* x, float* y, size_t M, size_t K
);

// G groups of eight rows are accumulated in G independent AVX2 vectors. Each
// lane still walks K in exactly the same k/b order as the scalar kernel, so
// per-row FP operation order (and NaN/Inf/signed-zero behaviour) is unchanged.
// Four groups (32 rows) give the CPU four independent accumulation chains while
// keeping all live state in AVX2 registers.
template <int G>
__attribute__((target("avx2")))
void avx2_rows_group(
    const int32_t* packed_A, const float* x, float* y, size_t M, size_t K
) {
    constexpr size_t W = static_cast<size_t>(G) * 8;
    size_t words_per_row = (K + 31) / 32;
    size_t i = 0;
    const __m256i all_ones = _mm256_set1_epi32(-1);
    const __m256i zero = _mm256_setzero_si256();
    const int w_int = static_cast<int>(words_per_row);
    const __m256i offsets = _mm256_set_epi32(
        7 * w_int, 6 * w_int, 5 * w_int, 4 * w_int,
        3 * w_int, 2 * w_int, 1 * w_int, 0
    );

    for (; i + W <= M; i += W) {
        __m256 sums[G];
        for (int g = 0; g < G; ++g) sums[g] = _mm256_setzero_ps();
        const int32_t* bases[G];
        for (int g = 0; g < G; ++g) {
            bases[g] = packed_A + (i + g * 8) * words_per_row;
        }

        for (size_t w = 0; w < words_per_row; ++w) {
            __m256i words[G];
            for (int g = 0; g < G; ++g) {
                words[g] = _mm256_i32gather_epi32(
                    reinterpret_cast<const int*>(bases[g] + w), offsets, 4
                );
            }

            size_t k0 = w * 32;
            size_t bits_to_process = (K - k0 >= 32) ? 32 : (K - k0);

            for (size_t b = 0; b < bits_to_process; ++b) {
                __m256 vx = _mm256_set1_ps(x[k0 + b]);
                __m256i bitmask = _mm256_set1_epi32(static_cast<int>(1u << b));

                for (int g = 0; g < G; ++g) {
                    // Force src1=sum, src2=vx, exactly as the scalar
                    // `sum += val` and `sum -= val` instructions. The compiler
                    // is free to swap operands for the commutative add, which
                    // changes which NaN is propagated when both operands are
                    // NaNs.
                    __m256 add, sub;
                    __asm__ volatile("vaddps %2, %1, %0" : "=x"(add) : "x"(sums[g]), "x"(vx));
                    __asm__ volatile("vsubps %2, %1, %0" : "=x"(sub) : "x"(sums[g]), "x"(vx));

                    __m256i bit = _mm256_and_si256(words[g], bitmask);
                    __m256i is_one = _mm256_xor_si256(
                        _mm256_cmpeq_epi32(bit, zero), all_ones
                    );
                    sums[g] = _mm256_blendv_ps(add, sub, _mm256_castsi256_ps(is_one));
                }
            }
        }
        for (int g = 0; g < G; ++g) {
            _mm256_storeu_ps(y + i + g * 8, sums[g]);
        }
    }

    if (i < M) {
        if constexpr (G > 1) {
            // Keep the 8-row AVX2 fallback for incomplete groups; scalar is
            // reserved for the final 0..7 rows.
            avx2_rows_group<1>(packed_A + i * words_per_row, x, y + i, M - i, K);
        } else {
            scalar_fallback(packed_A + i * words_per_row, x, y + i, M - i, K);
        }
    }
}

__attribute__((target("avx2")))
void avx2_rows(
    const int32_t* packed_A, const float* x, float* y, size_t M, size_t K
) {
    avx2_rows_group<4>(packed_A, x, y, M, K);
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
