#include "littlebit/binary_packer.h"
#include "littlebit/binary_packer_neon_variants_internal.h"
#include "littlebit/scalar_kernels.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace littlebit {
namespace {

#if defined(__aarch64__) && defined(__ARM_NEON)

// Force src1=sum, src2=vx exactly as the frozen scalar `sum += val` and
// `sum -= val` operations. Intrinsics alone may be commuted by the compiler,
// which changes which NaN payload is propagated.
inline float32x4_t add_ordered(float32x4_t sum, float32x4_t vx) {
    float32x4_t result;
    __asm__ volatile("fadd %0.4s, %1.4s, %2.4s" : "=w"(result) : "w"(sum), "w"(vx));
    return result;
}

inline float32x4_t sub_ordered(float32x4_t sum, float32x4_t vx) {
    float32x4_t result;
    __asm__ volatile("fsub %0.4s, %1.4s, %2.4s" : "=w"(result) : "w"(sum), "w"(vx));
    return result;
}

// Build one 32-bit-lane vector from four row words. The lane order is the
// output-row order, matching vsetq_lane_* in the original NEON4 kernel.
inline uint32x4_t load_four_words(
    const int32_t* a, const int32_t* b, const int32_t* c, const int32_t* d
) {
    const uint64_t lo =
        static_cast<uint64_t>(static_cast<uint32_t>(*a)) |
        (static_cast<uint64_t>(static_cast<uint32_t>(*b)) << 32);
    const uint64_t hi =
        static_cast<uint64_t>(static_cast<uint32_t>(*c)) |
        (static_cast<uint64_t>(static_cast<uint32_t>(*d)) << 32);
    return vcombine_u32(vcreate_u32(lo), vcreate_u32(hi));
}

// Processes only complete groups of 4*G rows. Tail handling lives in
// neon_rows_group so every variant uses the same 32 -> 8 -> 4 -> scalar cascade.
template <int G, bool UseVtst, bool Prefetch>
size_t neon_rows_full(
    const int32_t* packed_A, const float* x, float* y, size_t M, size_t K
);

template <int G, bool UseVtst, bool Prefetch>
void neon_rows_group(
    const int32_t* packed_A, const float* x, float* y, size_t M, size_t K
);

template <int G, bool UseVtst, bool Prefetch>
size_t neon_rows_full(
    const int32_t* packed_A, const float* x, float* y, size_t M, size_t K
) {
    static_assert(G >= 1 && G <= 8, "NEON group count out of range");

    const size_t words_per_row = (K + 31) / 32;
    size_t i = 0;

    // G independent float32x4_t accumulators. Each lane still walks k in the
    // same k/b order as detail::scalar_binary_matvec, so per-row FP operations
    // (including NaN payload selection) are unchanged.
    for (; i + 4 * G <= M; i += 4 * G) {
        float32x4_t sums[G];
        const int32_t* rows[4 * G];

        for (int g = 0; g < G; ++g) {
            sums[g] = vdupq_n_f32(0.0f);
            for (int lane = 0; lane < 4; ++lane) {
                rows[g * 4 + lane] =
                    packed_A + (i + static_cast<size_t>(g * 4 + lane)) * words_per_row;
            }
        }

        for (size_t w = 0; w < words_per_row; ++w) {
            if constexpr (Prefetch) {
                const size_t ahead = w + 8;
                if (ahead < words_per_row) {
                    for (int r = 0; r < 4 * G; ++r) {
                        __builtin_prefetch(rows[r] + ahead, 0, 1);
                    }
                }
            }

            uint32x4_t words_vec[G];
            for (int g = 0; g < G; ++g) {
                const int base = g * 4;
                words_vec[g] = load_four_words(
                    rows[base + 0] + w,
                    rows[base + 1] + w,
                    rows[base + 2] + w,
                    rows[base + 3] + w
                );
            }

            const size_t k0 = w * 32;
            const size_t bits_to_process = (K - k0 >= 32) ? 32 : (K - k0);

            // LSB-first bit order, matching pack_binary_matrix and the oracle:
            // bit 0 = +1 (add), bit 1 = -1 (subtract).
            for (size_t b = 0; b < bits_to_process; ++b) {
                const float32x4_t vx = vdupq_n_f32(x[k0 + b]);
                const uint32x4_t bit_mask = vdupq_n_u32(1u << b);

                for (int g = 0; g < G; ++g) {
                    const float32x4_t add = add_ordered(sums[g], vx);
                    const float32x4_t sub = sub_ordered(sums[g], vx);
                    const uint32x4_t is_one = UseVtst
                        ? vtstq_u32(words_vec[g], bit_mask)
                        : vceqq_u32(vandq_u32(words_vec[g], bit_mask), bit_mask);
                    sums[g] = vbslq_f32(is_one, sub, add);
                }
            }
        }

        for (int g = 0; g < G; ++g) {
            vst1q_f32(y + i + static_cast<size_t>(g * 4), sums[g]);
        }
    }

    return i;
}

template <int G, bool UseVtst, bool Prefetch>
void neon_rows_group(
    const int32_t* packed_A, const float* x, float* y, size_t M, size_t K
) {
    const size_t words_per_row = (K + 31) / 32;
    size_t i = 0;

    if constexpr (G == 8) {
        i += neon_rows_full<8, UseVtst, Prefetch>(packed_A, x, y, M, K);
        i += neon_rows_full<2, UseVtst, Prefetch>(
            packed_A + i * words_per_row, x, y + i, M - i, K
        );
        i += neon_rows_full<1, UseVtst, Prefetch>(
            packed_A + i * words_per_row, x, y + i, M - i, K
        );
    } else if constexpr (G == 4) {
        i += neon_rows_full<4, UseVtst, Prefetch>(packed_A, x, y, M, K);
        i += neon_rows_full<2, UseVtst, Prefetch>(
            packed_A + i * words_per_row, x, y + i, M - i, K
        );
        i += neon_rows_full<1, UseVtst, Prefetch>(
            packed_A + i * words_per_row, x, y + i, M - i, K
        );
    } else if constexpr (G == 2) {
        i += neon_rows_full<2, UseVtst, Prefetch>(packed_A, x, y, M, K);
        i += neon_rows_full<1, UseVtst, Prefetch>(
            packed_A + i * words_per_row, x, y + i, M - i, K
        );
    } else {
        i += neon_rows_full<1, UseVtst, Prefetch>(packed_A, x, y, M, K);
    }

    if (i < M) {
        detail::scalar_binary_matvec(
            packed_A + i * words_per_row, x, y + i, M - i, K
        );
    }
}

// Current production engine shape: 4 rows / one accumulator, and+ceq sign test.
inline void neon_rows(
    const int32_t* packed_A, const float* x, float* y, size_t M, size_t K
) {
    neon_rows_group<1, false, false>(packed_A, x, y, M, K);
}

#endif // __aarch64__ && __ARM_NEON

} // namespace

#if defined(__aarch64__) && defined(__ARM_NEON)

bool neon_available() {
    static const bool supported = true; // NEON is baseline on arm64.
    const char* env = std::getenv("LITTLEBIT_DISABLE_NEON");
    const bool disabled = env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
    return supported && !disabled;
}

void binary_matvec_mult_neon(
    const int32_t* packed_A, const float* x, float* y, size_t M, size_t K
) {
    if (!neon_available()) {
        detail::scalar_binary_matvec(packed_A, x, y, M, K);
        return;
    }
    neon_rows(packed_A, x, y, M, K);
}

namespace bench {

void neon_variant_4(
    const int32_t* packed_A, const float* x, float* y, size_t M, size_t K
) {
    neon_rows_group<1, false, false>(packed_A, x, y, M, K);
}

void neon_variant_8(
    const int32_t* packed_A, const float* x, float* y, size_t M, size_t K
) {
    neon_rows_group<2, false, false>(packed_A, x, y, M, K);
}

void neon_variant_16(
    const int32_t* packed_A, const float* x, float* y, size_t M, size_t K
) {
    neon_rows_group<4, false, false>(packed_A, x, y, M, K);
}

void neon_variant_32(
    const int32_t* packed_A, const float* x, float* y, size_t M, size_t K
) {
    neon_rows_group<8, false, false>(packed_A, x, y, M, K);
}

void neon_variant_vtst(
    const int32_t* packed_A, const float* x, float* y, size_t M, size_t K
) {
    neon_rows_group<1, true, false>(packed_A, x, y, M, K);
}

void neon_variant_prefetch(
    const int32_t* packed_A, const float* x, float* y, size_t M, size_t K
) {
    neon_rows_group<8, false, true>(packed_A, x, y, M, K);
}

} // namespace bench

#else // !(__aarch64__ && __ARM_NEON)

bool neon_available() {
    return false;
}

void binary_matvec_mult_neon(
    const int32_t* packed_A, const float* x, float* y, size_t M, size_t K
) {
    detail::scalar_binary_matvec(packed_A, x, y, M, K);
}

namespace bench {

void neon_variant_4(
    const int32_t* packed_A, const float* x, float* y, size_t M, size_t K
) {
    detail::scalar_binary_matvec(packed_A, x, y, M, K);
}

void neon_variant_8(
    const int32_t* packed_A, const float* x, float* y, size_t M, size_t K
) {
    detail::scalar_binary_matvec(packed_A, x, y, M, K);
}

void neon_variant_16(
    const int32_t* packed_A, const float* x, float* y, size_t M, size_t K
) {
    detail::scalar_binary_matvec(packed_A, x, y, M, K);
}

void neon_variant_32(
    const int32_t* packed_A, const float* x, float* y, size_t M, size_t K
) {
    detail::scalar_binary_matvec(packed_A, x, y, M, K);
}

void neon_variant_vtst(
    const int32_t* packed_A, const float* x, float* y, size_t M, size_t K
) {
    detail::scalar_binary_matvec(packed_A, x, y, M, K);
}

void neon_variant_prefetch(
    const int32_t* packed_A, const float* x, float* y, size_t M, size_t K
) {
    detail::scalar_binary_matvec(packed_A, x, y, M, K);
}

} // namespace bench

#endif

} // namespace littlebit
