#include "littlebit/binary_packer.h"
#include "littlebit/scalar_kernels.h"

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

void neon_rows(
    const int32_t* packed_A, const float* x, float* y, size_t M, size_t K
) {
    const size_t words_per_row = (K + 31) / 32;
    size_t i = 0;

    // Four output rows per float32x4_t. Each lane walks k in the same order as
    // detail::scalar_binary_matvec, so every per-row FP operation is identical.
    for (; i + 4 <= M; i += 4) {
        float32x4_t sum = vdupq_n_f32(0.0f);
        const int32_t* row0 = packed_A + (i + 0) * words_per_row;
        const int32_t* row1 = packed_A + (i + 1) * words_per_row;
        const int32_t* row2 = packed_A + (i + 2) * words_per_row;
        const int32_t* row3 = packed_A + (i + 3) * words_per_row;

        for (size_t w = 0; w < words_per_row; ++w) {
            uint32x4_t words = vdupq_n_u32(0u);
            words = vsetq_lane_u32(static_cast<uint32_t>(row0[w]), words, 0);
            words = vsetq_lane_u32(static_cast<uint32_t>(row1[w]), words, 1);
            words = vsetq_lane_u32(static_cast<uint32_t>(row2[w]), words, 2);
            words = vsetq_lane_u32(static_cast<uint32_t>(row3[w]), words, 3);

            const size_t k0 = w * 32;
            const size_t bits_to_process = (K - k0 >= 32) ? 32 : (K - k0);

            // LSB-first bit order, matching pack_binary_matrix and the oracle:
            // bit 0 = +1 (add), bit 1 = -1 (subtract).
            for (size_t b = 0; b < bits_to_process; ++b) {
                const float32x4_t vx = vdupq_n_f32(x[k0 + b]);
                const float32x4_t add = add_ordered(sum, vx);
                const float32x4_t sub = sub_ordered(sum, vx);
                const uint32x4_t bit_mask = vdupq_n_u32(1u << b);
                const uint32x4_t is_one = vceqq_u32(vandq_u32(words, bit_mask), bit_mask);
                sum = vbslq_f32(is_one, sub, add);
            }
        }
        vst1q_f32(y + i, sum);
    }

    if (i < M) {
        detail::scalar_binary_matvec(
            packed_A + i * words_per_row, x, y + i, M - i, K
        );
    }
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

#else // !(__aarch64__ && __ARM_NEON)

bool neon_available() {
    return false;
}

void binary_matvec_mult_neon(
    const int32_t* packed_A, const float* x, float* y, size_t M, size_t K
) {
    detail::scalar_binary_matvec(packed_A, x, y, M, K);
}

#endif

} // namespace littlebit
