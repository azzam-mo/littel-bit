#pragma once

#include <cstddef>
#include <cstdint>

namespace littlebit {
namespace detail {

// Ordered scalar FP operations used by the frozen oracle and every scalar tail.
//
// On AArch64, `sum -= val` may be lowered by the compiler as `fneg val` followed
// by `fadd sum, -val`. That is valid C++ but changes which NaN sign/payload is
// propagated: AArch64 FPProcessNaNs selects operand A for two-input add/sub, so
// negating `val` before the add flips a NaN produced by `val`. The NEON kernel
// emits ordered vector `fsub sum, val`; force the scalar side to the same ISA
// operation so the oracle, scalar tails, and NEON lanes agree bit-for-bit.
#if defined(__aarch64__) && defined(__ARM_NEON)
inline float ordered_add(float a, float b) {
    float result;
    __asm__ volatile("fadd %s0, %s1, %s2"
                     : "=&w"(result)
                     : "w"(a), "w"(b));
    return result;
}

inline float ordered_sub(float a, float b) {
    float result;
    __asm__ volatile("fsub %s0, %s1, %s2"
                     : "=&w"(result)
                     : "w"(a), "w"(b));
    return result;
}
#else
inline float ordered_add(float a, float b) { return a + b; }
inline float ordered_sub(float a, float b) { return a - b; }
#endif

// Frozen scalar binary matvec oracle from commit 87c2048. All architecture
// kernels must match this function bit-for-bit, including NaN payload handling.
void scalar_binary_matvec(
    const int32_t* packed_A, const float* x, float* y, size_t M, size_t K
);

} // namespace detail
} // namespace littlebit
