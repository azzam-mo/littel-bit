#pragma once

#include <cstddef>
#include <cstdint>

namespace littlebit {
namespace detail {

// Frozen scalar binary matvec oracle from commit 87c2048. All architecture
// kernels must match this function bit-for-bit, including NaN payload handling.
void scalar_binary_matvec(
    const int32_t* packed_A, const float* x, float* y, size_t M, size_t K
);

} // namespace detail
} // namespace littlebit
