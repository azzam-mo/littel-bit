#pragma once

// Internal benchmark/test-only entry points for experimental ARM64 NEON
// binary-matvec kernels. These are intentionally not used by the engine dispatch;
// binary_matvec_mult() continues to call the current NEON4 kernel.

#include "littlebit/common.h"

namespace littlebit {
namespace bench {

// Current production kernel shape: 4 rows per float32x4_t accumulator.
void neon_variant_4(
    const int32_t* packed_A, const float* x, float* y, size_t M, size_t K
);

// Wider groups: 8, 16 and 32 rows using 2, 4 and 8 float32x4_t accumulators.
void neon_variant_8(
    const int32_t* packed_A, const float* x, float* y, size_t M, size_t K
);
void neon_variant_16(
    const int32_t* packed_A, const float* x, float* y, size_t M, size_t K
);
void neon_variant_32(
    const int32_t* packed_A, const float* x, float* y, size_t M, size_t K
);

// Cheaper sign test (vtstq_u32) on the current 4-row kernel shape.
void neon_variant_vtst(
    const int32_t* packed_A, const float* x, float* y, size_t M, size_t K
);

// Prefetching on top of the widest (32-row) group kernel.
void neon_variant_prefetch(
    const int32_t* packed_A, const float* x, float* y, size_t M, size_t K
);

} // namespace bench
} // namespace littlebit
