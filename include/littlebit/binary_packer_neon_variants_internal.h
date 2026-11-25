#pragma once

// Internal benchmark/test-only entry points for experimental ARM64 NEON
// binary-matvec kernels. These are intentionally not used by the engine dispatch;
// binary_matvec_mult() continues to call the current NEON4 kernel.

#include "littlebit/common.h"

#include <cstddef>
#include <cstdint>

namespace littlebit {
namespace bench {

// Diagnostic-only lane trace for the 4-row production kernel shape. The
// callback receives freshly computed float bit patterns for lanes 0..3 after
// each packed bit is applied. It exists so raw-bit regressions can name the
// first divergent intermediate without instrumenting production code.
using NeonLaneTrace = void (*)(
    size_t bit_index, size_t lane_count, const uint32_t* lane_bits, void* context
);

void neon_variant_4_trace(
    const int32_t* packed_A, const float* x, float* y, size_t M, size_t K,
    NeonLaneTrace trace, void* context
);

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
