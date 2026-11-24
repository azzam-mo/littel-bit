#pragma once

#include "littlebit/common.h"
#include <vector>
#include <cstdint>

namespace littlebit {

// Packs a matrix of {-1, +1} values (stored as int8_t) into row-major 32-bit words (lsb_first).
// Matches SamsungLabs/LittleBit binary_packer:
// +1 -> bit 0
// -1 -> bit 1
std::vector<int32_t> pack_binary_matrix(const int8_t* data, size_t n_rows, size_t n_cols);

// Unpacks 32-bit packed words back into int8_t {-1, +1} matrix of shape (n_rows, n_cols).
std::vector<int8_t> unpack_binary_matrix(const int32_t* packed, size_t n_rows, size_t n_cols);

// High-performance Direct Binary Matrix-Vector Multiplication:
// Computes y = A * x, where:
// - A is (M x K) binary matrix stored in packed 32-bit format (M rows, words_per_row = ceil(K/32))
// - x is (K) floating-point vector (FP32)
// - y is (M) output floating-point vector (FP32)
// Since A elements are {-1, +1}, for each bit b:
// if bit == 0 (representing +1): y_i += x_j
// if bit == 1 (representing -1): y_i -= x_j
//
// ZERO unpacking memory allocated!
void binary_matvec_mult(
    const int32_t* packed_A,
    const float* x,
    float* y,
    size_t M,
    size_t K
);

// Separate AVX2 kernel for Phase 9 Optimization 2. It processes eight output
// rows in parallel without changing the per-row accumulation order. Runtime
// detection and a bit-exact scalar fallback are built in.
// Set LITTLEBIT_DISABLE_AVX2=1 to force the scalar fallback.
void binary_matvec_mult_avx2(
    const int32_t* packed_A,
    const float* x,
    float* y,
    size_t M,
    size_t K
);

// True only when AVX2 is supported by the CPU and not disabled at runtime.
bool avx2_available();

// ARM64 NEON kernel for binary matvec. Declared on all platforms so test code
// can be shared; always falls back to the frozen scalar kernel when unavailable.
bool neon_available();
void binary_matvec_mult_neon(
    const int32_t* packed_A,
    const float* x,
    float* y,
    size_t M,
    size_t K
);

// High-performance Transposed Binary Matrix-Vector Multiplication:
// Computes y = A^T * x, where:
// - A is (M x K) packed matrix
// - x is (M) vector
// - y is (K) vector
void binary_matvec_mult_transposed(
    const int32_t* packed_A,
    const float* x,
    float* y,
    size_t M,
    size_t K
);

} // namespace littlebit
