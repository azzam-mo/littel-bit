#include "littlebit/binary_packer.h"
#include "littlebit/scalar_kernels.h"
#include <cstring>

namespace littlebit {

std::vector<int32_t> pack_binary_matrix(const int8_t* data, size_t n_rows, size_t n_cols) {
    size_t words_per_row = (n_cols + 31) / 32;
    std::vector<int32_t> packed(n_rows * words_per_row, 0);

    for (size_t r = 0; r < n_rows; ++r) {
        for (size_t c = 0; c < n_cols; ++c) {
            int8_t val = data[r * n_cols + c];
            uint32_t bit = (val == -1) ? 1u : 0u; // +1 -> 0, -1 -> 1
            size_t word_idx = c / 32;
            size_t bit_idx = c % 32;
            packed[r * words_per_row + word_idx] |= (bit << bit_idx);
        }
    }

    return packed;
}

std::vector<int8_t> unpack_binary_matrix(const int32_t* packed, size_t n_rows, size_t n_cols) {
    size_t words_per_row = (n_cols + 31) / 32;
    std::vector<int8_t> unpacked(n_rows * n_cols);

    for (size_t r = 0; r < n_rows; ++r) {
        for (size_t c = 0; c < n_cols; ++c) {
            size_t word_idx = c / 32;
            size_t bit_idx = c % 32;
            uint32_t word = static_cast<uint32_t>(packed[r * words_per_row + word_idx]);
            uint32_t bit = (word >> bit_idx) & 1u;
            unpacked[r * n_cols + c] = (bit == 0u) ? static_cast<int8_t>(1) : static_cast<int8_t>(-1);
        }
    }

    return unpacked;
}

void binary_matvec_mult(
    const int32_t* packed_A,
    const float* x,
    float* y,
    size_t M,
    size_t K
) {
#if defined(__aarch64__) && defined(__ARM_NEON)
    if (littlebit::neon_available()) {
        littlebit::binary_matvec_mult_neon(packed_A, x, y, M, K);
        return;
    }
#endif
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64)
    if (littlebit::avx2_available()) {
        littlebit::binary_matvec_mult_avx2(packed_A, x, y, M, K);
        return;
    }
#endif
    littlebit::detail::scalar_binary_matvec(packed_A, x, y, M, K);
}

void binary_matvec_mult_transposed(
    const int32_t* packed_A,
    const float* x,
    float* y,
    size_t M,
    size_t K
) {
    // y is length K, initialize with 0
    std::memset(y, 0, K * sizeof(float));
    size_t words_per_row = (K + 31) / 32;

    for (size_t i = 0; i < M; ++i) {
        float x_i = x[i];
        if (x_i == 0.0f) continue;

        const int32_t* row_words = packed_A + i * words_per_row;
        size_t k = 0;
        for (size_t w = 0; w < words_per_row; ++w) {
            uint32_t word = static_cast<uint32_t>(row_words[w]);
            size_t bits_to_process = (K - k >= 32) ? 32 : (K - k);

            for (size_t b = 0; b < bits_to_process; ++b) {
                if ((word & (1u << b)) == 0u) {
                    y[k++] += x_i;
                } else {
                    y[k++] -= x_i;
                }
            }
        }
    }
}

} // namespace littlebit
