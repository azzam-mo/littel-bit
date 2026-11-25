#include "littlebit/scalar_kernels.h"

namespace littlebit {
namespace detail {

// Exact copy of the branchy scalar kernel at commit 87c2048. This is the frozen
// oracle used by the x86 AVX2 fallback, the ARM64 NEON tail, and raw-bit tests.
void scalar_binary_matvec(
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
                    sum = ordered_add(sum, val); // bit 0 = +1
                } else {
                    sum = ordered_sub(sum, val); // bit 1 = -1
                }
            }
        }
        y[i] = sum;
    }
}

} // namespace detail
} // namespace littlebit
