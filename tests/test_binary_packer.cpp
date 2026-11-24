#include "littlebit/binary_packer.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <random>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <limits>

// Reference implementation frozen at commit 87c2048 (branchy scalar).
// Used only to prove bit-exact parity of the optimized kernel.
static void reference_binary_matvec_mult(
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
                    sum += val;
                } else {
                    sum -= val;
                }
            }
        }
        y[i] = sum;
    }
}

static bool same_float_bits(float a, float b) {
    uint32_t ua, ub;
    std::memcpy(&ua, &a, sizeof(ua));
    std::memcpy(&ub, &b, sizeof(ub));
    return ua == ub;
}

static float avx2_test_float_from_bits(uint32_t u) {
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// Candidate A: flip the IEEE-754 sign bit. Addition order is identical to the
// frozen reference; every sum += (+/-val) uses the same k and b sequence.
static void candidate_xor_sign(
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
                uint32_t bits;
                std::memcpy(&bits, &val, sizeof(bits));
                bits ^= ((word >> b) & 1u) << 31;
                std::memcpy(&val, &bits, sizeof(val));
                sum += val;
            }
        }
        y[i] = sum;
    }
}

// Candidate B: multiply by exact +/-1.0f. This is faster on finite weights but
// is intentionally rejected if it is not bit-exact for special float values.
static void candidate_mul_sign(
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
                float sign = 1.0f - 2.0f * static_cast<float>((word >> b) & 1u);
                sum += x[k++] * sign;
            }
        }
        y[i] = sum;
    }
}

using MatvecFn = void (*)(const int32_t*, const float*, float*, size_t, size_t);

static bool candidate_matches_reference(
    size_t M, size_t K, int pattern, bool dirty_padding, MatvecFn fn
) {
    std::vector<int8_t> matrix(M * K);
    for (size_t i = 0; i < M * K; ++i) {
        switch (pattern) {
            case 0: matrix[i] = 1; break;
            case 1: matrix[i] = -1; break;
            case 2: matrix[i] = (i % 2 == 0) ? 1 : -1; break;
            default: matrix[i] = ((i * 2654435761u) >> 31) ? -1 : 1; break;
        }
    }
    auto packed = littlebit::pack_binary_matrix(matrix.data(), M, K);

    if (dirty_padding && K % 32 != 0 && M > 0) {
        size_t words_per_row = (K + 31) / 32;
        uint32_t pad_mask = ~0u << (K % 32);
        for (size_t r = 0; r < M; ++r) {
            packed[r * words_per_row + words_per_row - 1] |= static_cast<int32_t>(pad_mask);
        }
    }

    const float specials[] = {
        0.0f, -0.0f, 1.0f, -1.0f,
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
        -std::numeric_limits<float>::quiet_NaN(),
        avx2_test_float_from_bits(0x7fa12345u),
        avx2_test_float_from_bits(0xffa12345u),
        std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max(),
        std::numeric_limits<float>::min(),
        std::numeric_limits<float>::denorm_min(),
        3.14159265f, -2.71828183f
    };
    std::vector<float> x(K);
    for (size_t i = 0; i < K; ++i) x[i] = specials[i % (sizeof(specials) / sizeof(specials[0]))];

    std::vector<float> y_ref(M, 0.0f), y_cand(M, 0.0f);
    reference_binary_matvec_mult(packed.data(), x.data(), y_ref.data(), M, K);
    fn(packed.data(), x.data(), y_cand.data(), M, K);
    for (size_t i = 0; i < M; ++i) {
        if (!same_float_bits(y_ref[i], y_cand[i])) return false;
    }
    return true;
}

static void check_matvec_case(size_t M, size_t K, int pattern, bool dirty_padding) {
    std::vector<int8_t> matrix(M * K);
    for (size_t i = 0; i < M * K; ++i) {
        switch (pattern) {
            case 0: matrix[i] = 1; break;
            case 1: matrix[i] = -1; break;
            case 2: matrix[i] = (i % 2 == 0) ? 1 : -1; break;
            default: matrix[i] = ((i * 2654435761u) >> 31) ? -1 : 1; break;
        }
    }
    auto packed = littlebit::pack_binary_matrix(matrix.data(), M, K);

    if (dirty_padding && K % 32 != 0 && M > 0) {
        size_t words_per_row = (K + 31) / 32;
        size_t used_bits = K % 32;
        uint32_t pad_mask = (used_bits == 0) ? 0u : (~0u << used_bits);
        for (size_t r = 0; r < M; ++r) {
            packed[r * words_per_row + words_per_row - 1] |= static_cast<int32_t>(pad_mask);
        }
    }

    std::vector<float> x(K);
    const float specials[] = {
        0.0f, -0.0f, 1.0f, -1.0f,
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
        -std::numeric_limits<float>::quiet_NaN(),
        avx2_test_float_from_bits(0x7fa12345u),
        avx2_test_float_from_bits(0xffa12345u),
        std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max(),
        std::numeric_limits<float>::min(),
        std::numeric_limits<float>::denorm_min(),
        3.14159265f, -2.71828183f
    };
    for (size_t i = 0; i < K; ++i) {
        x[i] = specials[i % (sizeof(specials) / sizeof(specials[0]))];
    }

    std::vector<float> y_ref(M, 0.0f), y_opt(M, 0.0f);
    reference_binary_matvec_mult(packed.data(), x.data(), y_ref.data(), M, K);
    littlebit::binary_matvec_mult(packed.data(), x.data(), y_opt.data(), M, K);
    for (size_t i = 0; i < M; ++i) {
        if (!same_float_bits(y_ref[i], y_opt[i])) {
            std::cerr << "MISMATCH M=" << M << " K=" << K << " pattern=" << pattern
                      << " dirty_padding=" << dirty_padding << " row=" << i
                      << " ref_bits vs opt_bits differ\n";
            std::abort();
        }
    }
}

int main() {
    std::cout << "[TEST] Running binary_packer unit tests..." << std::endl;

    // Test 1: Simple 1D row packing
    {
        std::vector<int8_t> data = {1, -1, 1, 1, -1, -1, 1, -1}; // 8 elements
        auto packed = littlebit::pack_binary_matrix(data.data(), 1, 8);
        assert(packed.size() == 1);

        // Expected bits:
        // index 0: +1 -> 0
        // index 1: -1 -> 1
        // index 2: +1 -> 0
        // index 3: +1 -> 0
        // index 4: -1 -> 1
        // index 5: -1 -> 1
        // index 6: +1 -> 0
        // index 7: -1 -> 1
        // Bits: 0b10110010 = 0xB2 = 178
        uint32_t expected = (1 << 1) | (1 << 4) | (1 << 5) | (1 << 7);
        assert(static_cast<uint32_t>(packed[0]) == expected);

        auto unpacked = littlebit::unpack_binary_matrix(packed.data(), 1, 8);
        assert(unpacked.size() == 8);
        for (size_t i = 0; i < 8; ++i) {
            assert(unpacked[i] == data[i]);
        }
        std::cout << "  ✓ Test 1: 8-element row packing & unpacking passed." << std::endl;
    }

    // Test 2: Multi-row, arbitrary column sizes (e.g. 50 columns > 32)
    {
        size_t rows = 5;
        size_t cols = 50;
        std::vector<int8_t> data(rows * cols);
        for (size_t i = 0; i < rows * cols; ++i) {
            data[i] = (i % 3 == 0) ? -1 : 1;
        }

        auto packed = littlebit::pack_binary_matrix(data.data(), rows, cols);
        size_t expected_words_per_row = (cols + 31) / 32; // (50 + 31) / 32 = 2
        assert(packed.size() == rows * expected_words_per_row);

        auto unpacked = littlebit::unpack_binary_matrix(packed.data(), rows, cols);
        assert(unpacked.size() == rows * cols);
        for (size_t i = 0; i < rows * cols; ++i) {
            assert(unpacked[i] == data[i]);
        }
        std::cout << "  ✓ Test 2: (5 x 50) matrix pack & unpack roundtrip passed." << std::endl;
    }

    // Test 3: Binary Matvec multiplication verification
    {
        size_t M = 16;
        size_t K = 64;
        std::vector<int8_t> matrix(M * K);
        std::vector<float> x(K);

        for (size_t i = 0; i < M * K; ++i) {
            matrix[i] = (i % 2 == 0) ? 1 : -1;
        }
        for (size_t i = 0; i < K; ++i) {
            x[i] = static_cast<float>(i + 1) * 0.1f;
        }

        auto packed = littlebit::pack_binary_matrix(matrix.data(), M, K);

        std::vector<float> y_fast(M, 0.0f);
        littlebit::binary_matvec_mult(packed.data(), x.data(), y_fast.data(), M, K);

        // Ground truth calculation
        for (size_t i = 0; i < M; ++i) {
            float expected_sum = 0.0f;
            for (size_t j = 0; j < K; ++j) {
                expected_sum += static_cast<float>(matrix[i * K + j]) * x[j];
            }
            assert(std::fabs(y_fast[i] - expected_sum) < 1e-4f);
        }
        std::cout << "  ✓ Test 3: Direct binary_matvec_mult matches ground truth." << std::endl;
    }

    // Test 4: Transposed Binary Matvec multiplication verification
    {
        size_t M = 16;
        size_t K = 64;
        std::vector<int8_t> matrix(M * K);
        std::vector<float> x(M);

        for (size_t i = 0; i < M * K; ++i) {
            matrix[i] = (i % 3 == 0) ? -1 : 1;
        }
        for (size_t i = 0; i < M; ++i) {
            x[i] = static_cast<float>(i + 1) * 0.5f;
        }

        auto packed = littlebit::pack_binary_matrix(matrix.data(), M, K);

        std::vector<float> y_fast(K, 0.0f);
        littlebit::binary_matvec_mult_transposed(packed.data(), x.data(), y_fast.data(), M, K);

        // Ground truth calculation: y = A^T * x -> y_j = sum_i(A_{i,j} * x_i)
        for (size_t j = 0; j < K; ++j) {
            float expected_sum = 0.0f;
            for (size_t i = 0; i < M; ++i) {
                expected_sum += static_cast<float>(matrix[i * K + j]) * x[i];
            }
            assert(std::fabs(y_fast[j] - expected_sum) < 1e-4f);
        }
        std::cout << "  ✓ Test 4: Transposed binary_matvec_mult_transposed matches ground truth." << std::endl;
    }

    // Test 5: Edge cases and bit-exact parity against the frozen reference.
    {
        // Empty / degenerate shapes must not read out of bounds.
        check_matvec_case(0, 0, 0, false);
        check_matvec_case(0, 5, 0, false);
        check_matvec_case(3, 0, 0, false);

        const size_t ks[] = {1, 2, 7, 31, 32, 33, 63, 64, 65, 96, 97, 127, 128, 129, 255, 256, 257};
        for (size_t K : ks) {
            for (size_t M : {size_t(1), size_t(3)}) {
                for (int pattern = 0; pattern < 4; ++pattern) {
                    check_matvec_case(M, K, pattern, false);
                    check_matvec_case(M, K, pattern, true); // padding bits must be ignored
                }
            }
        }
        std::cout << "  ✓ Test 5: binary_matvec_mult edge cases are bit-exact vs reference." << std::endl;
    }

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64)
    // Test 6: separate AVX2 kernel, bit-exact against the frozen scalar oracle.
    // AVX2 sources are not compiled on ARM64, so this whole section is x86-only.
    {
        const size_t ks[] = {1, 2, 7, 31, 32, 33, 63, 64, 65, 96, 97, 127, 128, 129, 255, 256, 257};
        const size_t ms[] = {1, 3, 7, 8, 9, 16, 17, 24, 32, 33};
        size_t checked = 0;
        for (size_t K : ks) {
            for (size_t M : ms) {
                for (int pattern = 0; pattern < 4; ++pattern) {
                    for (bool dirty : {false, true}) {
                        if (!candidate_matches_reference(M, K, pattern, dirty,
                                                         littlebit::binary_matvec_mult_avx2)) {
                            std::cerr << "AVX2 MISMATCH M=" << M << " K=" << K
                                      << " pattern=" << pattern << " dirty=" << dirty << '\n';
                            std::abort();
                        }
                        ++checked;
                    }
                }
            }
        }

        // Also verify that the runtime fallback path is bit-exact when disabled.
        setenv("LITTLEBIT_DISABLE_AVX2", "1", 1);
        for (size_t K : {size_t(7), size_t(64), size_t(129)}) {
            for (size_t M : {size_t(1), size_t(8), size_t(17)}) {
                if (!candidate_matches_reference(M, K, 3, true,
                                                 littlebit::binary_matvec_mult_avx2)) {
                    std::cerr << "AVX2 fallback MISMATCH M=" << M << " K=" << K << '\n';
                    std::abort();
                }
            }
        }
        unsetenv("LITTLEBIT_DISABLE_AVX2");
        std::cout << "  ✓ Test 6: AVX2 kernel (" << checked
                  << " shape/pattern cases) bit-exact incl. NaN payloads and padding."
                  << std::endl;
    }
#endif // defined(__x86_64__) || defined(__i386__) || defined(_M_X64)

#if defined(__aarch64__) && defined(__ARM_NEON)
    // Test 7: separate ARM64 NEON kernel, bit-exact against the frozen oracle.
    // This covers 4-row lane groups, scalar tails, dirty padding bits, NaN
    // payloads, infinities, signed zero, subnormals and non-multiple-of-32 K.
    {
        const size_t ks[] = {
            1, 2, 7, 31, 32, 33, 63, 64, 65, 96, 97,
            127, 128, 129, 255, 256, 257, 544, 545
        };
        const size_t ms[] = {1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 33};
        size_t checked = 0;
        for (size_t K : ks) {
            for (size_t M : ms) {
                for (int pattern = 0; pattern < 4; ++pattern) {
                    for (bool dirty : {false, true}) {
                        if (!candidate_matches_reference(M, K, pattern, dirty,
                                                         littlebit::binary_matvec_mult_neon)) {
                            std::cerr << "NEON MISMATCH M=" << M << " K=" << K
                                      << " pattern=" << pattern << " dirty=" << dirty << '\n';
                            std::abort();
                        }
                        ++checked;
                    }
                }
            }
        }

        // LITTLEBIT_DISABLE_NEON must route the kernel through the same scalar
        // oracle, including the final-tail path.
        setenv("LITTLEBIT_DISABLE_NEON", "1", 1);
        assert(!littlebit::neon_available());
        for (size_t K : {size_t(7), size_t(64), size_t(129), size_t(545)}) {
            for (size_t M : {size_t(1), size_t(4), size_t(9)}) {
                if (!candidate_matches_reference(M, K, 3, true,
                                                 littlebit::binary_matvec_mult_neon)) {
                    std::cerr << "NEON disabled-path MISMATCH M=" << M << " K=" << K << '\n';
                    std::abort();
                }
            }
        }
        unsetenv("LITTLEBIT_DISABLE_NEON");
        assert(littlebit::neon_available());
        std::cout << "  ✓ Test 7: ARM64 NEON kernel (" << checked
                  << " shape/pattern cases) bit-exact incl. tails and NaN payloads."
                  << std::endl;
    }
#endif

    std::cout << "All binary_packer unit tests passed successfully!" << std::endl;
    return 0;
}
