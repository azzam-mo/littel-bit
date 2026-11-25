#include "littlebit/binary_packer.h"
#include "littlebit/scalar_kernels.h"
#if defined(__aarch64__) && defined(__ARM_NEON)
#include "littlebit/binary_packer_neon_variants_internal.h"
#endif
#include <iostream>
#include <cassert>
#include <vector>
#include <random>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <string>

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
                    sum = littlebit::detail::ordered_add(sum, val);
                } else {
                    sum = littlebit::detail::ordered_sub(sum, val);
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

static uint32_t float_to_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static std::string hex32(uint32_t value) {
    std::ostringstream out;
    out << "0x" << std::hex << std::setw(8) << std::setfill('0') << value;
    return out.str();
}

static std::string float_diag(float value) {
    std::ostringstream out;
    out << "bits=" << hex32(float_to_bits(value)) << " value=" << value;
    return out.str();
}

static void print_packed_row_words(
    std::ostream& out, const int32_t* packed, size_t row, size_t words_per_row
) {
    out << "    row " << row << " packed words (hex / little-endian bytes):";
    for (size_t w = 0; w < words_per_row; ++w) {
        const uint32_t word = static_cast<uint32_t>(packed[row * words_per_row + w]);
        const unsigned char* bytes = reinterpret_cast<const unsigned char*>(&word);
        out << ' ' << hex32(word) << " [";
        for (size_t byte = 0; byte < sizeof(word); ++byte) {
            if (byte != 0) out << ' ';
            out << std::hex << std::setw(2) << std::setfill('0')
                << static_cast<unsigned>(bytes[byte]);
        }
        out << ']';
    }
    out << '\n';
}

// Print the final ordered-FP operations for a row. This makes a tail/NaN
// divergence visible even when the candidate has no lane-level trace hook.
static void print_reference_tail_trace(
    std::ostream& out, const int32_t* packed, const float* x, size_t row, size_t K
) {
    const size_t words_per_row = (K + 31) / 32;
    const size_t tail_start = K > 8 ? K - 8 : 0;
    float sum = 0.0f;
    size_t k = 0;
    out << "  reference tail accumulators (k, sign, x bits, accumulator bits):\n";
    for (size_t w = 0; w < words_per_row; ++w) {
        const uint32_t word = static_cast<uint32_t>(packed[row * words_per_row + w]);
        const size_t bits_to_process = (K - k >= 32) ? 32 : (K - k);
        for (size_t b = 0; b < bits_to_process; ++b) {
            const float val = x[k];
            const bool is_one = ((word >> b) & 1u) != 0u;
            sum = is_one
                ? littlebit::detail::ordered_sub(sum, val)
                : littlebit::detail::ordered_add(sum, val);
            if (k >= tail_start) {
                out << "    k=" << k << " op=" << (is_one ? "sub" : "add")
                    << " x=" << hex32(float_to_bits(val))
                    << " acc=" << hex32(float_to_bits(sum)) << '\n';
            }
            ++k;
        }
    }
}

static float avx2_test_float_from_bits(uint32_t u) {
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// Rejected sign-handling candidates (XOR sign bit and multiply by +/-1.0f)
// remain in tests/binary_matvec_probe.cpp, where they are exercised explicitly.
// They are intentionally not duplicated here because this test only checks
// kernels that are eligible for the engine.
using MatvecFn = void (*)(const int32_t*, const float*, float*, size_t, size_t);

static bool candidate_matches_reference(
    size_t M, size_t K, int pattern, bool dirty_padding, MatvecFn fn,
    const char* label = "candidate"
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
        avx2_test_float_from_bits(0x7fc00001u),
        avx2_test_float_from_bits(0xffc00001u),
        avx2_test_float_from_bits(0x7f800001u),
        avx2_test_float_from_bits(0xff800001u),
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
        if (!same_float_bits(y_ref[i], y_cand[i])) {
            const size_t words_per_row = (K + 31) / 32;
            std::cerr << label << " MISMATCH M=" << M << " K_logical=" << K
                      << " K_padded=" << (words_per_row * 32)
                      << " pattern=" << pattern << " dirty=" << dirty_padding
                      << " row=" << i << '\n';
            std::cerr << "  stride_words=" << words_per_row
                      << " stride_bytes=" << (words_per_row * sizeof(int32_t))
                      << " packed_bytes=" << (packed.size() * sizeof(int32_t)) << '\n';
            std::cerr << "  expected output: " << float_diag(y_ref[i]) << '\n';
            std::cerr << "  actual   output: " << float_diag(y_cand[i]) << '\n';
            print_packed_row_words(std::cerr, packed.data(), i, words_per_row);
            print_reference_tail_trace(std::cerr, packed.data(), x.data(), i, K);
            return false;
        }
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
        avx2_test_float_from_bits(0x7fc00001u),
        avx2_test_float_from_bits(0xffc00001u),
        avx2_test_float_from_bits(0x7f800001u),
        avx2_test_float_from_bits(0xff800001u),
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
            const size_t words_per_row = (K + 31) / 32;
            std::cerr << "ENGINE MISMATCH M=" << M << " K_logical=" << K
                      << " K_padded=" << (words_per_row * 32)
                      << " pattern=" << pattern << " dirty=" << dirty_padding
                      << " row=" << i << '\n';
            std::cerr << "  stride_words=" << words_per_row
                      << " stride_bytes=" << (words_per_row * sizeof(int32_t))
                      << " packed_bytes=" << (packed.size() * sizeof(int32_t)) << '\n';
            std::cerr << "  expected output: " << float_diag(y_ref[i]) << '\n';
            std::cerr << "  actual   output: " << float_diag(y_opt[i]) << '\n';
            print_packed_row_words(std::cerr, packed.data(), i, words_per_row);
            print_reference_tail_trace(std::cerr, packed.data(), x.data(), i, K);
            std::abort();
        }
    }
}

#if defined(__aarch64__) && defined(__ARM_NEON)
struct Neon4TraceState {
    size_t M = 0;
    size_t K = 0;
    size_t row_offset = 0;
    const int32_t* packed = nullptr;
    const float* x = nullptr;
    bool found = false;
    size_t first_bad_step = 0;
    size_t first_bad_lane = 0;
    uint32_t first_bad_expected = 0;
    uint32_t first_bad_actual = 0;
};

static void neon4_trace_callback(
    size_t bit_index, size_t lane_count, const uint32_t* lane_bits, void* context
) {
    auto* state = static_cast<Neon4TraceState*>(context);
    const size_t words_per_row = (state->K + 31) / 32;

    for (size_t lane = 0; lane < lane_count && lane < 4; ++lane) {
        float expected = 0.0f;
        size_t k = 0;
        for (size_t w = 0; w < words_per_row && k <= bit_index; ++w) {
            const uint32_t word = static_cast<uint32_t>(
                state->packed[(state->row_offset + lane) * words_per_row + w]
            );
            const size_t bits_to_process = (state->K - k >= 32) ? 32 : (state->K - k);
            for (size_t b = 0; b < bits_to_process && k <= bit_index; ++b) {
                const float val = state->x[k];
                expected = ((word >> b) & 1u) != 0u
                    ? littlebit::detail::ordered_sub(expected, val)
                    : littlebit::detail::ordered_add(expected, val);
                ++k;
            }
        }
        const uint32_t expected_bits = float_to_bits(expected);
        if (!state->found && expected_bits != lane_bits[lane]) {
            state->found = true;
            state->first_bad_step = bit_index;
            state->first_bad_lane = lane;
            state->first_bad_expected = expected_bits;
            state->first_bad_actual = lane_bits[lane];
        }
    }
}

static void build_special_case(
    size_t M, size_t K, int pattern, bool dirty_padding,
    std::vector<int8_t>& matrix, std::vector<int32_t>& packed, std::vector<float>& x
) {
    matrix.assign(M * K, 1);
    for (size_t i = 0; i < M * K; ++i) {
        switch (pattern) {
            case 0: matrix[i] = 1; break;
            case 1: matrix[i] = -1; break;
            case 2: matrix[i] = (i % 2 == 0) ? 1 : -1; break;
            default: matrix[i] = ((i * 2654435761u) >> 31) ? -1 : 1; break;
        }
    }
    packed = littlebit::pack_binary_matrix(matrix.data(), M, K);

    if (dirty_padding && K % 32 != 0 && M > 0) {
        const size_t words_per_row = (K + 31) / 32;
        const uint32_t pad_mask = ~0u << (K % 32);
        for (size_t r = 0; r < M; ++r) {
            packed[r * words_per_row + words_per_row - 1] |=
                static_cast<int32_t>(pad_mask);
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
        avx2_test_float_from_bits(0x7fc00001u),
        avx2_test_float_from_bits(0xffc00001u),
        avx2_test_float_from_bits(0x7f800001u),
        avx2_test_float_from_bits(0xff800001u),
        std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max(),
        std::numeric_limits<float>::min(),
        std::numeric_limits<float>::denorm_min(),
        3.14159265f, -2.71828183f
    };
    x.assign(K, 0.0f);
    for (size_t i = 0; i < K; ++i) {
        x[i] = specials[i % (sizeof(specials) / sizeof(specials[0]))];
    }
}
#endif

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
        [[maybe_unused]] uint32_t expected = (1 << 1) | (1 << 4) | (1 << 5) | (1 << 7);
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
        [[maybe_unused]] size_t expected_words_per_row = (cols + 31) / 32; // (50 + 31) / 32 = 2
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
            [[maybe_unused]] float expected_sum = 0.0f;
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
            [[maybe_unused]] float expected_sum = 0.0f;
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
    // Test 7: every ARM64 NEON variant, bit-exact against the frozen oracle.
    // Covers 4/8/16/32-row groups, vtst sign handling, prefetch, scalar tails,
    // dirty padding bits, NaN payloads, Inf, signed zeros and subnormals.
    {
        struct Variant { const char* name; MatvecFn fn; };
        const Variant variants[] = {
            {"NEON4-engine", littlebit::binary_matvec_mult_neon},
            {"NEON8", littlebit::bench::neon_variant_8},
            {"NEON16", littlebit::bench::neon_variant_16},
            {"NEON32", littlebit::bench::neon_variant_32},
            {"vtst4", littlebit::bench::neon_variant_vtst},
            {"prefetch32", littlebit::bench::neon_variant_prefetch},
            // Public engine entry point: with NEON enabled this dispatches to
            // NEON32, so the default production path is covered by the same
            // raw-bit oracle and explicit both-NaN pair checks.
            {"engine", littlebit::binary_matvec_mult},
        };

        const size_t ks[] = {
            1, 2, 7, 31, 32, 33, 63, 64, 65, 96, 97,
            127, 128, 129, 255, 256, 257, 544, 545
        };
        const size_t ms[] = {1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 33, 40, 41, 45, 69};
        size_t checked = 0;
        for (size_t K : ks) {
            for (size_t M : ms) {
                for (int pattern = 0; pattern < 4; ++pattern) {
                    for (bool dirty : {false, true}) {
                        for (const Variant& variant : variants) {
                            if (!candidate_matches_reference(M, K, pattern, dirty,
                                                             variant.fn, variant.name)) {
                                std::cerr << "NEON VARIANT MISMATCH [" << variant.name
                                          << "] M=" << M << " K=" << K
                                          << " pattern=" << pattern << " dirty=" << dirty
                                          << '\n';
                                std::abort();
                            }
                            ++checked;
                        }
                    }
                }
            }
        }

        // Explicit controlled case: k=0 loads a positive qNaN into sum with an
        // add; k=1 combines that NaN sum with a negative qNaN of a different
        // payload using either add or sub. This is not inferred from the cycling
        // specials arrays above.
        auto explicit_nan_pair_matches = [&](bool subtract, MatvecFn fn) {
            const size_t M = 33;
            const size_t K = 64;
            const size_t words_per_row = (K + 31) / 32;
            std::vector<int32_t> packed(M * words_per_row, 0);
            for (size_t r = 0; r < M; ++r) {
                // bit 0 = +1 (add x[0]); bit 1 selects +1 (add) or -1 (sub x[1]).
                packed[r * words_per_row] = subtract ? 2 : 0;
            }

            std::vector<float> x(K, 1.0f);
            x[0] = avx2_test_float_from_bits(0x7fc00001u); // +qNaN, payload 1
            x[1] = avx2_test_float_from_bits(0xffc12345u); // -qNaN, different payload

            std::vector<float> y_ref(M, 0.0f), y_cand(M, 0.0f);
            reference_binary_matvec_mult(packed.data(), x.data(), y_ref.data(), M, K);
            fn(packed.data(), x.data(), y_cand.data(), M, K);
            for (size_t i = 0; i < M; ++i) {
                if (!same_float_bits(y_ref[i], y_cand[i])) return false;
            }
            return true;
        };

        const size_t variant_count = sizeof(variants) / sizeof(variants[0]);
        size_t nan_pair_checked = 0;
        for (bool subtract : {false, true}) {
            for (const Variant& variant : variants) {
                if (!explicit_nan_pair_matches(subtract, variant.fn)) {
                    std::cerr << "NEON BOTH-NAN PAIR MISMATCH [" << variant.name
                              << "] op=" << (subtract ? "sub" : "add") << '\n';
                    std::abort();
                }
                ++nan_pair_checked;
            }
        }

        // Test 7c: permanent regression for the reported M=4, K=7, pattern=2,
        // dirty=0 NEON4 failure and its adjacent tail boundaries. The trace hook
        // walks the exact four-lane kernel so a future mismatch names the first
        // divergent accumulator instead of only the final output bits.
        {
            struct RegressionVariant { const char* name; MatvecFn fn; };
            const RegressionVariant regression_variants[] = {
                {"NEON4-engine", littlebit::binary_matvec_mult_neon},
                {"NEON8", littlebit::bench::neon_variant_8},
                {"NEON16", littlebit::bench::neon_variant_16},
                {"NEON32", littlebit::bench::neon_variant_32},
                {"vtst4", littlebit::bench::neon_variant_vtst},
                {"prefetch32", littlebit::bench::neon_variant_prefetch},
                {"engine", littlebit::binary_matvec_mult},
            };

            std::vector<int8_t> trace_matrix;
            std::vector<int32_t> trace_packed;
            std::vector<float> trace_x;
            build_special_case(4, 7, 2, false, trace_matrix, trace_packed, trace_x);
            std::vector<float> trace_y(4, 0.0f);
            Neon4TraceState trace_state;
            trace_state.M = 4;
            trace_state.K = 7;
            trace_state.row_offset = 0;
            trace_state.packed = trace_packed.data();
            trace_state.x = trace_x.data();
            littlebit::bench::neon_variant_4_trace(
                trace_packed.data(), trace_x.data(), trace_y.data(), 4, 7,
                neon4_trace_callback, &trace_state
            );
            if (trace_state.found) {
                const size_t words_per_row = (7 + 31) / 32;
                std::cerr << "NEON4 TRACE FIRST BAD INTERMEDIATE M=4 K=7 "
                          << "pattern=2 dirty=0\n";
                std::cerr << "  first divergence: row/lane=" << trace_state.first_bad_lane
                          << " k=" << trace_state.first_bad_step
                          << " expected=" << hex32(trace_state.first_bad_expected)
                          << " actual=" << hex32(trace_state.first_bad_actual) << '\n';
                std::cerr << "  stride_words=" << words_per_row
                          << " stride_bytes=" << (words_per_row * sizeof(int32_t))
                          << " K_logical=7 K_padded=" << (words_per_row * 32) << '\n';
                for (size_t row = 0; row < 4; ++row) {
                    print_packed_row_words(std::cerr, trace_packed.data(), row,
                                           words_per_row);
                }
                print_reference_tail_trace(std::cerr, trace_packed.data(),
                                           trace_x.data(), 0, 7);
                std::cerr << "  final outputs:";
                for (size_t row = 0; row < 4; ++row) {
                    std::cerr << " row" << row << '=' << float_diag(trace_y[row]);
                }
                std::cerr << '\n';
                std::abort();
            }

            size_t regression_checked = 0;
            for (size_t K : {size_t(5), size_t(6), size_t(7), size_t(8), size_t(9),
                             size_t(31), size_t(32), size_t(33), size_t(63),
                             size_t(64), size_t(65)}) {
                for (int pattern : {2, 3}) {
                    for (bool dirty : {false, true}) {
                        for (const RegressionVariant& variant : regression_variants) {
                            if (!candidate_matches_reference(
                                    4, K, pattern, dirty, variant.fn, variant.name)) {
                                std::cerr << "NEON4 TAIL REGRESSION [" << variant.name
                                          << "] M=4 K=" << K << " pattern=" << pattern
                                          << " dirty=" << dirty << '\n';
                                std::abort();
                            }
                            ++regression_checked;
                        }
                    }
                }
            }
            std::cout << "  ✓ Test 7c: NEON4 tail regression ("
                      << regression_checked << " cases across "
                      << (sizeof(regression_variants) / sizeof(regression_variants[0]))
                      << " kernels; first divergent intermediate: none)."
                      << std::endl;
        }

        // LITTLEBIT_DISABLE_NEON must route the engine kernel through the same
        // scalar oracle, including the final-tail path.
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

        std::cout << "  ✓ Test 7: ARM64 NEON variants (" << checked
                  << " shape/pattern cases across " << variant_count
                  << " kernels) bit-exact incl. tails and NaN payloads."
                  << std::endl;
        std::cout << "  ✓ Test 7b: explicit both-NaN sum/x pairs ("
                  << nan_pair_checked << " raw-bit checks across " << variant_count
                  << " kernels) bit-exact for add and sub." << std::endl;
    }
#endif

    std::cout << "All binary_packer unit tests passed successfully!" << std::endl;
    return 0;
}
