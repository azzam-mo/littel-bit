#include "littlebit/binary_packer.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <random>
#include <cmath>

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

    std::cout << "All binary_packer unit tests passed successfully!" << std::endl;
    return 0;
}
