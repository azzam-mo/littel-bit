#include "littlebit/ops.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

int main() {
    std::cout << "[TEST] Running ops unit tests..." << std::endl;

    // Test 1: RMSNorm
    {
        std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> weight = {0.5f, 1.0f, 1.5f, 2.0f};
        std::vector<float> out(4);

        littlebit::rmsnorm(x.data(), weight.data(), out.data(), 4, 1e-6f);

        // mean(x^2) = (1 + 4 + 9 + 16)/4 = 30/4 = 7.5
        // rms = sqrt(7.5) = 2.7386127875
        float rms = std::sqrt(7.5f);
        for (size_t i = 0; i < 4; ++i) {
            float expected = (x[i] / rms) * weight[i];
            assert(std::fabs(out[i] - expected) < 1e-4f);
        }
        std::cout << "  ✓ RMSNorm calculation matches exact formula." << std::endl;
    }

    // Test 2: SiLU
    {
        float x = 2.0f;
        float silu_val = littlebit::silu_scalar(x);
        float expected = 2.0f / (1.0f + std::exp(-2.0f));
        assert(std::fabs(silu_val - expected) < 1e-5f);
        std::cout << "  ✓ SiLU scalar calculation passed." << std::endl;
    }

    // Test 3: Softmax
    {
        std::vector<float> logits = {1.0f, 2.0f, 3.0f};
        littlebit::softmax(logits.data(), 3);
        float sum = logits[0] + logits[1] + logits[2];
        assert(std::fabs(sum - 1.0f) < 1e-5f);
        assert(logits[2] > logits[1] && logits[1] > logits[0]);
        std::cout << "  ✓ Softmax sums to 1.0 and preserves order." << std::endl;
    }

    // Test 4: RoPE Embeddings
    {
        littlebit::RoPEEmbeddings rope;
        rope.initialize(128, 128, 1000000.0f);

        std::vector<float> q(128, 1.0f);
        std::vector<float> k(128, 1.0f);

        rope.apply(q.data(), k.data(), 1, 1, 0); // Position 0: cos(0)=1, sin(0)=0 -> unchanged!
        for (size_t i = 0; i < 128; ++i) {
            assert(std::fabs(q[i] - 1.0f) < 1e-4f);
            assert(std::fabs(k[i] - 1.0f) < 1e-4f);
        }

        // Position 1: should rotate
        rope.apply(q.data(), k.data(), 1, 1, 1);
        float norm_before = std::sqrt(128.0f);
        float norm_after = 0.0f;
        for (size_t i = 0; i < 128; ++i) norm_after += q[i] * q[i];
        norm_after = std::sqrt(norm_after);
        // RoPE is an orthogonal rotation, norm must be preserved!
        assert(std::fabs(norm_after - norm_before) < 1e-3f);
        std::cout << "  ✓ RoPE maintains orthogonality and norm preservation." << std::endl;
    }

    std::cout << "All ops unit tests passed successfully!" << std::endl;
    return 0;
}
