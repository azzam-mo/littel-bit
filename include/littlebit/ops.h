#pragma once

#include "littlebit/common.h"
#include <vector>
#include <cmath>
#include <cstddef>

namespace littlebit {

// Root Mean Square Normalization (RMSNorm)
// y = (x / sqrt(mean(x^2) + eps)) * weight
void rmsnorm(
    const float* x,
    const float* weight,
    float* out,
    size_t dim,
    float eps = 1e-6f
);

// In-place or out-of-place SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))
inline float silu_scalar(float x) {
    return x / (1.0f + std::exp(-x));
}

void silu(const float* x, float* out, size_t size);

// Element-wise multiply: out[i] = a[i] * b[i]
void elementwise_mul(const float* a, const float* b, float* out, size_t size);

// Element-wise add: out[i] = a[i] + b[i]
void elementwise_add(const float* a, const float* b, float* out, size_t size);

// In-place Softmax
void softmax(float* x, size_t size);

// Precomputed Rotary Position Embedding (RoPE) tables
class RoPEEmbeddings {
public:
    size_t head_dim = 128;
    float theta = 1000000.0f; // Qwen3 default theta

    std::vector<float> cos_table; // [max_seq_len, head_dim / 2]
    std::vector<float> sin_table; // [max_seq_len, head_dim / 2]

    void initialize(size_t max_seq_len = 4096, size_t h_dim = 128, float th = 1000000.0f);

    // Apply RoPE to query and key vectors for a specific position
    void apply(
        float* q,          // [num_q_heads * head_dim]
        float* k,          // [num_kv_heads * head_dim]
        size_t num_q_heads,
        size_t num_kv_heads,
        size_t position
    ) const;
};

// Sampling functions
int32_t sample_greedy(const float* logits, size_t vocab_size);
int32_t sample_temperature_top_p(
    const float* logits,
    size_t vocab_size,
    float temperature = 0.7f,
    float top_p = 0.9f
);

} // namespace littlebit
