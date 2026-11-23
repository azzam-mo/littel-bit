#pragma once

#include "littlebit/common.h"
#include "littlebit/linear.h"
#include "littlebit/ops.h"
#include <vector>
#include <memory>

namespace littlebit {

struct KVCache {
    size_t max_seq_len = 1024;
    size_t num_kv_heads = 8;
    size_t head_dim = 128;
    size_t current_len = 0;

    // Flattened storage: [max_seq_len * num_kv_heads * head_dim]
    std::vector<float> k_data;
    std::vector<float> v_data;

    void initialize(size_t max_len, size_t kv_heads, size_t h_dim);
    void clear();

    float* get_k_ptr(size_t pos) {
        return k_data.data() + pos * (num_kv_heads * head_dim);
    }
    const float* get_k_ptr(size_t pos) const {
        return k_data.data() + pos * (num_kv_heads * head_dim);
    }

    float* get_v_ptr(size_t pos) {
        return v_data.data() + pos * (num_kv_heads * head_dim);
    }
    const float* get_v_ptr(size_t pos) const {
        return v_data.data() + pos * (num_kv_heads * head_dim);
    }
};

class Attention {
public:
    size_t hidden_size = 4096;
    size_t num_q_heads = 32;
    size_t num_kv_heads = 8;
    size_t head_dim = 128;
    float scale = 1.0f / std::sqrt(128.0f);

    // Linear projections
    LittleBitLinear q_proj;
    LittleBitLinear k_proj;
    LittleBitLinear v_proj;
    LittleBitLinear o_proj;

    // QK Normalization weights
    std::vector<float> q_norm_weight; // [head_dim]
    std::vector<float> k_norm_weight; // [head_dim]

    // Preallocated buffers for inference
    mutable std::vector<float> q_buf;   // [num_q_heads * head_dim]
    mutable std::vector<float> k_buf;   // [num_kv_heads * head_dim]
    mutable std::vector<float> v_buf;   // [num_kv_heads * head_dim]
    mutable std::vector<float> attn_scores; // [max_seq_len]
    mutable std::vector<float> attn_out;    // [hidden_size]

    Attention() = default;

    void initialize(
        size_t h_size = 4096,
        size_t q_heads = 32,
        size_t kv_heads = 8,
        size_t h_dim = 128,
        size_t max_seq = 1024
    );

    // Forward pass for single token at position `pos`
    void forward(
        const float* x,
        float* out,
        size_t pos,
        const RoPEEmbeddings& rope,
        KVCache& kv_cache
    ) const;
};

} // namespace littlebit
