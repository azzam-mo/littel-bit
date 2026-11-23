#pragma once

#include "littlebit/common.h"
#include "littlebit/attention.h"
#include "littlebit/mlp.h"
#include "littlebit/ops.h"
#include "littlebit/safetensors.h"

namespace littlebit {

class TransformerBlock {
public:
    size_t layer_idx = 0;
    size_t hidden_size = 4096;

    std::vector<float> input_layernorm_weight;          // [hidden_size]
    std::vector<float> post_attention_layernorm_weight; // [hidden_size]

    Attention self_attn;
    MLP mlp;

    // Preallocated buffers
    mutable std::vector<float> norm_buf;
    mutable std::vector<float> attn_out_buf;
    mutable std::vector<float> residual_buf;
    mutable std::vector<float> mlp_out_buf;

    TransformerBlock() = default;

    void initialize(
        size_t idx,
        size_t h_size = 4096,
        size_t inter_size = 12288,
        size_t max_seq = 1024,
        size_t q_heads = 32,
        size_t kv_heads = 8,
        size_t h_dim = 128
    );

    // Load weights from SafeTensors file
    bool load_from_safetensors(const SafeTensorsFile& file, const std::string& layer_prefix);

    // Forward pass for single token at position pos
    void forward(
        const float* x,
        float* out,
        size_t pos,
        const RoPEEmbeddings& rope,
        KVCache& kv_cache
    ) const;
};

} // namespace littlebit
