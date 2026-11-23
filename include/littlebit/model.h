#pragma once

#include "littlebit/common.h"
#include "littlebit/safetensors.h"
#include "littlebit/tokenizer.h"
#include "littlebit/transformer_block.h"
#include "littlebit/ops.h"

#include <string>
#include <vector>
#include <functional>
#include <chrono>

namespace littlebit {

struct ModelConfig {
    std::string model_type = "qwen3";
    size_t vocab_size = 151936;
    size_t hidden_size = 4096;
    size_t num_hidden_layers = 36;
    size_t num_attention_heads = 32;
    size_t num_key_value_heads = 8;
    size_t head_dim = 128;
    size_t intermediate_size = 12288;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;
    size_t max_seq_len = 1024;
};

struct PerformanceMetrics {
    double load_time_ms = 0.0;
    double ttft_ms = 0.0;           // Time to first token
    double total_gen_time_ms = 0.0;
    size_t tokens_generated = 0;
    double tokens_per_second = 0.0;
    size_t resident_memory_mb = 0;
    size_t kv_cache_mb = 0;
};

class Qwen3Model {
public:
    ModelConfig config;
    Tokenizer tokenizer;
    SafeTensorsFile weights_file;

    RoPEEmbeddings rope;
    std::vector<TransformerBlock> layers;
    std::vector<KVCache> kv_caches;

    // Normalization and Head
    std::vector<float> final_norm_weight; // [hidden_size]

    // Pointers into mmap for zero-RAM tables
    const uint16_t* embed_tokens_mmap = nullptr; // BF16 [vocab_size, hidden_size]
    const uint16_t* lm_head_mmap = nullptr;      // BF16 [vocab_size, hidden_size]

    // Preallocated buffers for inference
    mutable std::vector<float> hidden_state_a;
    mutable std::vector<float> hidden_state_b;
    mutable std::vector<float> logits_buf;

    Qwen3Model() = default;

    // Load full model from directory containing model.safetensors, config.json, vocab.json, merges.txt
    bool load(const std::string& model_dir, size_t max_seq_len = 1024, size_t num_layers_to_load = 0);

    // Get current RAM usage in MB
    static size_t get_current_rss_mb();

    // Single step forward: takes token_id at position pos, returns pointer to logits [vocab_size]
    const float* forward_token(int32_t token_id, size_t pos);

    // Streaming generation loop
    PerformanceMetrics generate(
        const std::string& prompt,
        size_t max_new_tokens = 128,
        float temperature = 0.6f,
        float top_p = 0.95f,
        std::function<bool(int32_t token_id, const std::string& token_str)> on_token = nullptr
    );

    void clear_cache();
};

} // namespace littlebit
