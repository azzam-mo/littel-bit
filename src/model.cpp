#include "littlebit/model.h"
#include "lm_head_avx2.h"
#include "nlohmann/json.hpp"

#include <fstream>
#include <iostream>
#include <chrono>

#if defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace littlebit {

size_t Qwen3Model::get_current_rss_mb() {
#if defined(__APPLE__)
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) == KERN_SUCCESS) {
        return static_cast<size_t>(info.resident_size / (1024 * 1024));
    }
    return 0;
#elif defined(__linux__)
    long rss_pages = 0;
    FILE* fp = fopen("/proc/self/statm", "r");
    if (fp) {
        if (fscanf(fp, "%*s%ld", &rss_pages) == 1) {
            fclose(fp);
            long page_size_kb = sysconf(_SC_PAGESIZE) / 1024;
            return static_cast<size_t>((rss_pages * page_size_kb) / 1024);
        }
        fclose(fp);
    }
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return static_cast<size_t>(usage.ru_maxrss / 1024);
    }
    return 0;
#else
    return 0;
#endif
}

bool Qwen3Model::load(const std::string& model_dir, size_t max_seq_len, size_t num_layers_to_load) {
    auto start_time = std::chrono::high_resolution_clock::now();

    config.max_seq_len = max_seq_len;

    // 1. Load config.json
    std::string config_path = model_dir + "/config.json";
    std::ifstream cfg_in(config_path);
    if (cfg_in.is_open()) {
        try {
            nlohmann::json cfg_json;
            cfg_in >> cfg_json;
            config.vocab_size = cfg_json.value("vocab_size", config.vocab_size);
            config.hidden_size = cfg_json.value("hidden_size", config.hidden_size);
            config.num_hidden_layers = cfg_json.value("num_hidden_layers", config.num_hidden_layers);
            config.num_attention_heads = cfg_json.value("num_attention_heads", config.num_attention_heads);
            config.num_key_value_heads = cfg_json.value("num_key_value_heads", config.num_key_value_heads);
            config.head_dim = cfg_json.value("head_dim", config.head_dim);
            config.intermediate_size = cfg_json.value("intermediate_size", config.intermediate_size);
            config.rope_theta = cfg_json.value("rope_theta", config.rope_theta);
            std::cout << "[MODEL] Loaded config: " << config.num_hidden_layers << " layers, hidden=" 
                      << config.hidden_size << ", vocab=" << config.vocab_size << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[MODEL] Warning: Failed to parse config.json: " << e.what() << std::endl;
        }
    }

    if (num_layers_to_load > 0 && num_layers_to_load < config.num_hidden_layers) {
        config.num_hidden_layers = num_layers_to_load;
        std::cout << "[MODEL] Limiting layers to load: " << config.num_hidden_layers << std::endl;
    }

    // Validate architecture before allocating any buffers. A mismatch here would
    // otherwise cause out-of-bounds RoPE/KV accesses on device.
    if (config.num_attention_heads == 0 || config.num_key_value_heads == 0 ||
        config.num_attention_heads % config.num_key_value_heads != 0 ||
        config.head_dim == 0 || config.head_dim % 2 != 0 ||
        config.num_hidden_layers == 0 || config.hidden_size == 0 || config.vocab_size == 0) {
        std::cerr << "[MODEL] Error: Invalid model architecture in config.json" << std::endl;
        return false;
    }
    if (config.num_attention_heads * config.head_dim != config.hidden_size) {
        std::cerr << "[MODEL] Error: num_attention_heads * head_dim != hidden_size" << std::endl;
        return false;
    }

    // 2. Load Tokenizer — prefer the official tokenizer.json because it contains
    // merge ranks and added tokens that vocab.json/merges.txt can omit.
    std::string tokenizer_json_path = model_dir + "/tokenizer.json";
    std::string vocab_path = model_dir + "/vocab.json";
    std::string merges_path = model_dir + "/merges.txt";
    bool tokenizer_ok = tokenizer.load_from_json(tokenizer_json_path);
    if (!tokenizer_ok) {
        tokenizer_ok = tokenizer.load(vocab_path, merges_path);
    }
    if (!tokenizer_ok) {
        std::cerr << "[MODEL] Error: Tokenizer files not found or invalid in " << model_dir << std::endl;
        return false;
    }

    // 3. Open SafeTensors with mmap
    std::string weights_path = model_dir + "/model.safetensors";
    if (!weights_file.open(weights_path)) {
        std::cerr << "[MODEL] Error: Cannot open weights file " << weights_path << std::endl;
        return false;
    }

    // 4. Memory-map Embedding table
    if (weights_file.has_tensor("model.embed_tokens.weight")) {
        const auto& info = weights_file.get_tensor_info("model.embed_tokens.weight");
        embed_tokens_mmap = reinterpret_cast<const uint16_t*>(info.data_ptr);
        std::cout << "[MODEL] Memory-mapped embedding table: " << info.shape[0] << "x" << info.shape[1] << std::endl;
    }

    // 5. Memory-map LM Head
    if (weights_file.has_tensor("lm_head.weight")) {
        const auto& info = weights_file.get_tensor_info("lm_head.weight");
        lm_head_mmap = reinterpret_cast<const uint16_t*>(info.data_ptr);
        std::cout << "[MODEL] Memory-mapped lm_head table: " << info.shape[0] << "x" << info.shape[1] << std::endl;
    }

    // 6. Final Norm
    weights_file.load_tensor_to_float("model.norm.weight", final_norm_weight);
    if (final_norm_weight.empty()) {
        final_norm_weight.assign(config.hidden_size, 1.0f);
    }

    // 7. Initialize RoPE
    rope.initialize(config.max_seq_len, config.head_dim, config.rope_theta);

    // 8. Allocate buffers
    hidden_state_a.resize(config.hidden_size);
    hidden_state_b.resize(config.hidden_size);
    logits_buf.resize(config.vocab_size);

    // 9. Load Layers
    layers.resize(config.num_hidden_layers);
    kv_caches.resize(config.num_hidden_layers);

    for (size_t l = 0; l < config.num_hidden_layers; ++l) {
        layers[l].initialize(
            l, config.hidden_size, config.intermediate_size, config.max_seq_len,
            config.num_attention_heads, config.num_key_value_heads, config.head_dim
        );
        kv_caches[l].initialize(config.max_seq_len, config.num_key_value_heads, config.head_dim);

        std::string prefix = "model.layers." + std::to_string(l);
        layers[l].load_from_safetensors(weights_file, prefix);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    last_load_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    std::cout << "[MODEL] Successfully loaded in " << (last_load_time_ms / 1000.0) << "s. Current RSS: "
              << get_current_rss_mb() << " MB" << std::endl;

    return true;
}

void Qwen3Model::clear_cache() {
    for (auto& cache : kv_caches) {
        cache.clear();
    }
}

const float* Qwen3Model::forward_token(int32_t token_id, size_t pos) {
    // 1. Embedding lookup from mmap
    if (embed_tokens_mmap && token_id >= 0 && static_cast<size_t>(token_id) < config.vocab_size) {
        const uint16_t* token_embed_bf16 = embed_tokens_mmap + token_id * config.hidden_size;
        for (size_t i = 0; i < config.hidden_size; ++i) {
            hidden_state_a[i] = bf16_to_fp32(token_embed_bf16[i]);
        }
    } else {
        std::fill(hidden_state_a.begin(), hidden_state_a.end(), 0.0f);
    }

    // 2. Transformer layers forward
    float* in_ptr = hidden_state_a.data();
    float* out_ptr = hidden_state_b.data();

    for (size_t l = 0; l < config.num_hidden_layers; ++l) {
        layers[l].forward(in_ptr, out_ptr, pos, rope, kv_caches[l]);
        std::swap(in_ptr, out_ptr);
    }

    // 3. Final RMSNorm
    rmsnorm(in_ptr, final_norm_weight.data(), out_ptr, config.hidden_size, config.rms_norm_eps);

    // 4. Compute Logits from LM Head
    if (lm_head_mmap) {
        lm_head_logits(
            lm_head_mmap, out_ptr, logits_buf.data(),
            config.vocab_size, config.hidden_size
        );
    } else {
        std::fill(logits_buf.begin(), logits_buf.end(), 0.0f);
    }

    return logits_buf.data();
}

PerformanceMetrics Qwen3Model::generate(
    const std::string& prompt,
    size_t max_new_tokens,
    float temperature,
    float top_p,
    std::function<bool(int32_t, const std::string&)> on_token
) {
    PerformanceMetrics metrics;
    metrics.load_time_ms = last_load_time_ms;
    clear_cache();

    auto gen_start = std::chrono::high_resolution_clock::now();

    // 1. Encode prompt using the Qwen3 chat template (control tokens kept verbatim)
    std::vector<int32_t> input_tokens = tokenizer.encode(tokenizer.apply_chat_template(prompt), false);
    if (input_tokens.empty()) {
        input_tokens.push_back(tokenizer.bos_id);
    }

    // 2. Prefill phase — never step past the RoPE table / KV cache capacity.
    const float* logits = nullptr;
    size_t prefill_len = std::min(input_tokens.size(), config.max_seq_len);
    for (size_t i = 0; i < prefill_len; ++i) {
        logits = forward_token(input_tokens[i], i);
    }
    if (!logits) {
        return metrics;
    }

    auto first_token_time = std::chrono::high_resolution_clock::now();
    metrics.ttft_ms = std::chrono::duration<double, std::milli>(first_token_time - gen_start).count();

    // 3. Generation phase
    size_t cur_pos = prefill_len;
    int32_t next_token = sample_temperature_top_p(logits, config.vocab_size, temperature, top_p);

    for (size_t gen_i = 0; gen_i < max_new_tokens; ++gen_i) {
        metrics.tokens_generated++;

        std::string token_str = tokenizer.decode(next_token);
        if (on_token) {
            bool keep_going = on_token(next_token, token_str);
            if (!keep_going) break;
        }

        if (tokenizer.is_eos(next_token)) {
            break;
        }

        if (cur_pos >= config.max_seq_len) {
            break;
        }

        logits = forward_token(next_token, cur_pos++);
        next_token = sample_temperature_top_p(logits, config.vocab_size, temperature, top_p);
    }

    auto gen_end = std::chrono::high_resolution_clock::now();
    metrics.total_gen_time_ms = std::chrono::duration<double, std::milli>(gen_end - gen_start).count();

    if (metrics.total_gen_time_ms > 0.0) {
        metrics.tokens_per_second = (static_cast<double>(metrics.tokens_generated) * 1000.0) / metrics.total_gen_time_ms;
    }

    metrics.resident_memory_mb = get_current_rss_mb();

    // Calculate KV cache size
    size_t total_kv_elements = config.num_hidden_layers * 2 * config.max_seq_len * config.num_key_value_heads * config.head_dim;
    metrics.kv_cache_mb = (total_kv_elements * sizeof(float)) / (1024 * 1024);

    return metrics;
}

} // namespace littlebit
