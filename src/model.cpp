#include "littlebit/model.h"
#include "lm_head_avx2.h"
#include "nlohmann/json.hpp"

#include <fstream>
#include <iostream>
#include <chrono>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <sstream>

#if defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace littlebit {

namespace {

struct LittleBitCheckpointMeta {
    bool residual = false;
};

const char* dtype_name(DType dtype) {
    switch (dtype) {
        case DType::BF16: return "BF16";
        case DType::F16:  return "F16";
        case DType::F32:  return "F32";
        case DType::I32:  return "I32";
        case DType::I64:  return "I64";
        case DType::U8:   return "U8";
        default:          return "UNKNOWN";
    }
}

size_t dtype_item_size(DType dtype) {
    switch (dtype) {
        case DType::BF16:
        case DType::F16:  return 2;
        case DType::F32:
        case DType::I32:  return 4;
        case DType::I64:  return 8;
        case DType::U8:   return 1;
        default:          return 0;
    }
}

std::string shape_to_string(const std::vector<int64_t>& shape) {
    std::ostringstream os;
    os << '[';
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) os << ',';
        os << shape[i];
    }
    os << ']';
    return os.str();
}

bool shape_equals(const SafeTensorInfo& info, std::initializer_list<int64_t> expected) {
    if (info.shape.size() != expected.size()) return false;
    size_t i = 0;
    for (int64_t dim : expected) {
        if (info.shape[i++] != dim) return false;
    }
    return true;
}

bool has_float_dtype(DType dtype) {
    return dtype == DType::BF16 || dtype == DType::F16 || dtype == DType::F32;
}

bool tensor_has_data(const SafeTensorInfo& info, const std::string& name, std::string& err) {
    if (!info.data_ptr) {
        err = "Tensor has no mapped data: " + name;
        return false;
    }
    return true;
}

bool check_packed_tensor(
    const SafeTensorsFile& weights, const std::string& name,
    int64_t rows, int64_t words_per_row, std::string& err
) {
    if (!weights.has_tensor(name)) {
        err = "Missing LittleBit tensor: " + name;
        return false;
    }
    const auto& info = weights.get_tensor_info(name);
    if (info.dtype != DType::I32) {
        err = "Expected I32 for " + name + ", got " + dtype_name(info.dtype);
        return false;
    }
    if (!shape_equals(info, {rows, words_per_row})) {
        err = "Unexpected shape for " + name + ": " + shape_to_string(info.shape) +
              ", expected [" + std::to_string(rows) + ',' + std::to_string(words_per_row) + ']';
        return false;
    }
    const size_t expected_bytes = static_cast<size_t>(rows) * static_cast<size_t>(words_per_row) * sizeof(int32_t);
    if (info.byte_size() != expected_bytes) {
        err = "Unexpected byte size for " + name + ": " + std::to_string(info.byte_size()) +
              ", expected " + std::to_string(expected_bytes);
        return false;
    }
    return tensor_has_data(info, name, err);
}

bool check_float_tensor(
    const SafeTensorsFile& weights, const std::string& name,
    int64_t expected_numel, std::string& err
) {
    if (!weights.has_tensor(name)) {
        err = "Missing LittleBit tensor: " + name;
        return false;
    }
    const auto& info = weights.get_tensor_info(name);
    if (!has_float_dtype(info.dtype)) {
        err = "Expected BF16/F16/F32 scale for " + name + ", got " + dtype_name(info.dtype);
        return false;
    }
    if (static_cast<int64_t>(info.numel()) != expected_numel) {
        err = "Unexpected scale length for " + name + ": " + std::to_string(info.numel()) +
              ", expected " + std::to_string(expected_numel);
        return false;
    }
    const size_t expected_bytes = info.numel() * dtype_item_size(info.dtype);
    if (info.byte_size() != expected_bytes) {
        err = "Unexpected byte size for " + name + ": " + std::to_string(info.byte_size()) +
              ", expected " + std::to_string(expected_bytes);
        return false;
    }
    return tensor_has_data(info, name, err);
}

bool read_i64_scalar(
    const SafeTensorsFile& weights, const std::string& name,
    int64_t& value, std::string& err
) {
    if (!weights.has_tensor(name)) {
        err = "Missing LittleBit metadata tensor: " + name;
        return false;
    }
    const auto& info = weights.get_tensor_info(name);
    if (info.dtype != DType::I64) {
        err = "Expected I64 for " + name + ", got " + dtype_name(info.dtype);
        return false;
    }
    if (!info.shape.empty()) {
        err = "Expected scalar shape [] for " + name + ", got " + shape_to_string(info.shape);
        return false;
    }
    if (info.byte_size() < sizeof(int64_t) || !tensor_has_data(info, name, err)) return false;
    std::memcpy(&value, info.data_ptr, sizeof(int64_t));
    return true;
}

bool read_i64_pair(
    const SafeTensorsFile& weights, const std::string& name,
    std::vector<int64_t>& values, std::string& err
) {
    if (!weights.has_tensor(name)) {
        err = "Missing LittleBit metadata tensor: " + name;
        return false;
    }
    const auto& info = weights.get_tensor_info(name);
    if (info.dtype != DType::I64) {
        err = "Expected I64 for " + name + ", got " + dtype_name(info.dtype);
        return false;
    }
    if (!shape_equals(info, {2})) {
        err = "Expected shape [2] for " + name + ", got " + shape_to_string(info.shape);
        return false;
    }
    if (info.byte_size() < 2 * sizeof(int64_t) || !tensor_has_data(info, name, err)) return false;
    values.resize(2);
    std::memcpy(values.data(), info.data_ptr, 2 * sizeof(int64_t));
    return true;
}

bool validate_littlebit_config(
    const std::string& model_dir, LittleBitCheckpointMeta& meta, std::string& err
) {
    const std::string path = model_dir + "/littlebit_config.json";
    std::ifstream in(path);
    if (!in.is_open()) {
        err = "Missing " + path + "; this loader requires a converted LittleBit checkpoint, not plain Qwen3 weights.";
        return false;
    }

    nlohmann::json lb;
    try {
        in >> lb;
    } catch (const std::exception& e) {
        err = "Failed to parse " + path + ": " + e.what();
        return false;
    }

    const std::string quant_mod = lb.value("quant_mod", "");
    if (quant_mod != "LittleBitLinear") {
        err = path + ": quant_mod must be \"LittleBitLinear\", got \"" + quant_mod + "\".";
        return false;
    }
    if (lb.value("quant_func", "").empty()) {
        err = path + ": missing quant_func.";
        return false;
    }
    if (!lb.contains("eff_bit") || !lb["eff_bit"].is_number()) {
        err = path + ": missing or invalid eff_bit.";
        return false;
    }
    const double eff_bit = lb["eff_bit"].get<double>();
    if (!std::isfinite(eff_bit) || eff_bit <= 0.0 || eff_bit > 2.0) {
        err = path + ": eff_bit must be in (0, 2].";
        return false;
    }
    if (!lb.contains("residual") || !lb["residual"].is_boolean()) {
        err = path + ": missing or invalid residual flag.";
        return false;
    }
    meta.residual = lb["residual"].get<bool>();
    if (!lb.contains("use_itq") || !lb["use_itq"].is_boolean()) {
        err = path + ": missing or invalid use_itq flag.";
        return false;
    }
    if (!lb.contains("split_dim") || !lb["split_dim"].is_number_integer() ||
        lb["split_dim"].get<int64_t>() <= 0) {
        err = path + ": missing or invalid split_dim.";
        return false;
    }
    if (!lb.contains("min_split_dim") || !lb["min_split_dim"].is_number_integer() ||
        lb["min_split_dim"].get<int64_t>() <= 0) {
        err = path + ": missing or invalid min_split_dim.";
        return false;
    }
    if (!lb.contains("kv_factor") || !lb["kv_factor"].is_number() ||
        lb["kv_factor"].get<double>() <= 0.0) {
        err = path + ": missing or invalid kv_factor.";
        return false;
    }
    if (!lb.contains("itq_n_iter") || !lb["itq_n_iter"].is_number_integer() ||
        lb["itq_n_iter"].get<int64_t>() < 0) {
        err = path + ": missing or invalid itq_n_iter.";
        return false;
    }

    // config.json duplicates these LittleBit knobs in the real release; require
    // agreement when both files define a key.
    std::ifstream cfg_in(model_dir + "/config.json");
    if (cfg_in.is_open()) {
        try {
            nlohmann::json cfg;
            cfg_in >> cfg;
            const char* keys[] = {
                "quant_func", "eff_bit", "split_dim", "residual", "kv_factor",
                "min_split_dim", "quant_mod", "use_itq", "itq_n_iter"
            };
            for (const char* key : keys) {
                if (lb.contains(key) && cfg.contains(key) && lb[key] != cfg[key]) {
                    err = std::string(path) + " disagrees with config.json on '" + key + "'.";
                    return false;
                }
            }
        } catch (const std::exception&) {
            // config.json parse failure is already warned about by load().
        }
    }
    return true;
}

bool has_littlebit_packed_tensors(const SafeTensorsFile& weights) {
    for (const auto& kv : weights.all_tensors()) {
        if (kv.first.find(".U_packed") != std::string::npos) return true;
    }
    return false;
}

bool has_plain_qwen3_linears(const SafeTensorsFile& weights) {
    if (weights.has_tensor("model.layers.0.self_attn.q_proj.weight")) return true;
    for (const auto& kv : weights.all_tensors()) {
        if (kv.first.find(".self_attn.q_proj.weight") != std::string::npos) return true;
    }
    return false;
}

bool vectors_equal(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(int64_t)) == 0;
}

bool validate_littlebit_tensors(
    const SafeTensorsFile& weights, const ModelConfig& config,
    bool residual, std::string& err
) {
    struct ProjSpec {
        const char* name;
        int64_t in;
        int64_t out;
    };

    const int64_t hidden = static_cast<int64_t>(config.hidden_size);
    const int64_t kv_dim = static_cast<int64_t>(config.num_key_value_heads * config.head_dim);
    const int64_t inter = static_cast<int64_t>(config.intermediate_size);
    const std::vector<ProjSpec> specs = {
        {"self_attn.q_proj", hidden, hidden},
        {"self_attn.k_proj", hidden, kv_dim},
        {"self_attn.v_proj", hidden, kv_dim},
        {"self_attn.o_proj", hidden, hidden},
        {"mlp.gate_proj", hidden, inter},
        {"mlp.up_proj", hidden, inter},
        {"mlp.down_proj", inter, hidden},
    };

    size_t validated = 0;
    for (size_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        for (const auto& spec : specs) {
            const std::string prefix = "model.layers." + std::to_string(layer) + "." + spec.name;
            const std::string where = prefix;

            std::vector<std::string> required = {
                prefix + ".U_packed", prefix + ".V_packed",
                prefix + ".U_shape", prefix + ".V_shape",
                prefix + "._split_dim_final",
                prefix + ".u1", prefix + ".u2", prefix + ".v1", prefix + ".v2",
            };
            if (residual) {
                required.insert(required.end(), {
                    prefix + ".U_R_packed", prefix + ".V_R_packed",
                    prefix + ".U_R_shape", prefix + ".V_R_shape",
                    prefix + ".u1_R", prefix + ".u2_R", prefix + ".v1_R", prefix + ".v2_R",
                });
            }
            for (const auto& name : required) {
                if (!weights.has_tensor(name)) {
                    err = where + ": missing required LittleBit tensor " + name;
                    return false;
                }
            }

            std::vector<int64_t> u_shape, v_shape;
            if (!read_i64_pair(weights, prefix + ".U_shape", u_shape, err) ||
                !read_i64_pair(weights, prefix + ".V_shape", v_shape, err)) {
                return false;
            }
            int64_t split = 0;
            if (!read_i64_scalar(weights, prefix + "._split_dim_final", split, err)) return false;

            if (split <= 0 || u_shape[1] != split || v_shape[0] != split) {
                err = where + ": split metadata disagrees: U_shape=" + shape_to_string(u_shape) +
                      ", V_shape=" + shape_to_string(v_shape) + ", _split_dim_final=" +
                      std::to_string(split);
                return false;
            }
            if (u_shape[0] != spec.out || v_shape[1] != spec.in) {
                err = where + ": logical shape mismatch: U_shape=" + shape_to_string(u_shape) +
                      ", V_shape=" + shape_to_string(v_shape) + ", expected out=" +
                      std::to_string(spec.out) + ", in=" + std::to_string(spec.in);
                return false;
            }

            const int64_t u_words = (split + 31) / 32;
            const int64_t v_words = (spec.in + 31) / 32;
            if (!check_packed_tensor(weights, prefix + ".U_packed", u_shape[0], u_words, err) ||
                !check_packed_tensor(weights, prefix + ".V_packed", v_shape[0], v_words, err)) {
                return false;
            }
            if (!check_float_tensor(weights, prefix + ".u1", u_shape[0], err) ||
                !check_float_tensor(weights, prefix + ".u2", split, err) ||
                !check_float_tensor(weights, prefix + ".v1", split, err) ||
                !check_float_tensor(weights, prefix + ".v2", spec.in, err)) {
                return false;
            }

            if (residual) {
                std::vector<int64_t> u_r_shape, v_r_shape;
                if (!read_i64_pair(weights, prefix + ".U_R_shape", u_r_shape, err) ||
                    !read_i64_pair(weights, prefix + ".V_R_shape", v_r_shape, err)) {
                    return false;
                }
                if (!vectors_equal(u_r_shape, u_shape) || !vectors_equal(v_r_shape, v_shape)) {
                    err = where + ": residual shape metadata differs from primary: U_R_shape=" +
                          shape_to_string(u_r_shape) + ", V_R_shape=" + shape_to_string(v_r_shape);
                    return false;
                }
                if (!check_packed_tensor(weights, prefix + ".U_R_packed", u_shape[0], u_words, err) ||
                    !check_packed_tensor(weights, prefix + ".V_R_packed", v_shape[0], v_words, err)) {
                    return false;
                }
                if (!check_float_tensor(weights, prefix + ".u1_R", u_shape[0], err) ||
                    !check_float_tensor(weights, prefix + ".u2_R", split, err) ||
                    !check_float_tensor(weights, prefix + ".v1_R", split, err) ||
                    !check_float_tensor(weights, prefix + ".v2_R", spec.in, err)) {
                    return false;
                }
            }
            ++validated;
        }
    }

    std::cout << "[MODEL] LittleBit validation OK: " << validated << " linears, residual="
              << (residual ? "yes" : "no") << std::endl;
    return true;
}

bool validate_littlebit_checkpoint(
    const std::string& model_dir, const ModelConfig& config,
    const SafeTensorsFile& weights, LittleBitCheckpointMeta& meta, std::string& err
) {
    const bool has_littlebit = has_littlebit_packed_tensors(weights);
    if (!has_littlebit) {
        if (has_plain_qwen3_linears(weights)) {
            err = "Plain Qwen3 checkpoint detected. This loader requires a converted single-file "
                  "LittleBit checkpoint (I32 *_U_packed/*_V_packed tensors plus U_shape/V_shape metadata); "
                  "plain or sharded Qwen3 weights are not supported and no converter is bundled.";
        } else {
            err = "No LittleBit *_U_packed/*_V_packed tensors found; not a LittleBit checkpoint.";
        }
        return false;
    }

    if (!validate_littlebit_config(model_dir, meta, err)) return false;
    return validate_littlebit_tensors(weights, config, meta.residual, err);
}

} // namespace

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
    std::ifstream index_in(model_dir + "/model.safetensors.index.json");
    if (index_in.is_open()) {
        std::cerr << "[MODEL] Error: Plain/sharded Qwen3 checkpoint detected (model.safetensors.index.json). "
                     "This loader requires a converted single-file LittleBit checkpoint; "
                     "sharded weights are not supported and no converter is bundled." << std::endl;
        return false;
    }
    if (!weights_file.open(weights_path)) {
        std::cerr << "[MODEL] Error: Cannot open weights file " << weights_path << std::endl;
        return false;
    }

    LittleBitCheckpointMeta lb_meta;
    std::string validation_error;
    if (!validate_littlebit_checkpoint(model_dir, config, weights_file, lb_meta, validation_error)) {
        std::cerr << "[MODEL] Error: " << validation_error << std::endl;
        weights_file.close();
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
    std::function<bool(int32_t, const std::string&)> on_token,
    std::function<bool()> should_cancel
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
        // Defaulted probe: existing callers pass nothing and are unaffected.
        // The bridge uses this to observe an unload requested during prefill.
        if (should_cancel && should_cancel()) {
            return metrics;
        }
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
        if (should_cancel && should_cancel()) {
            break;
        }
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
