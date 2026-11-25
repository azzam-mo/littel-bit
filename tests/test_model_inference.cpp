#include "littlebit/model.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace {

// Rewrite one safetensors header field in a copy of the fixture. Used to prove
// that untrusted Files.app metadata is validated before mmap pointers are used.
std::string rewritten_fixture(const std::string& source, const std::string& suffix) {
    namespace fs = std::filesystem;
    const fs::path target_dir = fs::temp_directory_path() /
        ("littlebit-phase12-" + suffix + "-" +
         std::to_string(static_cast<long long>(
             fs::last_write_time(source).time_since_epoch().count())));
    fs::create_directories(target_dir);
    for (const auto& entry : fs::directory_iterator(source)) {
        fs::copy(entry.path(), target_dir / entry.path().filename(),
                 fs::copy_options::overwrite_existing);
    }

    const fs::path weights = target_dir / "model.safetensors";
    std::ifstream in(weights, std::ios::binary);
    uint64_t header_length = 0;
    in.read(reinterpret_cast<char*>(&header_length), sizeof(header_length));
    std::string header(static_cast<size_t>(header_length), '\0');
    in.read(header.data(), static_cast<std::streamsize>(header.size()));
    auto json = nlohmann::json::parse(header);
    if (suffix == "bad-shape") {
        json["lm_head.weight"]["shape"] = {1, 1};
    } else if (suffix == "no-head") {
        json.erase("lm_head.weight");
    }
    const std::string new_header = json.dump();
    const std::vector<char> tensor_bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();

    const uint64_t new_header_length = new_header.size();
    std::ofstream out(weights, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(&new_header_length), sizeof(new_header_length));
    out.write(new_header.data(), static_cast<std::streamsize>(new_header.size()));
    out.write(tensor_bytes.data(), static_cast<std::streamsize>(tensor_bytes.size()));
    return target_dir.string();
}

} // namespace

int main(int argc, char** argv) {
    std::string model_dir = (argc > 1) ? argv[1] : "tiny-model";
    std::cout << "[TEST] Full inference path with synthetic model: " << model_dir << std::endl;

    littlebit::Qwen3Model model;
    const bool loaded = model.load(model_dir, 32);
    assert(loaded);
    if (!loaded) return 1;
    assert(model.config.num_hidden_layers == 1);
    assert(model.config.num_attention_heads == 4);
    assert(model.config.num_key_value_heads == 2);
    assert(model.config.head_dim == 8);
    assert(model.last_load_time_ms > 0.0);
    assert(model.embed_tokens_mmap != nullptr && model.lm_head_mmap != nullptr);

    // Streaming + early stop: callback returns false on the third token.
    int emitted = 0;
    auto metrics = model.generate(
        "hello", 8, 0.0f, 1.0f,
        [&emitted](int32_t token_id, const std::string& token_str) -> bool {
            assert(token_id >= 0);
            (void)token_id;
            (void)token_str;
            return ++emitted < 3;
        }
    );

    assert(emitted == 3);
    assert(metrics.tokens_generated == 3);
    assert(metrics.ttft_ms >= 0.0 && std::isfinite(metrics.ttft_ms));
    assert(metrics.tokens_per_second >= 0.0 && std::isfinite(metrics.tokens_per_second));
    // Tiny fixture has a sub-MB KV cache; assert the underlying allocation instead.
    assert(!model.kv_caches.empty() && !model.kv_caches[0].k_data.empty());
    assert(metrics.load_time_ms > 0.0);
    for (float v : model.logits_buf) {
        assert(std::isfinite(v));
        (void)v;
    }
    for (const auto& cache : model.kv_caches) {
        assert(cache.current_len <= cache.max_seq_len);
        (void)cache;
    }
    std::cout << "  ✓ load " << metrics.load_time_ms << " ms, ttft " << metrics.ttft_ms
              << " ms, " << metrics.tokens_per_second << " tok/s, KV " << metrics.kv_cache_mb
              << " MB, streamed " << emitted << " tokens" << std::endl;

    // Cache must be reusable after clear_cache().
    model.clear_cache();
    auto metrics2 = model.generate("again", 2, 0.0f, 1.0f, nullptr);
    assert(metrics2.tokens_generated <= 2);
    (void)metrics2;
    assert(model.kv_caches[0].current_len >= 1);
    std::cout << "  ✓ second generation after clear_cache, KV length = "
              << model.kv_caches[0].current_len << std::endl;

    // Phase 12 regressions: mmap pointers must never survive a failed load or
    // point at a tensor whose header metadata is inconsistent with config.json.
    const std::string bad_shape = rewritten_fixture(model_dir, "bad-shape");
    littlebit::Qwen3Model invalid;
    assert(!invalid.load(bad_shape, 32));
    assert(invalid.embed_tokens_mmap == nullptr);
    assert(invalid.lm_head_mmap == nullptr);

    littlebit::Qwen3Model reused;
    assert(reused.load(model_dir, 32));
    assert(reused.lm_head_mmap != nullptr);
    assert(!reused.load(bad_shape, 32));
    assert(reused.embed_tokens_mmap == nullptr && reused.lm_head_mmap == nullptr);

    const std::string no_head = rewritten_fixture(model_dir, "no-head");
    littlebit::Qwen3Model missing_head;
    assert(missing_head.load(no_head, 32));
    assert(missing_head.lm_head_mmap == nullptr);
    assert(missing_head.embed_tokens_mmap != nullptr);
    std::cout << "  ✓ invalid/optional mmap tensors are rejected without dangling pointers" << std::endl;

    std::cout << "All full-path inference tests passed successfully!" << std::endl;
    return 0;
}
