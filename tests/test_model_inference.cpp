#include "littlebit/model.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

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

    std::cout << "All full-path inference tests passed successfully!" << std::endl;
    return 0;
}
