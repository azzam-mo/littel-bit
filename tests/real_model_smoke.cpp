#include "littlebit/model.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

// Usage: real_model_smoke <model_dir> [max_new_tokens] [max_seq_len]
// Loads a REAL littlebit-qwen3 checkpoint and runs one greedy generation.
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: real_model_smoke <model_dir> [max_new_tokens] [max_seq_len]\n";
        return 2;
    }
    const std::string model_dir = argv[1];
    const size_t max_new = (argc > 2) ? std::stoul(argv[2]) : 2;
    const size_t max_seq = (argc > 3) ? std::stoul(argv[3]) : 512;

    littlebit::Qwen3Model model;
    if (!model.load(model_dir, max_seq)) return 3;

    const size_t rss_before = littlebit::Qwen3Model::get_current_rss_mb();
    std::vector<int32_t> generated;
    auto metrics = model.generate(
        "مرحبا", max_new, 0.0f, 1.0f,
        [&generated](int32_t token_id, const std::string&) -> bool {
            generated.push_back(token_id);
            return true;
        }
    );

    if (metrics.tokens_generated == 0 || generated.empty()) {
        std::cerr << "No tokens generated\n";
        return 4;
    }
    for (float v : model.logits_buf) {
        if (!std::isfinite(v)) {
            std::cerr << "Non-finite logit detected\n";
            return 5;
        }
    }

    std::cout << "[SMOKE] prompt=مرحبا generated=" << metrics.tokens_generated
              << " tokens=";
    for (int32_t id : generated) std::cout << id << ',';
    std::cout << "\n[SMOKE] decoded='" << model.tokenizer.decode(generated) << "'\n";
    std::cout << "[SMOKE] load=" << metrics.load_time_ms << " ms ttft=" << metrics.ttft_ms
              << " ms rate=" << metrics.tokens_per_second << " tok/s kv=" << metrics.kv_cache_mb
              << " MB rss_before=" << rss_before << " MB rss_after=" << metrics.resident_memory_mb
              << " MB\n";
    std::cout << "All real-model smoke checks passed.\n";
    return 0;
}
