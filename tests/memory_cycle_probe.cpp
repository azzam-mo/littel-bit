#include "littlebit/model.h"

#include <cstddef>
#include <iostream>
#include <string>

#if defined(__linux__)
#include <sys/resource.h>
#endif

namespace {

std::size_t peak_rss_mb() {
#if defined(__linux__)
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
        return static_cast<std::size_t>(ru.ru_maxrss / 1024);
    }
#endif
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: memory_cycle_probe <model_dir> [cycles]\n";
        return 2;
    }
    const std::string model_dir = argv[1];
    const int cycles = argc > 2 ? std::stoi(argv[2]) : 3;
    if (cycles < 1) {
        std::cerr << "cycles must be >= 1\n";
        return 2;
    }

    for (int cycle = 0; cycle < cycles; ++cycle) {
        {
            littlebit::Qwen3Model model;
            if (!model.load(model_dir, 32)) {
                std::cerr << "model load failed in cycle " << cycle << '\n';
                return 3;
            }
            const std::size_t after_load = littlebit::Qwen3Model::get_current_rss_mb();
            for (std::size_t pos = 0; pos < 8; ++pos) {
                if (model.forward_token(static_cast<int32_t>(pos), pos) == nullptr) {
                    std::cerr << "forward failed in cycle " << cycle << " at pos " << pos << '\n';
                    return 4;
                }
            }
            const std::size_t after_prefill = littlebit::Qwen3Model::get_current_rss_mb();
            model.clear_cache();
            const std::size_t after_clear = littlebit::Qwen3Model::get_current_rss_mb();
            std::cout << "[CYCLE] cycle=" << cycle
                      << " rss_after_load_mb=" << after_load
                      << " rss_after_prefill_mb=" << after_prefill
                      << " rss_after_clear_mb=" << after_clear
                      << " peak_rss_mb=" << peak_rss_mb() << '\n';
        }
        std::cout << "[CYCLE] cycle=" << cycle
                  << " rss_after_scope_exit_mb=" << littlebit::Qwen3Model::get_current_rss_mb()
                  << " peak_rss_mb=" << peak_rss_mb() << '\n';
    }

    std::cout << "[CYCLE] completed cycles=" << cycles
              << " final_rss_mb=" << littlebit::Qwen3Model::get_current_rss_mb()
              << " peak_rss_mb=" << peak_rss_mb() << '\n';
    return 0;
}
