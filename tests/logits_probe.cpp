#include "littlebit/model.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// Usage: logits_probe <model_dir> <comma,separated,token,ids> <logits.bin> [num_layers]
// Writes the logits after every supplied token as float32 [tokens][vocab_size].
int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: logits_probe <model_dir> <token_ids> <logits.bin> [num_layers]\n";
        return 2;
    }
    const std::string model_dir = argv[1];
    const size_t num_layers = (argc > 4) ? std::stoul(argv[4]) : 0;

    littlebit::Qwen3Model model;
    if (!model.load(model_dir, 64, num_layers)) return 3;

    std::vector<int32_t> tokens;
    std::stringstream ss(argv[2]);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) tokens.push_back(std::stoi(item));
    }
    if (tokens.empty()) return 4;

    std::ofstream out(argv[3], std::ios::binary);
    for (size_t p = 0; p < tokens.size(); ++p) {
        const float* logits = model.forward_token(tokens[p], p);
        out.write(reinterpret_cast<const char*>(logits),
                  static_cast<std::streamsize>(model.config.vocab_size * sizeof(float)));
    }
    std::cout << "tokens=" << tokens.size() << " vocab=" << model.config.vocab_size
              << " layers=" << model.config.num_hidden_layers << '\n';
    return 0;
}
