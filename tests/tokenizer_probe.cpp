#include "littlebit/tokenizer.h"

#include <iostream>
#include <string>

// Usage: tokenizer_probe <vocab.json> <merges.txt> <text> [tokenizer.json]
int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: tokenizer_probe <vocab.json> <merges.txt> <text> [tokenizer.json]\n";
        return 2;
    }
    littlebit::Tokenizer tokenizer;
    bool loaded = (argc > 4) ? tokenizer.load_from_json(argv[4])
                             : tokenizer.load(argv[1], argv[2]);
    if (!loaded) return 3;
    for (int32_t id : tokenizer.encode(argv[3], false)) {
        std::cout << id << ' ';
    }
    std::cout << '\n';
    return 0;
}
