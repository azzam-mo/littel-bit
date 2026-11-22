#include "littlebit/tokenizer.h"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "[TEST] Running Tokenizer unit tests..." << std::endl;

    littlebit::Tokenizer tok;
    std::string vocab_path = "/home/mrdp_24193/.gemini/antigravity/scratch/model-configs/vocab.json";
    std::string merges_path = "/home/mrdp_24193/.gemini/antigravity/scratch/model-configs/merges.txt";

    bool ok = tok.load(vocab_path, merges_path);
    assert(ok && "Failed to load vocab/merges");
    std::cout << "  ✓ Tokenizer loaded " << tok.vocab_size() << " tokens." << std::endl;

    // Test 1: Simple English encoding & decoding
    {
        std::string text = "Hello world";
        auto tokens = tok.encode(text);
        assert(!tokens.empty());
        std::string decoded = tok.decode(tokens);
        std::cout << "  English: '" << text << "' -> tokens [";
        for (auto t : tokens) std::cout << t << " ";
        std::cout << "] -> '" << decoded << "'" << std::endl;
        assert(decoded == text);
        std::cout << "  ✓ English encode/decode roundtrip passed." << std::endl;
    }

    // Test 2: Arabic text encoding & decoding
    {
        std::string arabic_text = "مرحبا بالعالم";
        auto tokens = tok.encode(arabic_text);
        assert(!tokens.empty());
        std::string decoded = tok.decode(tokens);
        std::cout << "  Arabic: '" << arabic_text << "' -> tokens [";
        for (auto t : tokens) std::cout << t << " ";
        std::cout << "] -> '" << decoded << "'" << std::endl;
        assert(decoded == arabic_text);
        std::cout << "  ✓ Arabic encode/decode roundtrip passed." << std::endl;
    }

    std::cout << "All Tokenizer unit tests passed successfully!" << std::endl;
    return 0;
}
