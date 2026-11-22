#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace littlebit {

class Tokenizer {
public:
    int32_t bos_id = 151643;
    int32_t eos_id = 151645;
    int32_t pad_id = 151643;

    Tokenizer() = default;

    // Load from vocab.json and merges.txt
    bool load(const std::string& vocab_file, const std::string& merges_file);

    // Encode text to token IDs
    std::vector<int32_t> encode(const std::string& text, bool add_bos = false) const;

    // Decode token IDs to text
    std::string decode(int32_t token_id) const;
    std::string decode(const std::vector<int32_t>& tokens) const;

    size_t vocab_size() const { return token_to_id_.size(); }
    bool is_eos(int32_t token_id) const { return token_id == eos_id; }

private:
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::unordered_map<int32_t, std::string> id_to_token_;
    std::unordered_map<std::string, int32_t> bpe_ranks_;

    // Helper functions for byte-level BPE
    std::vector<std::string> bpe(const std::string& token) const;
};

} // namespace littlebit
