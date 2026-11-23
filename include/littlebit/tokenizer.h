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
    int32_t im_start_id = 151644;
    int32_t im_end_id = 151645;

    Tokenizer() = default;

    // Load from vocab.json and merges.txt
    bool load(const std::string& vocab_file, const std::string& merges_file);

    // Encode text to token IDs. Qwen control strings are recognised verbatim.
    std::vector<int32_t> encode(const std::string& text, bool add_bos = false) const;

    // Qwen3 chat template: <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
    std::string apply_chat_template(
        const std::string& user_message,
        const std::string& system_message = ""
    ) const;

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
    void encode_plain(const std::string& text, std::vector<int32_t>& out) const;
};

} // namespace littlebit
