#include "littlebit/tokenizer.h"
#include "nlohmann/json.hpp"

#include <fstream>
#include <sstream>
#include <iostream>
#include <regex>
#include <algorithm>
#include <cstring>

namespace littlebit {

// Byte to unicode mapping (GPT-2 / Qwen byte level BPE standard)
static std::unordered_map<uint8_t, std::string> create_byte_to_unicode() {
    std::unordered_map<uint8_t, std::string> byte_to_unicode;
    std::vector<int> bs;
    for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
    for (int b = 161; b <= 172; ++b) bs.push_back(b);
    for (int b = 174; b <= 255; ++b) bs.push_back(b);

    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }

    for (size_t i = 0; i < bs.size(); ++i) {
        uint8_t b = static_cast<uint8_t>(bs[i]);
        int code = cs[i];
        // Convert codepoint to UTF-8
        std::string s;
        if (code <= 0x7F) {
            s += static_cast<char>(code);
        } else if (code <= 0x7FF) {
            s += static_cast<char>(0xC0 | ((code >> 6) & 0x1F));
            s += static_cast<char>(0x80 | (code & 0x3F));
        } else {
            s += static_cast<char>(0xE0 | ((code >> 12) & 0x0F));
            s += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
            s += static_cast<char>(0x80 | (code & 0x3F));
        }
        byte_to_unicode[b] = s;
    }
    return byte_to_unicode;
}

static const auto& get_byte_encoder() {
    static auto byte_encoder = create_byte_to_unicode();
    return byte_encoder;
}

static const auto& get_byte_decoder() {
    static auto byte_decoder = []() {
        std::unordered_map<std::string, uint8_t> dec;
        for (const auto& [b, u] : get_byte_encoder()) {
            dec[u] = b;
        }
        return dec;
    }();
    return byte_decoder;
}

bool Tokenizer::load(const std::string& vocab_file, const std::string& merges_file) {
    token_to_id_.clear();
    id_to_token_.clear();
    bpe_ranks_.clear();

    // 1. Load vocab.json
    std::ifstream v_in(vocab_file);
    if (!v_in.is_open()) {
        std::cerr << "Tokenizer: Cannot open vocab file: " << vocab_file << std::endl;
        return false;
    }

    try {
        nlohmann::json v_json;
        v_in >> v_json;
        for (auto it = v_json.begin(); it != v_json.end(); ++it) {
            std::string token = it.key();
            int32_t id = it.value().get<int32_t>();
            token_to_id_[token] = id;
            id_to_token_[id] = token;
        }
    } catch (const std::exception& e) {
        std::cerr << "Tokenizer: JSON error in vocab: " << e.what() << std::endl;
        return false;
    }

    // 2. Load merges.txt
    std::ifstream m_in(merges_file);
    if (!m_in.is_open()) {
        std::cerr << "Tokenizer: Cannot open merges file: " << merges_file << std::endl;
        return false;
    }

    std::string line;
    int32_t rank = 0;
    while (std::getline(m_in, line)) {
        if (line.empty() || line[0] == '#') continue;
        bpe_ranks_[line] = rank++;
    }

    return true;
}

std::vector<std::string> Tokenizer::bpe(const std::string& token) const {
    if (token.empty()) return {};

    std::vector<std::string> word;
    // Split into individual UTF-8 characters
    for (size_t i = 0; i < token.size();) {
        unsigned char c = token[i];
        size_t len = 1;
        if (c >= 0xF0) len = 4;
        else if (c >= 0xE0) len = 3;
        else if (c >= 0xC0) len = 2;
        len = std::min(len, token.size() - i);
        word.push_back(token.substr(i, len));
        i += len;
    }

    if (word.size() <= 1) return word;

    while (true) {
        int min_rank = -1;
        int best_idx = -1;

        for (size_t i = 0; i + 1 < word.size(); ++i) {
            std::string pair = word[i] + " " + word[i + 1];
            auto it = bpe_ranks_.find(pair);
            if (it != bpe_ranks_.end()) {
                if (min_rank == -1 || it->second < min_rank) {
                    min_rank = it->second;
                    best_idx = static_cast<int>(i);
                }
            }
        }

        if (best_idx == -1) break;

        // Merge the best pair
        std::vector<std::string> next_word;
        for (size_t i = 0; i < word.size(); ++i) {
            if (static_cast<int>(i) == best_idx) {
                next_word.push_back(word[i] + word[i + 1]);
                i++; // skip next
            } else {
                next_word.push_back(word[i]);
            }
        }
        word = std::move(next_word);
        if (word.size() <= 1) break;
    }

    return word;
}

std::vector<int32_t> Tokenizer::encode(const std::string& text, bool add_bos) const {
    std::vector<int32_t> token_ids;
    if (add_bos) {
        token_ids.push_back(bos_id);
    }
    if (text.empty()) return token_ids;

    // Qwen control tokens must not go through byte-level BPE.
    static const struct { const char* text; int32_t id; } kSpecial[] = {
        {"<|im_start|>", 151644},
        {"<|im_end|>",   151645},
        {"<|endoftext|>",151643},
        {"<|im_sep|>",   151647},
    };

    size_t pos = 0;
    while (pos < text.size()) {
        size_t best_pos = std::string::npos;
        const char* best_text = nullptr;
        int32_t best_id = 0;
        for (const auto& sp : kSpecial) {
            size_t found = text.find(sp.text, pos);
            if (found != std::string::npos && found < best_pos) {
                best_pos = found;
                best_text = sp.text;
                best_id = sp.id;
            }
        }

        if (best_pos == std::string::npos) {
            encode_plain(text.substr(pos), token_ids);
            break;
        }
        if (best_pos > pos) {
            encode_plain(text.substr(pos, best_pos - pos), token_ids);
        }
        // Prefer the id from the loaded vocabulary when available.
        auto it = token_to_id_.find(best_text);
        token_ids.push_back(it != token_to_id_.end() ? it->second : best_id);
        pos = best_pos + std::strlen(best_text);
    }

    return token_ids;
}

void Tokenizer::encode_plain(const std::string& text, std::vector<int32_t>& out) const {
    if (text.empty()) return;

    // Convert raw UTF-8 bytes to byte-level BPE string
    const auto& byte_encoder = get_byte_encoder();
    std::string bpe_str;
    for (unsigned char byte_val : text) {
        bpe_str += byte_encoder.at(byte_val);
    }

    auto bpe_tokens = bpe(bpe_str);
    for (const auto& tok : bpe_tokens) {
        auto it = token_to_id_.find(tok);
        if (it != token_to_id_.end()) {
            out.push_back(it->second);
        } else {
            for (size_t i = 0; i < tok.size();) {
                unsigned char c = tok[i];
                size_t len = (c >= 0xF0) ? 4 : (c >= 0xE0) ? 3 : (c >= 0xC0) ? 2 : 1;
                len = std::min(len, tok.size() - i);
                std::string sub = tok.substr(i, len);
                auto sub_it = token_to_id_.find(sub);
                if (sub_it != token_to_id_.end()) {
                    out.push_back(sub_it->second);
                }
                i += len;
            }
        }
    }
}

std::string Tokenizer::apply_chat_template(
    const std::string& user_message,
    const std::string& system_message
) const {
    std::string prompt;
    if (!system_message.empty()) {
        prompt = "<|im_start|>system\n" + system_message + "<|im_end|>\n";
    }
    prompt += "<|im_start|>user\n" + user_message + "<|im_end|>\n";
    prompt += "<|im_start|>assistant\n";
    return prompt;
}

std::string Tokenizer::decode(int32_t token_id) const {
    auto it = id_to_token_.find(token_id);
    if (it == id_to_token_.end()) return "";

    const std::string& tok = it->second;
    const auto& byte_decoder = get_byte_decoder();

    std::string raw_bytes;
    for (size_t i = 0; i < tok.size();) {
        unsigned char c = tok[i];
        size_t len = (c >= 0xF0) ? 4 : (c >= 0xE0) ? 3 : (c >= 0xC0) ? 2 : 1;
        len = std::min(len, tok.size() - i);
        std::string char_str = tok.substr(i, len);
        auto dec_it = byte_decoder.find(char_str);
        if (dec_it != byte_decoder.end()) {
            raw_bytes += static_cast<char>(dec_it->second);
        } else {
            raw_bytes += char_str;
        }
        i += len;
    }
    return raw_bytes;
}

std::string Tokenizer::decode(const std::vector<int32_t>& tokens) const {
    std::string result;
    for (int32_t t : tokens) {
        result += decode(t);
    }
    return result;
}

} // namespace littlebit
