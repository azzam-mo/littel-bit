#include "littlebit/tokenizer.h"
#include "littlebit/unicode_ranges.h"
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

static bool utf8_next(const std::string& s, size_t pos, uint32_t& cp, size_t& len) {
    if (pos >= s.size()) return false;
    unsigned char c = static_cast<unsigned char>(s[pos]);
    if (c < 0x80) { cp = c; len = 1; return true; }
    if ((c & 0xE0) == 0xC0 && pos + 1 < s.size()) {
        cp = ((c & 0x1F) << 6) | (static_cast<unsigned char>(s[pos + 1]) & 0x3F);
        len = 2; return true;
    }
    if ((c & 0xF0) == 0xE0 && pos + 2 < s.size()) {
        cp = ((c & 0x0F) << 12) | ((static_cast<unsigned char>(s[pos + 1]) & 0x3F) << 6) |
             (static_cast<unsigned char>(s[pos + 2]) & 0x3F);
        len = 3; return true;
    }
    if ((c & 0xF8) == 0xF0 && pos + 3 < s.size()) {
        cp = ((c & 0x07) << 18) | ((static_cast<unsigned char>(s[pos + 1]) & 0x3F) << 12) |
             ((static_cast<unsigned char>(s[pos + 2]) & 0x3F) << 6) |
             (static_cast<unsigned char>(s[pos + 3]) & 0x3F);
        len = 4; return true;
    }
    cp = c; len = 1; return true;
}

static bool match_contraction(const std::string& s, size_t pos, const char* suffix) {
    if (pos >= s.size() || s[pos] != '\'') return false;
    for (size_t i = 0; suffix[i]; ++i) {
        if (pos + 1 + i >= s.size()) return false;
        unsigned char got = static_cast<unsigned char>(s[pos + 1 + i]);
        unsigned char want = static_cast<unsigned char>(suffix[i]);
        if (got >= 'A' && got <= 'Z') got = static_cast<unsigned char>(got - 'A' + 'a');
        if (want >= 'A' && want <= 'Z') want = static_cast<unsigned char>(want - 'A' + 'a');
        if (got != want) return false;
    }
    return true;
}

bool Tokenizer::load(const std::string& vocab_file, const std::string& merges_file) {
    token_to_id_.clear();
    id_to_token_.clear();
    bpe_ranks_.clear();
    special_tokens_.clear();

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

bool Tokenizer::load_from_json(const std::string& tokenizer_json_file) {
    std::ifstream in(tokenizer_json_file);
    if (!in.is_open()) {
        std::cerr << "Tokenizer: Cannot open tokenizer.json: " << tokenizer_json_file << std::endl;
        return false;
    }

    try {
        nlohmann::json root;
        in >> root;
        const auto& model = root.at("model");
        if (model.value("type", "") != "BPE") {
            std::cerr << "Tokenizer: Unsupported model type in tokenizer.json" << std::endl;
            return false;
        }

        token_to_id_.clear();
        id_to_token_.clear();
        bpe_ranks_.clear();
        special_tokens_.clear();

        for (auto it = model.at("vocab").begin(); it != model.at("vocab").end(); ++it) {
            std::string token = it.key();
            int32_t id = it.value().get<int32_t>();
            token_to_id_[token] = id;
            id_to_token_[id] = token;
        }

        int32_t rank = 0;
        for (const auto& merge : model.at("merges")) {
            std::string pair;
            if (merge.is_array() && merge.size() == 2) {
                pair = merge[0].get<std::string>() + " " + merge[1].get<std::string>();
            } else if (merge.is_string()) {
                pair = merge.get<std::string>();
            }
            if (!pair.empty()) bpe_ranks_[pair] = rank++;
        }

        if (root.contains("added_tokens")) {
            for (const auto& added : root.at("added_tokens")) {
                std::string content = added.value("content", "");
                int32_t id = added.value("id", -1);
                if (!content.empty() && id >= 0) {
                    special_tokens_[content] = id;
                    if (!token_to_id_.count(content)) token_to_id_[content] = id;
                    if (!id_to_token_.count(id)) id_to_token_[id] = content;
                }
            }
        }

        if (root.contains("bos_token_id")) bos_id = root.at("bos_token_id").get<int32_t>();
        if (root.contains("eos_token_id")) eos_id = root.at("eos_token_id").get<int32_t>();
        return !token_to_id_.empty();
    } catch (const std::exception& e) {
        std::cerr << "Tokenizer: JSON error in tokenizer.json: " << e.what() << std::endl;
        return false;
    }
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

    // Added/special tokens must not go through byte-level BPE.
    static const struct { const char* text; int32_t id; } kFallbackSpecial[] = {
        {"<|im_start|>", 151644},
        {"<|im_end|>",   151645},
        {"<|endoftext|>",151643},
        {"<|object_ref_start|>", 151646},
        {"<|object_ref_end|>", 151647},
    };

    size_t pos = 0;
    while (pos < text.size()) {
        size_t best_pos = std::string::npos;
        std::string best_text;
        int32_t best_id = 0;

        auto consider = [&](const std::string& token, int32_t id) {
            size_t found = text.find(token, pos);
            if (found != std::string::npos && found < best_pos) {
                best_pos = found;
                best_text = token;
                best_id = id;
            }
        };
        for (const auto& sp : special_tokens_) consider(sp.first, sp.second);
        if (special_tokens_.empty()) {
            for (const auto& sp : kFallbackSpecial) consider(sp.text, sp.id);
        }

        if (best_pos == std::string::npos) {
            encode_plain(text.substr(pos), token_ids);
            break;
        }
        if (best_pos > pos) {
            encode_plain(text.substr(pos, best_pos - pos), token_ids);
        }
        auto it = token_to_id_.find(best_text);
        token_ids.push_back(it != token_to_id_.end() ? it->second : best_id);
        pos = best_pos + best_text.size();
    }

    return token_ids;
}

std::vector<std::string> Tokenizer::pretokenize(const std::string& text) const {
    // Mirrors the Qwen3 tokenizer.json Split pre-tokenizer:
    // (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|
    //  ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
    std::vector<std::string> out;
    size_t i = 0;
    while (i < text.size()) {
        uint32_t cp = 0;
        size_t len = 0;
        utf8_next(text, i, cp, len);

        // Contractions
        static const char* kContractions[] = {"s", "t", "re", "ve", "m", "ll", "d"};
        if (cp == '\'') {
            bool matched = false;
            for (const char* suffix : kContractions) {
                if (match_contraction(text, i, suffix)) {
                    size_t n = 1 + std::strlen(suffix);
                    out.push_back(text.substr(i, n));
                    i += n;
                    matched = true;
                    break;
                }
            }
            if (matched) continue;
        }

        // Optional non-letter/number (except newline) followed by one or more letters.
        size_t p = i;
        if (cp != '\r' && cp != '\n' && !unicode::is_letter_or_number(cp)) {
            p += len;
        }
        if (p < text.size()) {
            uint32_t cp2 = 0;
            size_t len2 = 0;
            utf8_next(text, p, cp2, len2);
            if (unicode::is_letter(cp2)) {
                p += len2;
                while (p < text.size()) {
                    uint32_t c = 0;
                    size_t l = 0;
                    utf8_next(text, p, c, l);
                    if (!unicode::is_letter(c)) break;
                    p += l;
                }
                out.push_back(text.substr(i, p - i));
                i = p;
                continue;
            }
        }

        // Single number.
        if (unicode::is_number(cp)) {
            out.push_back(text.substr(i, len));
            i += len;
            continue;
        }

        // Optional ASCII space, then punctuation/symbol run, then optional newlines.
        size_t sym = i;
        if (cp == ' ' && i + 1 < text.size()) {
            uint32_t next = 0;
            size_t nlen = 0;
            utf8_next(text, i + 1, next, nlen);
            if (!unicode::is_space(next) && !unicode::is_letter_or_number(next)) sym = i + 1;
        }
        if (sym < text.size()) {
            uint32_t sc = 0;
            size_t sl = 0;
            utf8_next(text, sym, sc, sl);
            if (!unicode::is_space(sc) && !unicode::is_letter_or_number(sc)) {
                size_t q = sym;
                while (q < text.size()) {
                    uint32_t c = 0;
                    size_t l = 0;
                    utf8_next(text, q, c, l);
                    if (unicode::is_space(c) || unicode::is_letter_or_number(c)) break;
                    q += l;
                }
                while (q < text.size() && (text[q] == '\r' || text[q] == '\n')) ++q;
                out.push_back(text.substr(i, q - i));
                i = q;
                continue;
            }
        }

        // Whitespace: \s*[\r\n]+, then \s+(?!\S), then plain \s+.
        if (unicode::is_space(cp)) {
            size_t q = i;
            size_t last_newline_end = i;
            std::vector<size_t> boundaries;
            boundaries.push_back(i);
            while (q < text.size()) {
                uint32_t c = 0;
                size_t l = 0;
                utf8_next(text, q, c, l);
                if (!unicode::is_space(c)) break;
                q += l;
                boundaries.push_back(q);
                if (c == '\r' || c == '\n') last_newline_end = q;
            }
            if (last_newline_end > i) {
                out.push_back(text.substr(i, last_newline_end - i));
                i = last_newline_end;
                continue;
            }
            if (q == text.size()) {
                out.push_back(text.substr(i, q - i));
                i = q;
                continue;
            }
            // \s+(?!\S): leave the last whitespace character for the next token.
            if (boundaries.size() >= 3) {
                size_t emit_end = boundaries[boundaries.size() - 2];
                out.push_back(text.substr(i, emit_end - i));
                i = emit_end;
                continue;
            }
            out.push_back(text.substr(i, q - i));
            i = q;
            continue;
        }

        // Fallback: one UTF-8 character.
        out.push_back(text.substr(i, len));
        i += len;
    }
    return out;
}

void Tokenizer::encode_plain(const std::string& text, std::vector<int32_t>& out) const {
    if (text.empty()) return;

    const auto& byte_encoder = get_byte_encoder();
    for (const std::string& piece : pretokenize(text)) {
        // Convert this pre-token's UTF-8 bytes to the byte-level BPE alphabet.
        std::string bpe_str;
        for (unsigned char byte_val : piece) {
            bpe_str += byte_encoder.at(byte_val);
        }

        for (const auto& tok : bpe(bpe_str)) {
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
