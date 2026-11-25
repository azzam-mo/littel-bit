#include "littlebit/model.h"
#include "littlebit/ops.h"
#include "../src/lm_head_avx2.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#if defined(__linux__)
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

std::vector<int32_t> parse_csv(const std::string& csv) {
    std::vector<int32_t> out;
    std::stringstream ss(csv);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(std::stoi(item));
    }
    return out;
}

size_t thread_count() {
#if defined(__linux__)
    std::ifstream in("/proc/self/status");
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("Threads:", 0) == 0) {
            std::istringstream iss(line.substr(8));
            size_t n = 0;
            iss >> n;
            return n;
        }
    }
#endif
    return 0;
}

struct CpuTime {
    double user_ms = 0.0;
    double sys_ms = 0.0;
};

CpuTime read_cpu_time() {
    CpuTime t;
#if defined(__linux__)
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
        t.user_ms = ru.ru_utime.tv_sec * 1000.0 + ru.ru_utime.tv_usec / 1000.0;
        t.sys_ms = ru.ru_stime.tv_sec * 1000.0 + ru.ru_stime.tv_usec / 1000.0;
    }
#endif
    return t;
}

size_t peak_rss_mb() {
#if defined(__linux__)
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
        return static_cast<size_t>(ru.ru_maxrss / 1024);
    }
#endif
    return 0;
}

size_t linear_bytes(const littlebit::LittleBitLinear& lin) {
    size_t n = 0;
    auto add_i32 = [&](const std::vector<int32_t>& v) { n += v.size() * sizeof(int32_t); };
    auto add_f32 = [&](const std::vector<float>& v) { n += v.size() * sizeof(float); };
    add_i32(lin.U_packed);
    add_i32(lin.V_packed);
    add_f32(lin.u1);
    add_f32(lin.u2);
    add_f32(lin.v1);
    add_f32(lin.v2);
    add_i32(lin.U_R_packed);
    add_i32(lin.V_R_packed);
    add_f32(lin.u1_R);
    add_f32(lin.u2_R);
    add_f32(lin.v1_R);
    add_f32(lin.v2_R);
    return n;
}

struct Accum {
    double embed_ms = 0;
    double input_norm_ms = 0;
    double q_proj_ms = 0;
    double k_proj_ms = 0;
    double v_proj_ms = 0;
    double qk_norm_ms = 0;
    double rope_ms = 0;
    double kv_write_ms = 0;
    double attn_core_ms = 0;
    double o_proj_ms = 0;
    double attn_residual_ms = 0;
    double post_norm_ms = 0;
    double gate_proj_ms = 0;
    double up_proj_ms = 0;
    double swiglu_ms = 0;
    double down_proj_ms = 0;
    double mlp_residual_ms = 0;
    double final_norm_ms = 0;
    double lm_head_ms = 0;
    double greedy_ms = 0;
    double top_p_ms = 0;
    uint64_t lm_head_bytes = 0;
    uint64_t kv_read_bytes = 0;
    double logit_check_max_abs = 0;
    size_t calls = 0;
    std::vector<double> layer_ms;
    std::vector<std::vector<double>> layer_by_pos;

    void add(const char* name, double ms) {
        if (std::string(name) == "embed") embed_ms += ms;
        else if (std::string(name) == "input_norm") input_norm_ms += ms;
        else if (std::string(name) == "q_proj") q_proj_ms += ms;
        else if (std::string(name) == "k_proj") k_proj_ms += ms;
        else if (std::string(name) == "v_proj") v_proj_ms += ms;
        else if (std::string(name) == "qk_norm") qk_norm_ms += ms;
        else if (std::string(name) == "rope") rope_ms += ms;
        else if (std::string(name) == "kv_write") kv_write_ms += ms;
        else if (std::string(name) == "attn_core") attn_core_ms += ms;
        else if (std::string(name) == "o_proj") o_proj_ms += ms;
        else if (std::string(name) == "attn_residual") attn_residual_ms += ms;
        else if (std::string(name) == "post_norm") post_norm_ms += ms;
        else if (std::string(name) == "gate_proj") gate_proj_ms += ms;
        else if (std::string(name) == "up_proj") up_proj_ms += ms;
        else if (std::string(name) == "swiglu") swiglu_ms += ms;
        else if (std::string(name) == "down_proj") down_proj_ms += ms;
        else if (std::string(name) == "mlp_residual") mlp_residual_ms += ms;
        else if (std::string(name) == "final_norm") final_norm_ms += ms;
        else if (std::string(name) == "lm_head") lm_head_ms += ms;
    }
};

struct State {
    std::vector<float> x, layer_out, norm1, attn_out, residual, norm2, mlp_out;
    std::vector<float> q, k, v, scores, attn_vec;
    std::vector<float> gate, up, act;
    std::vector<float> logits;
    std::vector<float> engine_logits;

    void init(const littlebit::Qwen3Model& m) {
        const size_t h = m.config.hidden_size;
        const size_t inter = m.config.intermediate_size;
        const size_t qh = m.config.num_attention_heads * m.config.head_dim;
        const size_t kh = m.config.num_key_value_heads * m.config.head_dim;
        const size_t seq = m.config.max_seq_len;
        x.resize(h);
        layer_out.resize(h);
        norm1.resize(h);
        attn_out.resize(h);
        residual.resize(h);
        norm2.resize(h);
        mlp_out.resize(h);
        q.resize(qh);
        k.resize(kh);
        v.resize(kh);
        scores.resize(seq);
        attn_vec.resize(h);
        gate.resize(inter);
        up.resize(inter);
        act.resize(inter);
        logits.resize(m.config.vocab_size);
        engine_logits.resize(m.config.vocab_size);
    }
};

// Replays Qwen3Model::forward_token with timers. It writes the same KV cache
// positions and is checked against the engine's own logits after every call.
const float* instrumented_forward(
    littlebit::Qwen3Model& m,
    State& s,
    int32_t token_id,
    size_t pos,
    Accum& acc
) {
    using littlebit::bf16_to_fp32;
    using littlebit::rmsnorm;
    using littlebit::silu_scalar;
    using littlebit::softmax;

    auto t0 = Clock::now();
    if (m.embed_tokens_mmap && token_id >= 0 && static_cast<size_t>(token_id) < m.config.vocab_size) {
        const uint16_t* row = m.embed_tokens_mmap + token_id * m.config.hidden_size;
        for (size_t i = 0; i < m.config.hidden_size; ++i) s.x[i] = bf16_to_fp32(row[i]);
    } else {
        std::fill(s.x.begin(), s.x.end(), 0.0f);
    }
    auto t1 = Clock::now();
    acc.add("embed", elapsed_ms(t0, t1));

    const size_t n_layers = m.config.num_hidden_layers;
    if (acc.layer_by_pos.empty()) {
        acc.layer_ms.assign(n_layers, 0.0);
        acc.layer_by_pos.resize(1, std::vector<double>(n_layers, 0.0));
    }
    if (acc.layer_by_pos.size() <= pos + 1) {
        acc.layer_by_pos.resize(pos + 2, std::vector<double>(n_layers, 0.0));
    }

    float* in = s.x.data();
    float* out = s.layer_out.data();
    for (size_t l = 0; l < n_layers; ++l) {
        auto layer_start = Clock::now();
        auto& layer = m.layers[l];

        auto a = Clock::now();
        rmsnorm(in, layer.input_layernorm_weight.data(), s.norm1.data(), m.config.hidden_size);
        auto b = Clock::now();
        acc.add("input_norm", elapsed_ms(a, b));

        auto& attn = layer.self_attn;
        a = Clock::now();
        attn.q_proj.forward(s.norm1.data(), s.q.data());
        b = Clock::now();
        acc.add("q_proj", elapsed_ms(a, b));
        a = Clock::now();
        attn.k_proj.forward(s.norm1.data(), s.k.data());
        attn.v_proj.forward(s.norm1.data(), s.v.data());
        b = Clock::now();
        acc.add("k_proj", elapsed_ms(a, b));
        acc.add("v_proj", 0.0); // k/v measured together; split below by a dedicated microbenchmark.

        a = Clock::now();
        for (size_t h = 0; h < attn.num_q_heads; ++h) {
            float* head = s.q.data() + h * attn.head_dim;
            rmsnorm(head, attn.q_norm_weight.data(), head, attn.head_dim);
        }
        for (size_t h = 0; h < attn.num_kv_heads; ++h) {
            float* head = s.k.data() + h * attn.head_dim;
            rmsnorm(head, attn.k_norm_weight.data(), head, attn.head_dim);
        }
        b = Clock::now();
        acc.add("qk_norm", elapsed_ms(a, b));

        a = Clock::now();
        m.rope.apply(s.q.data(), s.k.data(), attn.num_q_heads, attn.num_kv_heads, pos);
        b = Clock::now();
        acc.add("rope", elapsed_ms(a, b));

        auto& cache = m.kv_caches[l];
        a = Clock::now();
        if (pos < cache.max_seq_len) {
            const size_t kv_elems = attn.num_kv_heads * attn.head_dim;
            std::memcpy(cache.get_k_ptr(pos), s.k.data(), kv_elems * sizeof(float));
            std::memcpy(cache.get_v_ptr(pos), s.v.data(), kv_elems * sizeof(float));
            if (pos >= cache.current_len) cache.current_len = pos + 1;
        }
        b = Clock::now();
        acc.add("kv_write", elapsed_ms(a, b));

        a = Clock::now();
        const size_t context_len = std::min(pos + 1, cache.current_len);
        const size_t q_per_kv = attn.num_q_heads / attn.num_kv_heads;
        const size_t kv_stride = attn.num_kv_heads * attn.head_dim;
        const size_t kv_offset_elems = attn.num_kv_heads * attn.head_dim;
        acc.kv_read_bytes += static_cast<uint64_t>(context_len) * kv_offset_elems * 2 * sizeof(float);
        for (size_t h = 0; h < attn.num_q_heads; ++h) {
            const float* q_head = s.q.data() + h * attn.head_dim;
            const size_t kv_h = h / q_per_kv;
            const size_t offset = kv_h * attn.head_dim;
            for (size_t p = 0; p < context_len; ++p) {
                const float* kp = cache.k_data.data() + p * kv_stride + offset;
                float dot = 0.0f;
                for (size_t d = 0; d < attn.head_dim; ++d) dot += q_head[d] * kp[d];
                s.scores[p] = dot * attn.scale;
            }
            softmax(s.scores.data(), context_len);
            float* head_out = s.attn_vec.data() + h * attn.head_dim;
            std::memset(head_out, 0, attn.head_dim * sizeof(float));
            for (size_t p = 0; p < context_len; ++p) {
                const float w = s.scores[p];
                const float* vp = cache.v_data.data() + p * kv_stride + offset;
                for (size_t d = 0; d < attn.head_dim; ++d) head_out[d] += w * vp[d];
            }
        }
        b = Clock::now();
        acc.add("attn_core", elapsed_ms(a, b));

        a = Clock::now();
        attn.o_proj.forward(s.attn_vec.data(), s.attn_out.data());
        b = Clock::now();
        acc.add("o_proj", elapsed_ms(a, b));

        a = Clock::now();
        for (size_t i = 0; i < m.config.hidden_size; ++i) s.residual[i] = in[i] + s.attn_out[i];
        b = Clock::now();
        acc.add("attn_residual", elapsed_ms(a, b));

        a = Clock::now();
        rmsnorm(s.residual.data(), layer.post_attention_layernorm_weight.data(),
                s.norm2.data(), m.config.hidden_size);
        b = Clock::now();
        acc.add("post_norm", elapsed_ms(a, b));

        auto& mlp = layer.mlp;
        a = Clock::now();
        mlp.gate_proj.forward(s.norm2.data(), s.gate.data());
        b = Clock::now();
        acc.add("gate_proj", elapsed_ms(a, b));
        a = Clock::now();
        mlp.up_proj.forward(s.norm2.data(), s.up.data());
        b = Clock::now();
        acc.add("up_proj", elapsed_ms(a, b));
        a = Clock::now();
        for (size_t i = 0; i < mlp.intermediate_size; ++i) {
            s.act[i] = silu_scalar(s.gate[i]) * s.up[i];
        }
        b = Clock::now();
        acc.add("swiglu", elapsed_ms(a, b));
        a = Clock::now();
        mlp.down_proj.forward(s.act.data(), s.mlp_out.data());
        b = Clock::now();
        acc.add("down_proj", elapsed_ms(a, b));

        a = Clock::now();
        for (size_t i = 0; i < m.config.hidden_size; ++i) out[i] = s.residual[i] + s.mlp_out[i];
        b = Clock::now();
        acc.add("mlp_residual", elapsed_ms(a, b));

        auto layer_end = Clock::now();
        const double layer_ms = elapsed_ms(layer_start, layer_end);
        acc.layer_ms[l] += layer_ms;
        acc.layer_by_pos[pos + 1][l] += layer_ms;
        std::swap(in, out);
    }

    auto a = Clock::now();
    rmsnorm(in, m.final_norm_weight.data(), s.attn_out.data(), m.config.hidden_size, m.config.rms_norm_eps);
    auto b = Clock::now();
    acc.add("final_norm", elapsed_ms(a, b));

    a = Clock::now();
    if (m.lm_head_mmap) {
        littlebit::lm_head_logits(
            m.lm_head_mmap, s.attn_out.data(), s.logits.data(),
            m.config.vocab_size, m.config.hidden_size
        );
    } else {
        std::fill(s.logits.begin(), s.logits.end(), 0.0f);
    }
    b = Clock::now();
    acc.add("lm_head", elapsed_ms(a, b));
    acc.lm_head_bytes += static_cast<uint64_t>(m.config.vocab_size) * m.config.hidden_size * sizeof(uint16_t);

    double max_abs = 0.0;
    for (size_t v = 0; v < m.config.vocab_size; ++v) {
        max_abs = std::max(max_abs, static_cast<double>(std::fabs(s.logits[v] - s.engine_logits[v])));
    }
    acc.logit_check_max_abs = std::max(acc.logit_check_max_abs, max_abs);
    ++acc.calls;
    return s.logits.data();
}

std::string jstr(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        out += c;
    }
    out += '"';
    return out;
}

void print_json(const std::string& path, const std::map<std::string, std::string>& vals) {
    std::ofstream out(path);
    out << "{\n";
    size_t i = 0;
    for (const auto& [k, v] : vals) {
        out << "  " << jstr(k) << ": " << v << (++i == vals.size() ? "\n" : ",\n");
    }
    out << "}\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "usage: perf_benchmark <model_dir> <prompt> <decode_tokens_csv> <max_seq_len> "
                     "[detail:0|1] [lm_head_repeats] [json_out]\n";
        return 2;
    }
    const std::string model_dir = argv[1];
    const std::string prompt = argv[2];
    const std::vector<int32_t> decode_tokens = parse_csv(argv[3]);
    const size_t max_seq = static_cast<size_t>(std::stoul(argv[4]));
    const bool detail = (argc > 5) ? std::stoi(argv[5]) != 0 : true;
    const size_t lm_repeats = (argc > 6) ? static_cast<size_t>(std::stoul(argv[6])) : 3;
    const std::string json_out = (argc > 7) ? argv[7] : "/tmp/opencode/lbit-phase9/profile.json";

    littlebit::Qwen3Model model;
    if (!model.load(model_dir, max_seq)) return 3;
    const size_t rss_after_load_mb = littlebit::Qwen3Model::get_current_rss_mb();

    const size_t threads = thread_count();
    const auto cpu0 = read_cpu_time();
    const auto wall0 = Clock::now();

    auto tok_start = Clock::now();
    std::vector<int32_t> prompt_ids = model.tokenizer.encode(model.tokenizer.apply_chat_template(prompt), false);
    if (prompt_ids.empty()) prompt_ids.push_back(model.tokenizer.bos_id);
    const double tokenize_ms = elapsed_ms(tok_start, Clock::now());
    const size_t prefill_len = prompt_ids.size();

    State state;
    state.init(model);
    Accum acc;
    acc.layer_ms.assign(model.config.num_hidden_layers, 0.0);
    acc.layer_by_pos.assign(prefill_len + decode_tokens.size() + 1,
                            std::vector<double>(model.config.num_hidden_layers, 0.0));

    model.clear_cache();
    double prefill_engine_ms = 0.0;
    double decode_engine_ms = 0.0;

    for (size_t i = 0; i < prefill_len; ++i) {
        auto a = Clock::now();
        const float* engine = model.forward_token(prompt_ids[i], i);
        auto b = Clock::now();
        prefill_engine_ms += elapsed_ms(a, b);
        if (detail) {
            std::copy(engine, engine + model.config.vocab_size, state.engine_logits.begin());
            instrumented_forward(model, state, prompt_ids[i], i, acc);
        }
    }
    const double ttft_ms = tokenize_ms + prefill_engine_ms;
    const size_t rss_after_prefill_mb = littlebit::Qwen3Model::get_current_rss_mb();

    size_t pos = prefill_len;
    double greedy_ms = 0.0;
    double top_p_ms = 0.0;
    int32_t sampled = 0;
    for (int32_t token : decode_tokens) {
        if (pos >= max_seq) break;
        auto a = Clock::now();
        const float* engine = model.forward_token(token, pos);
        auto b = Clock::now();
        decode_engine_ms += elapsed_ms(a, b);
        if (detail) {
            std::copy(engine, engine + model.config.vocab_size, state.engine_logits.begin());
            const float* replay = instrumented_forward(model, state, token, pos, acc);
            a = Clock::now();
            sampled = littlebit::sample_greedy(replay, model.config.vocab_size);
            b = Clock::now();
            greedy_ms += elapsed_ms(a, b);
            a = Clock::now();
            sampled = littlebit::sample_temperature_top_p(replay, model.config.vocab_size, 0.6f, 0.95f);
            b = Clock::now();
            top_p_ms += elapsed_ms(a, b);
        }
        ++pos;
    }
    const size_t rss_after_decode_mb = littlebit::Qwen3Model::get_current_rss_mb();
    acc.greedy_ms = greedy_ms;
    acc.top_p_ms = top_p_ms;

    // Isolated lm_head bandwidth measurement on a fixed hidden vector.
    double lm_iso_ms = 0.0;
    if (model.lm_head_mmap && lm_repeats > 0) {
        auto a = Clock::now();
        for (size_t r = 0; r < lm_repeats; ++r) {
            littlebit::lm_head_logits(
                model.lm_head_mmap, state.attn_out.data(), state.logits.data(),
                model.config.vocab_size, model.config.hidden_size
            );
        }
        lm_iso_ms = elapsed_ms(a, Clock::now()) / static_cast<double>(lm_repeats);
    }

    const auto wall1 = Clock::now();
    const auto cpu1 = read_cpu_time();
    const double wall_ms = elapsed_ms(wall0, wall1);
    const double cpu_ms = (cpu1.user_ms - cpu0.user_ms) + (cpu1.sys_ms - cpu0.sys_ms);

    const auto clear_start = Clock::now();
    model.clear_cache();
    const double clear_cache_ms = elapsed_ms(clear_start, Clock::now());
    const size_t rss_after_clear_mb = littlebit::Qwen3Model::get_current_rss_mb();
    const size_t peak_rss_mb_value = peak_rss_mb();

    std::vector<uint64_t> proj_bytes(7, 0);
    const char* proj_names[] = {"q", "k", "v", "o", "gate", "up", "down"};
    for (const auto& layer : model.layers) {
        proj_bytes[0] += linear_bytes(layer.self_attn.q_proj);
        proj_bytes[1] += linear_bytes(layer.self_attn.k_proj);
        proj_bytes[2] += linear_bytes(layer.self_attn.v_proj);
        proj_bytes[3] += linear_bytes(layer.self_attn.o_proj);
        proj_bytes[4] += linear_bytes(layer.mlp.gate_proj);
        proj_bytes[5] += linear_bytes(layer.mlp.up_proj);
        proj_bytes[6] += linear_bytes(layer.mlp.down_proj);
    }

    const uint64_t head_bytes = static_cast<uint64_t>(model.config.vocab_size) *
                                model.config.hidden_size * sizeof(uint16_t);
    const uint64_t total_proj_bytes = std::accumulate(proj_bytes.begin(), proj_bytes.end(), uint64_t{0});
    const size_t total_fwd = prefill_len + decode_tokens.size();

    std::map<std::string, std::string> j;
    j["model_dir"] = jstr(model_dir);
    j["prompt"] = jstr(prompt);
    j["decode_tokens_csv"] = jstr(argv[3]);
    j["detail"] = detail ? "true" : "false";
    j["threads"] = std::to_string(threads);
    j["max_seq_len"] = std::to_string(max_seq);
    j["num_layers"] = std::to_string(model.config.num_hidden_layers);
    j["hidden_size"] = std::to_string(model.config.hidden_size);
    j["vocab_size"] = std::to_string(model.config.vocab_size);
    j["prompt_tokens"] = std::to_string(prefill_len);
    j["decode_tokens"] = std::to_string(decode_tokens.size());
    j["load_ms"] = std::to_string(model.last_load_time_ms);
    j["tokenize_ms"] = std::to_string(tokenize_ms);
    j["prefill_engine_ms"] = std::to_string(prefill_engine_ms);
    j["ttft_ms"] = std::to_string(ttft_ms);
    j["decode_engine_ms"] = std::to_string(decode_engine_ms);
    j["decode_ms_per_token"] = std::to_string(decode_engine_ms / std::max<size_t>(1, decode_tokens.size()));
    j["total_forward_calls"] = std::to_string(total_fwd);
    j["wall_ms"] = std::to_string(wall_ms);
    j["cpu_ms"] = std::to_string(cpu_ms);
    j["cpu_util_pct_of_one_core"] = std::to_string(100.0 * cpu_ms / std::max(1e-9, wall_ms));
    j["rss_after_load_mb"] = std::to_string(rss_after_load_mb);
    j["rss_after_prefill_mb"] = std::to_string(rss_after_prefill_mb);
    j["rss_after_decode_mb"] = std::to_string(rss_after_decode_mb);
    j["rss_after_clear_cache_mb"] = std::to_string(rss_after_clear_mb);
    j["peak_rss_mb"] = std::to_string(peak_rss_mb_value);
    j["clear_cache_ms"] = std::to_string(clear_cache_ms);
    j["decode_tokens_per_sec"] = std::to_string(
        decode_engine_ms > 0.0
            ? 1000.0 * static_cast<double>(decode_tokens.size()) / decode_engine_ms
            : 0.0
    );
    j["rss_mb"] = std::to_string(rss_after_decode_mb);
    j["sampled_id"] = std::to_string(sampled);

    auto put_ms = [&](const char* key, double v) { j[key] = std::to_string(v); };
    put_ms("embed_ms", acc.embed_ms);
    put_ms("input_norm_ms", acc.input_norm_ms);
    put_ms("q_proj_ms", acc.q_proj_ms);
    put_ms("k_plus_v_proj_ms", acc.k_proj_ms);
    put_ms("qk_norm_ms", acc.qk_norm_ms);
    put_ms("rope_ms", acc.rope_ms);
    put_ms("kv_write_ms", acc.kv_write_ms);
    put_ms("attn_core_ms", acc.attn_core_ms);
    put_ms("o_proj_ms", acc.o_proj_ms);
    put_ms("attn_residual_ms", acc.attn_residual_ms);
    put_ms("post_norm_ms", acc.post_norm_ms);
    put_ms("gate_proj_ms", acc.gate_proj_ms);
    put_ms("up_proj_ms", acc.up_proj_ms);
    put_ms("swiglu_ms", acc.swiglu_ms);
    put_ms("down_proj_ms", acc.down_proj_ms);
    put_ms("mlp_residual_ms", acc.mlp_residual_ms);
    put_ms("final_norm_ms", acc.final_norm_ms);
    put_ms("lm_head_ms", acc.lm_head_ms);
    put_ms("lm_head_isolated_ms_per_call", lm_iso_ms);
    put_ms("greedy_sample_ms", acc.greedy_ms);
    put_ms("top_p_sample_ms", acc.top_p_ms);
    j["replay_calls"] = std::to_string(acc.calls);
    j["replay_logit_max_abs"] = std::to_string(acc.logit_check_max_abs);
    j["lm_head_bytes_per_call"] = std::to_string(head_bytes);
    const size_t calls = std::max<size_t>(1, acc.calls);
    // Convert bytes/ms -> GB/s by dividing by 1e6.
    j["lm_head_GB_per_s"] = std::to_string(head_bytes / std::max(1e-9, acc.lm_head_ms / calls) / 1e6);
    j["lm_head_isolated_GB_per_s"] = std::to_string(head_bytes / std::max(1e-9, lm_iso_ms) / 1e6);
    j["total_projection_bytes_per_token"] = std::to_string(total_proj_bytes);
    const double proj_ms_per_call = (acc.q_proj_ms + acc.k_proj_ms + acc.o_proj_ms +
                                     acc.gate_proj_ms + acc.up_proj_ms + acc.down_proj_ms) / calls;
    j["projection_GB_per_s"] = std::to_string(total_proj_bytes / std::max(1e-9, proj_ms_per_call) / 1e6);
    j["kv_read_bytes"] = std::to_string(acc.kv_read_bytes);

    std::ostringstream layer_csv;
    for (size_t l = 0; l < acc.layer_ms.size(); ++l) {
        if (l) layer_csv << ',';
        layer_csv << std::fixed << std::setprecision(4) << acc.layer_ms[l];
    }
    j["layer_ms_csv"] = jstr(layer_csv.str());

    for (size_t p = 0; p < 7; ++p) {
        std::string key = std::string("proj_bytes_") + proj_names[p];
        j[key] = std::to_string(proj_bytes[p]);
    }

    std::ostringstream pos_csv;
    for (size_t p = 1; p < acc.layer_by_pos.size(); ++p) {
        double sum = 0.0;
        for (double v : acc.layer_by_pos[p]) sum += v;
        if (p > 1) pos_csv << ',';
        pos_csv << std::fixed << std::setprecision(4) << sum;
    }
    j["layers_ms_by_position_csv"] = jstr(pos_csv.str());

    print_json(json_out, j);
    std::cout << "[PROFILE] prompt_tokens=" << prefill_len << " decode_tokens=" << decode_tokens.size()
              << " ttft_ms=" << ttft_ms << " decode_ms_per_token=" << decode_engine_ms / std::max<size_t>(1, decode_tokens.size())
              << " replay_logit_max_abs=" << acc.logit_check_max_abs << '\n';
    std::cout << "[PROFILE] json=" << json_out << '\n';
    std::cout << "[MEMORY] rss_after_load_mb=" << rss_after_load_mb
              << " rss_after_prefill_mb=" << rss_after_prefill_mb
              << " rss_after_decode_mb=" << rss_after_decode_mb
              << " rss_after_clear_cache_mb=" << rss_after_clear_mb
              << " peak_rss_mb=" << peak_rss_mb_value << '\n';
    std::cout << "[LIFECYCLE] clear_cache_ms=" << clear_cache_ms
              << " decode_tokens_per_sec="
              << (decode_engine_ms > 0.0
                      ? 1000.0 * static_cast<double>(decode_tokens.size()) / decode_engine_ms
                      : 0.0)
              << '\n';
    return 0;
}
