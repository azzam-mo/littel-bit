#include "littlebit/ops.h"
#include <algorithm>
#include <numeric>
#include <random>

namespace littlebit {

void rmsnorm(
    const float* x,
    const float* weight,
    float* out,
    size_t dim,
    float eps
) {
    float sum_sq = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        sum_sq += x[i] * x[i];
    }
    float rms = std::sqrt(sum_sq / static_cast<float>(dim) + eps);
    float scale = 1.0f / rms;

    for (size_t i = 0; i < dim; ++i) {
        out[i] = (x[i] * scale) * weight[i];
    }
}

void silu(const float* x, float* out, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        out[i] = silu_scalar(x[i]);
    }
}

void elementwise_mul(const float* a, const float* b, float* out, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        out[i] = a[i] * b[i];
    }
}

void elementwise_add(const float* a, const float* b, float* out, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        out[i] = a[i] + b[i];
    }
}

void softmax(float* x, size_t size) {
    if (size == 0) return;
    float max_val = x[0];
    for (size_t i = 1; i < size; ++i) {
        if (x[i] > max_val) max_val = x[i];
    }

    float sum_exp = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        x[i] = std::exp(x[i] - max_val);
        sum_exp += x[i];
    }

    float inv_sum = 1.0f / sum_exp;
    for (size_t i = 0; i < size; ++i) {
        x[i] *= inv_sum;
    }
}

void RoPEEmbeddings::initialize(size_t max_seq_len, size_t h_dim, float th) {
    head_dim = h_dim;
    theta = th;
    size_t half_dim = head_dim / 2;

    cos_table.resize(max_seq_len * half_dim);
    sin_table.resize(max_seq_len * half_dim);

    // Compute inv_freq: 1.0 / (theta ^ (2i / head_dim))
    std::vector<float> inv_freq(half_dim);
    for (size_t i = 0; i < half_dim; ++i) {
        inv_freq[i] = 1.0f / std::pow(theta, static_cast<float>(2 * i) / static_cast<float>(head_dim));
    }

    for (size_t pos = 0; pos < max_seq_len; ++pos) {
        for (size_t i = 0; i < half_dim; ++i) {
            float freq = static_cast<float>(pos) * inv_freq[i];
            cos_table[pos * half_dim + i] = std::cos(freq);
            sin_table[pos * half_dim + i] = std::sin(freq);
        }
    }
}

void RoPEEmbeddings::apply(
    float* q,
    float* k,
    size_t num_q_heads,
    size_t num_kv_heads,
    size_t position
) const {
    size_t half_dim = head_dim / 2;
    const float* cos_row = cos_table.data() + position * half_dim;
    const float* sin_row = sin_table.data() + position * half_dim;

    // Rotate Q heads: [q_0 ... q_{half-1}], [q_{half} ... q_{head_dim-1}]
    for (size_t h = 0; h < num_q_heads; ++h) {
        float* q_head = q + h * head_dim;
        for (size_t i = 0; i < half_dim; ++i) {
            float q0 = q_head[i];
            float q1 = q_head[i + half_dim];
            float c = cos_row[i];
            float s = sin_row[i];
            // HF rotate_half: (-q1, q0)
            q_head[i]            = q0 * c - q1 * s;
            q_head[i + half_dim] = q1 * c + q0 * s;
        }
    }

    // Rotate K heads
    for (size_t h = 0; h < num_kv_heads; ++h) {
        float* k_head = k + h * head_dim;
        for (size_t i = 0; i < half_dim; ++i) {
            float k0 = k_head[i];
            float k1 = k_head[i + half_dim];
            float c = cos_row[i];
            float s = sin_row[i];
            k_head[i]            = k0 * c - k1 * s;
            k_head[i + half_dim] = k1 * c + k0 * s;
        }
    }
}

int32_t sample_greedy(const float* logits, size_t vocab_size) {
    if (vocab_size == 0) return 0;
    int32_t best_id = 0;
    float best_val = logits[0];
    for (size_t i = 1; i < vocab_size; ++i) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best_id = static_cast<int32_t>(i);
        }
    }
    return best_id;
}

int32_t sample_temperature_top_p(
    const float* logits,
    size_t vocab_size,
    float temperature,
    float top_p
) {
    if (temperature <= 0.0f) {
        return sample_greedy(logits, vocab_size);
    }

    std::vector<std::pair<float, int32_t>> probs(vocab_size);
    float max_logit = logits[0];
    for (size_t i = 1; i < vocab_size; ++i) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    float sum_exp = 0.0f;
    for (size_t i = 0; i < vocab_size; ++i) {
        float p = std::exp((logits[i] - max_logit) / temperature);
        probs[i] = {p, static_cast<int32_t>(i)};
        sum_exp += p;
    }

    for (size_t i = 0; i < vocab_size; ++i) {
        probs[i].first /= sum_exp;
    }

    // Sort descending by probability
    std::sort(probs.begin(), probs.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });

    // Top-p truncation
    float cum_p = 0.0f;
    size_t cutoff = vocab_size;
    for (size_t i = 0; i < vocab_size; ++i) {
        cum_p += probs[i].first;
        if (cum_p >= top_p) {
            cutoff = i + 1;
            break;
        }
    }

    // Sample from top-p candidates
    static thread_local std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(0.0f, cum_p);
    float r = dist(rng);

    float acc = 0.0f;
    for (size_t i = 0; i < cutoff; ++i) {
        acc += probs[i].first;
        if (r <= acc) {
            return probs[i].second;
        }
    }
    return probs[0].second;
}

} // namespace littlebit
