#include "littlebit/attention.h"
#include <cstring>
#include <cmath>

namespace littlebit {

void KVCache::initialize(size_t max_len, size_t kv_heads, size_t h_dim) {
    max_seq_len = max_len;
    num_kv_heads = kv_heads;
    head_dim = h_dim;
    current_len = 0;
    k_data.assign(max_seq_len * num_kv_heads * head_dim, 0.0f);
    v_data.assign(max_seq_len * num_kv_heads * head_dim, 0.0f);
}

void KVCache::clear() {
    current_len = 0;
}

void Attention::initialize(
    size_t h_size,
    size_t q_heads,
    size_t kv_heads,
    size_t h_dim,
    size_t max_seq
) {
    hidden_size = h_size;
    num_q_heads = q_heads;
    num_kv_heads = kv_heads;
    head_dim = h_dim;
    scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    q_buf.resize(num_q_heads * head_dim);
    k_buf.resize(num_kv_heads * head_dim);
    v_buf.resize(num_kv_heads * head_dim);
    attn_scores.resize(max_seq);
    attn_out.resize(hidden_size);

    q_norm_weight.assign(head_dim, 1.0f);
    k_norm_weight.assign(head_dim, 1.0f);
}

void Attention::forward(
    const float* x,
    float* out,
    size_t pos,
    const RoPEEmbeddings& rope,
    KVCache& kv_cache
) const {
    // 1. Projections
    q_proj.forward(x, q_buf.data());
    k_proj.forward(x, k_buf.data());
    v_proj.forward(x, v_buf.data());

    // 2. QK Normalization (RMSNorm on each head vector of length head_dim)
    if (!q_norm_weight.empty()) {
        for (size_t h = 0; h < num_q_heads; ++h) {
            float* q_head = q_buf.data() + h * head_dim;
            rmsnorm(q_head, q_norm_weight.data(), q_head, head_dim);
        }
    }
    if (!k_norm_weight.empty()) {
        for (size_t h = 0; h < num_kv_heads; ++h) {
            float* k_head = k_buf.data() + h * head_dim;
            rmsnorm(k_head, k_norm_weight.data(), k_head, head_dim);
        }
    }

    // 3. Rotary Position Embeddings
    rope.apply(q_buf.data(), k_buf.data(), num_q_heads, num_kv_heads, pos);

    // 4. Append to KV Cache
    if (pos < kv_cache.max_seq_len) {
        float* k_dest = kv_cache.get_k_ptr(pos);
        float* v_dest = kv_cache.get_v_ptr(pos);
        std::memcpy(k_dest, k_buf.data(), num_kv_heads * head_dim * sizeof(float));
        std::memcpy(v_dest, v_buf.data(), num_kv_heads * head_dim * sizeof(float));
        if (pos >= kv_cache.current_len) {
            kv_cache.current_len = pos + 1;
        }
    }

    size_t context_len = std::min(pos + 1, kv_cache.current_len);
    size_t q_per_kv = num_q_heads / num_kv_heads; // Group size = 32 / 8 = 4

    // 5. Scaled Dot-Product Attention with GQA
    for (size_t h = 0; h < num_q_heads; ++h) {
        const float* q_head = q_buf.data() + h * head_dim;
        size_t kv_h = h / q_per_kv;
        size_t kv_stride = num_kv_heads * head_dim;
        size_t kv_offset = kv_h * head_dim;

        // Compute attention scores for each token up to pos
        for (size_t p = 0; p < context_len; ++p) {
            const float* k_past = kv_cache.k_data.data() + p * kv_stride + kv_offset;
            float dot = 0.0f;
            for (size_t d = 0; d < head_dim; ++d) {
                dot += q_head[d] * k_past[d];
            }
            attn_scores[p] = dot * scale;
        }

        // Softmax
        softmax(attn_scores.data(), context_len);

        // Weighted sum over V values
        float* head_out = attn_out.data() + h * head_dim;
        std::memset(head_out, 0, head_dim * sizeof(float));

        for (size_t p = 0; p < context_len; ++p) {
            float weight = attn_scores[p];
            const float* v_past = kv_cache.v_data.data() + p * kv_stride + kv_offset;
            for (size_t d = 0; d < head_dim; ++d) {
                head_out[d] += weight * v_past[d];
            }
        }
    }

    // 6. Final output projection: o_proj
    o_proj.forward(attn_out.data(), out);
}

} // namespace littlebit
