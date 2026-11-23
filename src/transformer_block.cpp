#include "littlebit/transformer_block.h"

namespace littlebit {

void TransformerBlock::initialize(
    size_t idx,
    size_t h_size,
    size_t inter_size,
    size_t max_seq,
    size_t q_heads,
    size_t kv_heads,
    size_t h_dim
) {
    layer_idx = idx;
    hidden_size = h_size;

    input_layernorm_weight.assign(hidden_size, 1.0f);
    post_attention_layernorm_weight.assign(hidden_size, 1.0f);

    // Use the model configuration instead of hardcoded Qwen3-8B dimensions.
    // This keeps the real 32/8/128 layout by default but allows small test models.
    self_attn.initialize(hidden_size, q_heads, kv_heads, h_dim, max_seq);
    mlp.initialize(hidden_size, inter_size);

    norm_buf.resize(hidden_size);
    attn_out_buf.resize(hidden_size);
    residual_buf.resize(hidden_size);
    mlp_out_buf.resize(hidden_size);
}

bool TransformerBlock::load_from_safetensors(const SafeTensorsFile& file, const std::string& prefix) {
    // 1. Layer norms
    file.load_tensor_to_float(prefix + ".input_layernorm.weight", input_layernorm_weight);
    file.load_tensor_to_float(prefix + ".post_attention_layernorm.weight", post_attention_layernorm_weight);

    // 2. QK norms
    file.load_tensor_to_float(prefix + ".self_attn.q_norm.weight", self_attn.q_norm_weight);
    file.load_tensor_to_float(prefix + ".self_attn.k_norm.weight", self_attn.k_norm_weight);

    // 3. Attention Projections
    bool ok = true;
    ok &= file.load_littlebit_linear(prefix + ".self_attn.q_proj", self_attn.q_proj);
    ok &= file.load_littlebit_linear(prefix + ".self_attn.k_proj", self_attn.k_proj);
    ok &= file.load_littlebit_linear(prefix + ".self_attn.v_proj", self_attn.v_proj);
    ok &= file.load_littlebit_linear(prefix + ".self_attn.o_proj", self_attn.o_proj);

    // 4. MLP Projections
    ok &= file.load_littlebit_linear(prefix + ".mlp.gate_proj", mlp.gate_proj);
    ok &= file.load_littlebit_linear(prefix + ".mlp.up_proj", mlp.up_proj);
    ok &= file.load_littlebit_linear(prefix + ".mlp.down_proj", mlp.down_proj);

    return ok;
}

void TransformerBlock::forward(
    const float* x,
    float* out,
    size_t pos,
    const RoPEEmbeddings& rope,
    KVCache& kv_cache
) const {
    // 1. Pre-attention RMSNorm: norm1 = input_layernorm(x)
    rmsnorm(x, input_layernorm_weight.data(), norm_buf.data(), hidden_size);

    // 2. Self Attention
    self_attn.forward(norm_buf.data(), attn_out_buf.data(), pos, rope, kv_cache);

    // 3. First Residual: residual_buf = x + attn_out
    for (size_t i = 0; i < hidden_size; ++i) {
        residual_buf[i] = x[i] + attn_out_buf[i];
    }

    // 4. Pre-MLP RMSNorm: norm2 = post_attention_layernorm(residual_buf)
    rmsnorm(residual_buf.data(), post_attention_layernorm_weight.data(), norm_buf.data(), hidden_size);

    // 5. MLP forward
    mlp.forward(norm_buf.data(), mlp_out_buf.data());

    // 6. Second Residual: out = residual_buf + mlp_out
    for (size_t i = 0; i < hidden_size; ++i) {
        out[i] = residual_buf[i] + mlp_out_buf[i];
    }
}

} // namespace littlebit
