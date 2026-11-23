#include "littlebit/transformer_block.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

int main() {
    std::cout << "[TEST] Running TransformerBlock unit tests..." << std::endl;

    size_t hidden_size = 64;
    size_t intermediate_size = 128;
    size_t num_q_heads = 4;
    size_t num_kv_heads = 2;
    size_t head_dim = 16;
    size_t max_seq = 32;

    littlebit::TransformerBlock block;
    block.initialize(0, hidden_size, intermediate_size, max_seq);
    block.self_attn.initialize(hidden_size, num_q_heads, num_kv_heads, head_dim, max_seq);

    // Initialize mock linear layers
    block.self_attn.q_proj.initialize(hidden_size, hidden_size, 32, false);
    block.self_attn.k_proj.initialize(hidden_size, num_kv_heads * head_dim, 16, false);
    block.self_attn.v_proj.initialize(hidden_size, num_kv_heads * head_dim, 16, false);
    block.self_attn.o_proj.initialize(hidden_size, hidden_size, 32, false);

    block.mlp.gate_proj.initialize(hidden_size, intermediate_size, 32, false);
    block.mlp.up_proj.initialize(hidden_size, intermediate_size, 32, false);
    block.mlp.down_proj.initialize(intermediate_size, hidden_size, 32, false);

    // Fill scale vectors so output is non-zero
    for (auto& v : block.self_attn.q_proj.u1) v = 0.1f;
    for (auto& v : block.self_attn.q_proj.v2) v = 0.1f;
    for (auto& v : block.self_attn.k_proj.u1) v = 0.1f;
    for (auto& v : block.self_attn.k_proj.v2) v = 0.1f;
    for (auto& v : block.self_attn.v_proj.u1) v = 0.1f;
    for (auto& v : block.self_attn.v_proj.v2) v = 0.1f;
    for (auto& v : block.self_attn.o_proj.u1) v = 0.1f;
    for (auto& v : block.self_attn.o_proj.v2) v = 0.1f;

    for (auto& v : block.mlp.gate_proj.u1) v = 0.1f;
    for (auto& v : block.mlp.gate_proj.v2) v = 0.1f;
    for (auto& v : block.mlp.up_proj.u1) v = 0.1f;
    for (auto& v : block.mlp.up_proj.v2) v = 0.1f;
    for (auto& v : block.mlp.down_proj.u1) v = 0.1f;
    for (auto& v : block.mlp.down_proj.v2) v = 0.1f;

    littlebit::RoPEEmbeddings rope;
    rope.initialize(max_seq, head_dim, 10000.0f);

    littlebit::KVCache cache;
    cache.initialize(max_seq, num_kv_heads, head_dim);

    std::vector<float> x(hidden_size, 0.5f);
    std::vector<float> out(hidden_size, 0.0f);

    // Forward step 0
    block.forward(x.data(), out.data(), 0, rope, cache);
    assert(cache.current_len == 1);
    for (size_t i = 0; i < hidden_size; ++i) {
        assert(!std::isnan(out[i]) && !std::isinf(out[i]));
    }
    std::cout << "  ✓ Step 0 forward completed successfully, KV length = " << cache.current_len << std::endl;

    // Forward step 1
    block.forward(out.data(), out.data(), 1, rope, cache);
    assert(cache.current_len == 2);
    for (size_t i = 0; i < hidden_size; ++i) {
        assert(!std::isnan(out[i]) && !std::isinf(out[i]));
    }
    std::cout << "  ✓ Step 1 forward completed successfully, KV length = " << cache.current_len << std::endl;

    std::cout << "All TransformerBlock unit tests passed successfully!" << std::endl;
    return 0;
}
