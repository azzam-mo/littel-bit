#include "littlebit/mlp.h"

namespace littlebit {

void MLP::initialize(size_t h_size, size_t inter_size) {
    hidden_size = h_size;
    intermediate_size = inter_size;

    gate_buf.resize(intermediate_size);
    up_buf.resize(intermediate_size);
    act_buf.resize(intermediate_size);
}

void MLP::forward(const float* x, float* out) const {
    // 1. Projections: gate and up
    gate_proj.forward(x, gate_buf.data());
    up_proj.forward(x, up_buf.data());

    // 2. SwiGLU: act = silu(gate) * up
    for (size_t i = 0; i < intermediate_size; ++i) {
        act_buf[i] = silu_scalar(gate_buf[i]) * up_buf[i];
    }

    // 3. Down projection: out = down_proj(act)
    down_proj.forward(act_buf.data(), out);
}

} // namespace littlebit
