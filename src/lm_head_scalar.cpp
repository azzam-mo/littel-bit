#include "littlebit/common.h"
#include "lm_head_avx2.h"

namespace littlebit {
namespace {

// Scalar-only definition of lm_head_logits for non-x86 builds. This is a copy
// of the frozen scalar oracle in src/lm_head_avx2.cpp; no NEON LM head is
// implemented in this step.
void scalar_lm_head(
    const uint16_t* W, const float* h, float* out, size_t V, size_t H
) {
    for (size_t v = 0; v < V; ++v) {
        const uint16_t* row = W + v * H;
        float dot = 0.0f;
        for (size_t i = 0; i < H; ++i) {
            dot += h[i] * bf16_to_fp32(row[i]);
        }
        out[v] = dot;
    }
}

} // namespace

void lm_head_logits(
    const uint16_t* W, const float* h, float* out, size_t V, size_t H
) {
    scalar_lm_head(W, h, out, V, H);
}

} // namespace littlebit
