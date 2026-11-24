#pragma once

#include <cstddef>
#include <cstdint>

namespace littlebit {

// Dense BF16 LM-head projection. Uses AVX2 row lanes when available, with the
// original scalar path preserved as a runtime fallback.
void lm_head_logits(
    const uint16_t* weights,
    const float* hidden,
    float* logits,
    size_t vocab_size,
    size_t hidden_size
);

} // namespace littlebit
