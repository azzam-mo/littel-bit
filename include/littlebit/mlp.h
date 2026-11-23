#pragma once

#include "littlebit/common.h"
#include "littlebit/linear.h"
#include "littlebit/ops.h"
#include <vector>

namespace littlebit {

class MLP {
public:
    size_t hidden_size = 4096;
    size_t intermediate_size = 12288;

    LittleBitLinear gate_proj;
    LittleBitLinear up_proj;
    LittleBitLinear down_proj;

    // Preallocated buffers
    mutable std::vector<float> gate_buf; // [intermediate_size]
    mutable std::vector<float> up_buf;   // [intermediate_size]
    mutable std::vector<float> act_buf;  // [intermediate_size]

    MLP() = default;

    void initialize(size_t h_size = 4096, size_t inter_size = 12288);

    // Forward pass: x [hidden_size] -> out [hidden_size]
    void forward(const float* x, float* out) const;
};

} // namespace littlebit
