#pragma once

#include "littlebit/common.h"
#include "littlebit/binary_packer.h"
#include <vector>
#include <memory>
#include <cstdint>

namespace littlebit {

class LittleBitLinear {
public:
    size_t in_features = 0;
    size_t out_features = 0;
    size_t split_dim = 0;
    bool has_residual = false;

    // Primary path weights
    std::vector<int32_t> U_packed;     // Shape [out_features, ceil(split_dim / 32)]
    std::vector<int32_t> V_packed;     // Shape [split_dim, ceil(in_features / 32)]
    std::vector<float> u1;             // Shape [out_features]
    std::vector<float> u2;             // Shape [split_dim]
    std::vector<float> v1;             // Shape [split_dim]
    std::vector<float> v2;             // Shape [in_features]

    // Residual path weights (if has_residual is true)
    std::vector<int32_t> U_R_packed;   // Shape [out_features, ceil(split_dim / 32)]
    std::vector<int32_t> V_R_packed;   // Shape [split_dim, ceil(in_features / 32)]
    std::vector<float> u1_R;           // Shape [out_features]
    std::vector<float> u2_R;           // Shape [split_dim]
    std::vector<float> v1_R;           // Shape [split_dim]
    std::vector<float> v2_R;           // Shape [in_features]

    // Optional bias
    std::vector<float> bias;

    // Scratch buffers for zero-allocation forward pass
    mutable std::vector<float> scratch_x_v2;
    mutable std::vector<float> scratch_h1;
    mutable std::vector<float> scratch_h2;
    mutable std::vector<float> scratch_h3;
    mutable std::vector<float> scratch_res;

    LittleBitLinear() = default;

    void initialize(size_t in_feat, size_t out_feat, size_t split_d, bool residual = true);

    // Forward pass for single token vector: x [in_features] -> y [out_features]
    void forward(const float* x, float* y) const;

private:
    void compute_path(
        const float* x,
        float* out,
        const std::vector<int32_t>& V_p,
        const std::vector<int32_t>& U_p,
        const std::vector<float>& v2_vec,
        const std::vector<float>& v1_vec,
        const std::vector<float>& u2_vec,
        const std::vector<float>& u1_vec
    ) const;
};

} // namespace littlebit
