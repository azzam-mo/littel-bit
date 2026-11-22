#include "littlebit/linear.h"
#include <cstring>

namespace littlebit {

void LittleBitLinear::initialize(size_t in_feat, size_t out_feat, size_t split_d, bool residual) {
    in_features = in_feat;
    out_features = out_feat;
    split_dim = split_d;
    has_residual = residual;

    size_t words_V = (in_features + 31) / 32;
    size_t words_U = (split_dim + 31) / 32;

    U_packed.assign(out_features * words_U, 0);
    V_packed.assign(split_dim * words_V, 0);
    u1.assign(out_features, 0.0f);
    u2.assign(split_dim, 0.0f);
    v1.assign(split_dim, 0.0f);
    v2.assign(in_features, 0.0f);

    if (has_residual) {
        U_R_packed.assign(out_features * words_U, 0);
        V_R_packed.assign(split_dim * words_V, 0);
        u1_R.assign(out_features, 0.0f);
        u2_R.assign(split_dim, 0.0f);
        v1_R.assign(split_dim, 0.0f);
        v2_R.assign(in_features, 0.0f);
        scratch_res.resize(out_features);
    }

    scratch_x_v2.resize(in_features);
    scratch_h1.resize(split_dim);
    scratch_h2.resize(split_dim);
    scratch_h3.resize(out_features);
}

void LittleBitLinear::compute_path(
    const float* x,
    float* out,
    const std::vector<int32_t>& V_p,
    const std::vector<int32_t>& U_p,
    const std::vector<float>& v2_vec,
    const std::vector<float>& v1_vec,
    const std::vector<float>& u2_vec,
    const std::vector<float>& u1_vec
) const {
    // 1. x_v2 = x * v2
    for (size_t k = 0; k < in_features; ++k) {
        scratch_x_v2[k] = x[k] * v2_vec[k];
    }

    // 2. h1 = x_v2 @ V^T
    // V has shape [split_dim, in_features], row s has V[s, k]
    // h1[s] = sum_k (V[s, k] * x_v2[k])
    binary_matvec_mult(V_p.data(), scratch_x_v2.data(), scratch_h1.data(), split_dim, in_features);

    // 3. h2 = h1 * (v1 * u2)
    for (size_t s = 0; s < split_dim; ++s) {
        scratch_h2[s] = scratch_h1[s] * (v1_vec[s] * u2_vec[s]);
    }

    // 4. h3 = h2 @ U^T
    // U has shape [out_features, split_dim], row o has U[o, s]
    // h3[o] = sum_s (U[o, s] * h2[s])
    binary_matvec_mult(U_p.data(), scratch_h2.data(), scratch_h3.data(), out_features, split_dim);

    // 5. out = h3 * u1
    for (size_t o = 0; o < out_features; ++o) {
        out[o] = scratch_h3[o] * u1_vec[o];
    }
}

void LittleBitLinear::forward(const float* x, float* y) const {
    // Primary path
    compute_path(x, y, V_packed, U_packed, v2, v1, u2, u1);

    // Residual path
    if (has_residual) {
        compute_path(x, scratch_res.data(), V_R_packed, U_R_packed, v2_R, v1_R, u2_R, u1_R);
        for (size_t o = 0; o < out_features; ++o) {
            y[o] += scratch_res[o];
        }
    }

    // Bias
    if (!bias.empty()) {
        for (size_t o = 0; o < out_features; ++o) {
            y[o] += bias[o];
        }
    }
}

} // namespace littlebit
