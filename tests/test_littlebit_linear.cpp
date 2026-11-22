#include "littlebit/linear.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

int main() {
    std::cout << "[TEST] Running LittleBitLinear unit tests..." << std::endl;

    size_t in_features = 8;
    size_t out_features = 4;
    size_t split_dim = 6;

    littlebit::LittleBitLinear layer;
    layer.initialize(in_features, out_features, split_dim, true);

    // Setup synthetic weights for Primary path
    std::vector<int8_t> raw_U(out_features * split_dim);
    for (size_t i = 0; i < raw_U.size(); ++i) raw_U[i] = (i % 2 == 0) ? 1 : -1;
    layer.U_packed = littlebit::pack_binary_matrix(raw_U.data(), out_features, split_dim);

    std::vector<int8_t> raw_V(split_dim * in_features);
    for (size_t i = 0; i < raw_V.size(); ++i) raw_V[i] = (i % 3 == 0) ? 1 : -1;
    layer.V_packed = littlebit::pack_binary_matrix(raw_V.data(), split_dim, in_features);

    for (size_t i = 0; i < out_features; ++i) layer.u1[i] = 0.5f + static_cast<float>(i) * 0.1f;
    for (size_t i = 0; i < split_dim; ++i) layer.u2[i] = 1.0f + static_cast<float>(i) * 0.05f;
    for (size_t i = 0; i < split_dim; ++i) layer.v1[i] = 0.8f - static_cast<float>(i) * 0.02f;
    for (size_t i = 0; i < in_features; ++i) layer.v2[i] = 1.2f + static_cast<float>(i) * 0.01f;

    // Setup synthetic weights for Residual path
    std::vector<int8_t> raw_U_R(out_features * split_dim);
    for (size_t i = 0; i < raw_U_R.size(); ++i) raw_U_R[i] = (i % 3 == 1) ? 1 : -1;
    layer.U_R_packed = littlebit::pack_binary_matrix(raw_U_R.data(), out_features, split_dim);

    std::vector<int8_t> raw_V_R(split_dim * in_features);
    for (size_t i = 0; i < raw_V_R.size(); ++i) raw_V_R[i] = (i % 2 == 1) ? 1 : -1;
    layer.V_R_packed = littlebit::pack_binary_matrix(raw_V_R.data(), split_dim, in_features);

    for (size_t i = 0; i < out_features; ++i) layer.u1_R[i] = 0.1f * static_cast<float>(i + 1);
    for (size_t i = 0; i < split_dim; ++i) layer.u2_R[i] = 0.2f * static_cast<float>(i + 1);
    for (size_t i = 0; i < split_dim; ++i) layer.v1_R[i] = 0.3f * static_cast<float>(i + 1);
    for (size_t i = 0; i < in_features; ++i) layer.v2_R[i] = 0.4f * static_cast<float>(i + 1);

    // Input vector
    std::vector<float> x = {1.0f, -0.5f, 2.0f, 0.3f, -1.2f, 0.8f, -0.4f, 1.5f};
    std::vector<float> y(out_features, 0.0f);

    layer.forward(x.data(), y.data());

    // Compute expected result manually using the exact mathematical definition:
    // primary: ((((x * v2) @ V^T) * (v1 * u2)) @ U^T) * u1
    auto compute_ref = [&](
        const std::vector<int8_t>& U_mat,
        const std::vector<int8_t>& V_mat,
        const std::vector<float>& u1_v,
        const std::vector<float>& u2_v,
        const std::vector<float>& v1_v,
        const std::vector<float>& v2_v
    ) {
        std::vector<float> x_v2(in_features);
        for (size_t k = 0; k < in_features; ++k) x_v2[k] = x[k] * v2_v[k];

        std::vector<float> h1(split_dim, 0.0f);
        for (size_t s = 0; s < split_dim; ++s) {
            for (size_t k = 0; k < in_features; ++k) {
                h1[s] += x_v2[k] * static_cast<float>(V_mat[s * in_features + k]);
            }
        }

        std::vector<float> h2(split_dim);
        for (size_t s = 0; s < split_dim; ++s) {
            h2[s] = h1[s] * (v1_v[s] * u2_v[s]);
        }

        std::vector<float> h3(out_features, 0.0f);
        for (size_t o = 0; o < out_features; ++o) {
            for (size_t s = 0; s < split_dim; ++s) {
                h3[o] += h2[s] * static_cast<float>(U_mat[o * split_dim + s]);
            }
        }

        std::vector<float> out(out_features);
        for (size_t o = 0; o < out_features; ++o) {
            out[o] = h3[o] * u1_v[o];
        }
        return out;
    };

    auto expected_primary = compute_ref(raw_U, raw_V, layer.u1, layer.u2, layer.v1, layer.v2);
    auto expected_residual = compute_ref(raw_U_R, raw_V_R, layer.u1_R, layer.u2_R, layer.v1_R, layer.v2_R);

    for (size_t o = 0; o < out_features; ++o) {
        float expected_y = expected_primary[o] + expected_residual[o];
        float diff = std::fabs(y[o] - expected_y);
        std::cout << "  Output[" << o << "] = " << y[o] << ", Expected = " << expected_y << ", Diff = " << diff << std::endl;
        assert(diff < 1e-4f);
    }

    std::cout << "✓ LittleBitLinear forward pass matches mathematical reference perfectly!" << std::endl;
    return 0;
}
