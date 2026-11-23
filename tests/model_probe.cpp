#include "littlebit/safetensors.h"
#include "littlebit/linear.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>

// Small standalone probe used by scripts/validate_real_model.py.
// Usage: model_probe <model.safetensors> <tensor_prefix> <x.bin> <y.bin>
int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: model_probe <model.safetensors> <prefix> <x.bin> <y.bin>\n";
        return 2;
    }

    littlebit::SafeTensorsFile file;
    if (!file.open(argv[1])) return 3;

    littlebit::LittleBitLinear layer;
    if (!file.load_littlebit_linear(argv[2], layer)) {
        std::cerr << "Failed to load linear layer: " << argv[2] << "\n";
        return 4;
    }

    std::ifstream x_in(argv[3], std::ios::binary);
    std::vector<float> x(layer.in_features);
    x_in.read(reinterpret_cast<char*>(x.data()), x.size() * sizeof(float));
    if (!x_in || static_cast<size_t>(x_in.gcount()) != x.size() * sizeof(float)) {
        std::cerr << "Failed to read input vector\n";
        return 5;
    }

    std::vector<float> y(layer.out_features);
    layer.forward(x.data(), y.data());

    std::ofstream y_out(argv[4], std::ios::binary);
    y_out.write(reinterpret_cast<const char*>(y.data()), y.size() * sizeof(float));
    std::cout << layer.in_features << " " << layer.out_features << " "
              << layer.split_dim << " " << (layer.has_residual ? 1 : 0) << "\n";
    return 0;
}
