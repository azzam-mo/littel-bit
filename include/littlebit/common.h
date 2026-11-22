#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace littlebit {

// Half precision / BF16 conversions
inline float bf16_to_fp32(uint16_t b) {
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(float));
    return f;
}

inline uint16_t fp32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(float));
    return static_cast<uint16_t>(u >> 16);
}

inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h & 0x7C00) >> 10;
    uint32_t mant = (h & 0x03FF);

    if (exp == 0) {
        if (mant == 0) {
            uint32_t res = sign;
            float f;
            std::memcpy(&f, &res, sizeof(float));
            return f;
        }
        while (!(mant & 0x0400)) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= ~0x0400;
    } else if (exp == 31) {
        uint32_t res = sign | 0x7F800000 | (mant << 13);
        float f;
        std::memcpy(&f, &res, sizeof(float));
        return f;
    }

    exp = exp + (127 - 15);
    mant = mant << 13;
    uint32_t res = sign | (exp << 23) | mant;
    float f;
    std::memcpy(&f, &res, sizeof(float));
    return f;
}

struct TensorShape {
    std::vector<int64_t> dims;

    size_t numel() const {
        if (dims.empty()) return 0;
        size_t total = 1;
        for (auto d : dims) total *= static_cast<size_t>(d);
        return total;
    }
};

} // namespace littlebit
