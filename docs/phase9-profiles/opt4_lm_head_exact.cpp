#include "littlebit/common.h"
#include "littlebit/model.h"
#include "lm_head_avx2.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

namespace {

uint32_t bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

float from_bits(uint32_t u) {
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

__attribute__((noinline))
void scalar_oracle(
    const uint16_t* W, const float* h, float* out, size_t V, size_t H
) {
    for (size_t v = 0; v < V; ++v) {
        const uint16_t* row = W + v * H;
        float dot = 0.0f;
        for (size_t i = 0; i < H; ++i) {
            dot += h[i] * littlebit::bf16_to_fp32(row[i]);
        }
        out[v] = dot;
    }
}

bool check_case(const std::vector<uint16_t>& W, const std::vector<float>& h,
                size_t V, size_t H, const char* label) {
    std::vector<float> avx(V), scalar(V), oracle(V);
    unsetenv("LITTLEBIT_DISABLE_AVX2");
    littlebit::lm_head_logits(W.data(), h.data(), avx.data(), V, H);
    setenv("LITTLEBIT_DISABLE_AVX2", "1", 1);
    littlebit::lm_head_logits(W.data(), h.data(), scalar.data(), V, H);
    unsetenv("LITTLEBIT_DISABLE_AVX2");
    scalar_oracle(W.data(), h.data(), oracle.data(), V, H);

    for (size_t v = 0; v < V; ++v) {
        if (bits(avx[v]) != bits(scalar[v]) || bits(scalar[v]) != bits(oracle[v])) {
            std::cerr << "MISMATCH " << label << " V=" << V << " H=" << H
                      << " v=" << v << " avx=" << std::hex << bits(avx[v])
                      << " scalar=" << bits(scalar[v])
                      << " oracle=" << bits(oracle[v]) << std::dec << '\n';
            for (size_t i = 0; i < H; ++i) {
                std::cerr << "  i=" << i << " h=" << std::hex << bits(h[i])
                          << " w=" << (uint32_t(W[v*H+i]) << 16) << std::dec << '\n';
            }
            return false;
        }
    }
    return true;
}

} // namespace

int main() {
    const bool finite_only = std::getenv("OPT4_FINITE_ONLY") != nullptr;
    const uint16_t w_patterns[] = {
        0x0000, 0x8000, 0x007f, 0x807f, 0x0080, 0x8080,
        0x3f80, 0xbf80, 0x7f80, 0xff80, 0x7fa1, 0xffa1,
        0x7fc0, 0xffc0
    };
    const uint32_t h_patterns[] = {
        0x00000000, 0x80000000, 0x00000001, 0x80000001,
        0x007fffff, 0x807fffff, 0x00800000, 0x80800000,
        0x3f800000, 0xbf800000, 0x7f7fffff, 0xff7fffff,
        0x7f800000, 0xff800000,
        0x7fa12345, 0xffa12345, 0x7fc00000, 0xffc00000
    };

    size_t cases = 0;
    for (size_t V : finite_only ? std::vector<size_t>{} : std::vector<size_t>{1, 7, 8, 9, 16, 17}) {
        for (size_t H : finite_only ? std::vector<size_t>{} : std::vector<size_t>{1, 7, 8, 9, 15, 16, 33, 64}) {
            std::vector<uint16_t> W(V * H);
            std::vector<float> h(H);
            for (size_t v = 0; v < V; ++v) {
                for (size_t i = 0; i < H; ++i) {
                    W[v * H + i] = w_patterns[(v * 3 + i * 5) % 14];
                }
            }
            for (size_t i = 0; i < H; ++i) {
                h[i] = from_bits(h_patterns[(i * 7) % 18]);
            }
            if (!check_case(W, h, V, H, "special")) return 1;
            ++cases;
        }
    }

    std::mt19937 rng(9605266);
    std::uniform_int_distribution<uint16_t> wdist(0, 0xffff);
    std::uniform_int_distribution<uint32_t> hdist(0, 0xffffffffu);
    for (int rep = 0; rep < 4; ++rep) {
        const size_t V = 257;
        const size_t H = 4096;
        std::vector<uint16_t> W(V * H);
        std::vector<float> h(H);
        for (auto& x : W) {
            x = wdist(rng);
            if (finite_only && ((x >> 7) & 0xff) == 0xff) x &= 0xbfff;
        }
        for (size_t i = 0; i < H; ++i) {
            const uint32_t u = hdist(rng);
            const uint32_t exponent = (u >> 23) & 0xff;
            h[i] = from_bits(
                finite_only && exponent == 0xff
                    ? (u & 0x807fffffu) | 0x7f000000u
                    : u
            );
        }
        if (!check_case(W, h, V, H, "random")) return 1;
        ++cases;
    }

    std::cout << "lm_head exact cases=" << cases << " PASS\n";
    return 0;
}
