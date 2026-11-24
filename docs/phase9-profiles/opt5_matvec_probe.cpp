#include "littlebit/binary_packer.h"
#include "littlebit/safetensors.h"

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

void reference_binary_matvec_mult(
    const int32_t* packed_A, const float* x, float* y, size_t M, size_t K
) {
    size_t words_per_row = (K + 31) / 32;
    for (size_t i = 0; i < M; ++i) {
        float sum = 0.0f;
        const int32_t* row_words = packed_A + i * words_per_row;
        size_t k = 0;
        for (size_t w = 0; w < words_per_row; ++w) {
            uint32_t word = static_cast<uint32_t>(row_words[w]);
            size_t bits_to_process = (K - k >= 32) ? 32 : (K - k);
            for (size_t b = 0; b < bits_to_process; ++b) {
                float val = x[k++];
                if ((word & (1u << b)) == 0u) {
                    sum += val;
                } else {
                    sum -= val;
                }
            }
        }
        y[i] = sum;
    }
}

__attribute__((target("avx2")))
void scalar_tail(
    const int32_t* packed_A, const float* x, float* y, size_t M, size_t K
) {
    size_t words_per_row = (K + 31) / 32;
    for (size_t i = 0; i < M; ++i) {
        float sum = 0.0f;
        const int32_t* row_words = packed_A + i * words_per_row;
        size_t k = 0;
        for (size_t w = 0; w < words_per_row; ++w) {
            uint32_t word = static_cast<uint32_t>(row_words[w]);
            size_t bits_to_process = (K - k >= 32) ? 32 : (K - k);
            for (size_t b = 0; b < bits_to_process; ++b) {
                float val = x[k++];
                if ((word & (1u << b)) == 0u) {
                    sum += val;
                } else {
                    sum -= val;
                }
            }
        }
        y[i] = sum;
    }
}

template <int G>
__attribute__((target("avx2")))
void avx2_rows_group(
    const int32_t* packed_A, const float* x, float* y, size_t M, size_t K
) {
    constexpr size_t W = static_cast<size_t>(G) * 8;
    size_t words_per_row = (K + 31) / 32;
    size_t i = 0;
    const __m256i all_ones = _mm256_set1_epi32(-1);
    const __m256i zero = _mm256_setzero_si256();
    const int w_int = static_cast<int>(words_per_row);
    const __m256i offsets = _mm256_set_epi32(
        7 * w_int, 6 * w_int, 5 * w_int, 4 * w_int,
        3 * w_int, 2 * w_int, 1 * w_int, 0
    );

    for (; i + W <= M; i += W) {
        __m256 sums[G];
        for (int g = 0; g < G; ++g) sums[g] = _mm256_setzero_ps();
        const int32_t* bases[G];
        for (int g = 0; g < G; ++g) {
            bases[g] = packed_A + (i + g * 8) * words_per_row;
        }

        for (size_t w = 0; w < words_per_row; ++w) {
            __m256i words[G];
            for (int g = 0; g < G; ++g) {
                words[g] = _mm256_i32gather_epi32(
                    reinterpret_cast<const int*>(bases[g] + w), offsets, 4
                );
            }

            size_t k0 = w * 32;
            size_t bits_to_process = (K - k0 >= 32) ? 32 : (K - k0);
            for (size_t b = 0; b < bits_to_process; ++b) {
                const __m256 vx = _mm256_set1_ps(x[k0 + b]);
                const __m256i bitmask = _mm256_set1_epi32(static_cast<int>(1u << b));
                for (int g = 0; g < G; ++g) {
                    __m256 add, sub;
                    __asm__ volatile("vaddps %2, %1, %0" : "=x"(add) : "x"(sums[g]), "x"(vx));
                    __asm__ volatile("vsubps %2, %1, %0" : "=x"(sub) : "x"(sums[g]), "x"(vx));
                    const __m256i bit = _mm256_and_si256(words[g], bitmask);
                    const __m256i is_one = _mm256_xor_si256(
                        _mm256_cmpeq_epi32(bit, zero), all_ones
                    );
                    sums[g] = _mm256_blendv_ps(add, sub, _mm256_castsi256_ps(is_one));
                }
            }
        }
        for (int g = 0; g < G; ++g) {
            _mm256_storeu_ps(y + i + g * 8, sums[g]);
        }
    }

    if (i < M) {
        if constexpr (G > 1) {
            avx2_rows_group<1>(packed_A + i * words_per_row, x, y + i, M - i, K);
        } else {
            scalar_tail(packed_A + i * words_per_row, x, y + i, M - i, K);
        }
    }
}

bool same_float_bits(float a, float b) {
    uint32_t ua, ub;
    std::memcpy(&ua, &a, sizeof(ua));
    std::memcpy(&ub, &b, sizeof(ub));
    return ua == ub;
}

std::vector<float> make_x(size_t K) {
    std::vector<float> x(K);
    for (size_t i = 0; i < K; ++i) {
        if (i % 257 == 0) x[i] = 0.0f;
        else if (i % 193 == 0) x[i] = -1.0f;
        else x[i] = 0.001f * static_cast<float>(i % 4096);
    }
    return x;
}

uint32_t float_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

void special_sweep() {
    const uint32_t patterns[] = {
        0x00000000, 0x80000000, 0x00000001, 0x80000001,
        0x007fffff, 0x807fffff, 0x00800000, 0x80800000,
        0x3f800000, 0xbf800000, 0x7f7fffff, 0xff7fffff,
        0x7f800000, 0xff800000, 0x7fa12345, 0xffa12345,
        0x7fc00000, 0xffc00000
    };
    for (size_t M : {size_t(1), size_t(7), size_t(8), size_t(16), size_t(31),
                     size_t(32), size_t(33), size_t(64), size_t(65)}) {
        for (size_t K : {size_t(1), size_t(7), size_t(8), size_t(31),
                         size_t(32), size_t(33), size_t(64), size_t(1024)}) {
            const size_t wpr = (K + 31) / 32;
            std::vector<int32_t> packed(M * wpr);
            std::vector<float> x(K);
            for (size_t i = 0; i < packed.size(); ++i) {
                packed[i] = static_cast<int32_t>(0x9e3779b9u * (i + 1) ^ 0xa5a5a5a5u);
            }
            for (size_t k = 0; k < K; ++k) {
                x[k] = [] (uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }(
                    patterns[(k * 7 + 3) % 18]);
            }
            std::vector<float> ref(M), got(M);
            reference_binary_matvec_mult(packed.data(), x.data(), ref.data(), M, K);
            avx2_rows_group<2>(packed.data(), x.data(), got.data(), M, K);
            for (size_t i = 0; i < M; ++i) {
                if (!same_float_bits(ref[i], got[i])) {
                    std::cerr << "SPECIAL MISMATCH G2 M=" << M << " K=" << K
                              << " i=" << i << " ref=" << std::hex << float_bits(ref[i])
                              << " got=" << float_bits(got[i]) << std::dec << '\n';
                    std::abort();
                }
            }
            avx2_rows_group<4>(packed.data(), x.data(), got.data(), M, K);
            for (size_t i = 0; i < M; ++i) {
                if (!same_float_bits(ref[i], got[i])) {
                    std::cerr << "SPECIAL MISMATCH G4 M=" << M << " K=" << K
                              << " i=" << i << " ref=" << std::hex << float_bits(ref[i])
                              << " got=" << float_bits(got[i]) << std::dec << '\n';
                    std::abort();
                }
            }
            avx2_rows_group<8>(packed.data(), x.data(), got.data(), M, K);
            for (size_t i = 0; i < M; ++i) {
                if (!same_float_bits(ref[i], got[i])) {
                    std::cerr << "SPECIAL MISMATCH G8 M=" << M << " K=" << K
                              << " i=" << i << " ref=" << std::hex << float_bits(ref[i])
                              << " got=" << float_bits(got[i]) << std::dec << '\n';
                    std::abort();
                }
            }
        }
    }
    std::cout << "special_sweep=PASS\n";
}

struct PathResult {
    std::string name;
    size_t M = 0;
    size_t K = 0;
    uint64_t bytes = 0;
    bool exact8 = false;
    bool exact16 = false;
    bool exact32 = false;
    bool exact64 = false;
    double engine_ms = 0;
    double avx8_ms = 0;
    double avx16_ms = 0;
    double avx32_ms = 0;
    double avx64_ms = 0;
};

template <typename Fn>
double bench(Fn fn, const int32_t* p, const float* x, float* y, size_t M,
             size_t K, size_t repeats) {
    for (int i = 0; i < 2; ++i) fn(p, x, y, M, K);
    volatile float sink = 0.0f;
    auto a = Clock::now();
    for (size_t r = 0; r < repeats; ++r) {
        fn(p, x, y, M, K);
        sink += y[0];
    }
    auto b = Clock::now();
    (void)sink;
    return std::chrono::duration<double, std::milli>(b - a).count() / repeats;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

PathResult measure(const std::string& name, const std::vector<int32_t>& packed,
                   size_t M, size_t K, size_t repeats) {
    PathResult r;
    r.name = name;
    r.M = M;
    r.K = K;
    r.bytes = static_cast<uint64_t>(packed.size()) * sizeof(int32_t);
    std::vector<float> x = make_x(K);
    std::vector<float> ref(M), y8(M), y16(M), y32(M), y64(M), yeng(M);
    reference_binary_matvec_mult(packed.data(), x.data(), ref.data(), M, K);
    littlebit::binary_matvec_mult_avx2(packed.data(), x.data(), yeng.data(), M, K);
    avx2_rows_group<2>(packed.data(), x.data(), y8.data(), M, K);
    avx2_rows_group<4>(packed.data(), x.data(), y16.data(), M, K);
    avx2_rows_group<8>(packed.data(), x.data(), y32.data(), M, K);
    avx2_rows_group<16>(packed.data(), x.data(), y64.data(), M, K);
    r.exact8 = std::equal(ref.begin(), ref.end(), yeng.begin(), yeng.end(), same_float_bits);
    r.exact16 = std::equal(ref.begin(), ref.end(), y16.begin(), y16.end(), same_float_bits);
    r.exact32 = std::equal(ref.begin(), ref.end(), y32.begin(), y32.end(), same_float_bits);
    r.exact64 = std::equal(ref.begin(), ref.end(), y64.begin(), y64.end(), same_float_bits);
    if (!r.exact8 || !r.exact16 || !r.exact32 || !r.exact64) {
        std::cerr << "MISMATCH " << name << " 8=" << r.exact8 << " 16=" << r.exact16
                  << " 32=" << r.exact32 << " 64=" << r.exact64 << '\n';
    }
    using Fn = void (*)(const int32_t*, const float*, float*, size_t, size_t);
    const Fn fns[5] = {
        littlebit::binary_matvec_mult_avx2,
        avx2_rows_group<2>, avx2_rows_group<4>,
        avx2_rows_group<8>, avx2_rows_group<16>
    };
    float* outs[5] = {yeng.data(), y8.data(), y16.data(), y32.data(), y64.data()};
    for (int warm = 0; warm < 2; ++warm) {
        for (int v = 0; v < 5; ++v) fns[v](packed.data(), x.data(), outs[v], M, K);
    }

    std::vector<double> samples[5];
    volatile float sink = 0.0f;
    for (size_t round = 0; round < repeats; ++round) {
        for (int step = 0; step < 5; ++step) {
            const int v = static_cast<int>((round + step) % 5);
            auto a = Clock::now();
            fns[v](packed.data(), x.data(), outs[v], M, K);
            auto b = Clock::now();
            sink += outs[v][0];
            samples[v].push_back(std::chrono::duration<double, std::milli>(b - a).count());
        }
    }
    (void)sink;
    r.engine_ms = median(samples[0]);
    r.avx8_ms = median(samples[1]);
    r.avx16_ms = median(samples[2]);
    r.avx32_ms = median(samples[3]);
    r.avx64_ms = median(samples[4]);
    return r;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: opt5_matvec_probe <model_dir> <json_out> [repeats]\n";
        return 2;
    }
    const std::string model_dir = argv[1];
    const std::string json_out = argv[2];
    const size_t repeats = argc > 3 ? static_cast<size_t>(std::stoul(argv[3])) : 5;

    special_sweep();

    littlebit::SafeTensorsFile st;
    if (!st.open(model_dir + "/model.safetensors")) return 3;

    const char* prefixes[] = {
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
        "model.layers.0.mlp.down_proj",
    };
    std::vector<PathResult> results;
    for (const char* prefix : prefixes) {
        littlebit::LittleBitLinear lin;
        if (!st.load_littlebit_linear(prefix, lin)) {
            std::cerr << "Cannot load " << prefix << '\n';
            return 4;
        }
        const std::string base = prefix;
        results.push_back(measure(base + ".V", lin.V_packed, lin.split_dim, lin.in_features, repeats));
        results.push_back(measure(base + ".U", lin.U_packed, lin.out_features, lin.split_dim, repeats));
        if (lin.has_residual) {
            results.push_back(measure(base + ".V_R", lin.V_R_packed, lin.split_dim, lin.in_features, repeats));
            results.push_back(measure(base + ".U_R", lin.U_R_packed, lin.out_features, lin.split_dim, repeats));
        }
    }

    std::cout << "path,M,K,engine_ms,avx8_ms,avx16_ms,avx32_ms,avx64_ms,exact8,exact16,exact32,exact64\n";
    double t_engine = 0, t8 = 0, t16 = 0, t32 = 0, t64 = 0;
    bool all_exact = true;
    for (const auto& r : results) {
        std::cout << r.name << ',' << r.M << ',' << r.K << ',' << r.engine_ms << ','
                  << r.avx8_ms << ',' << r.avx16_ms << ',' << r.avx32_ms << ',' << r.avx64_ms << ','
                  << r.exact8 << ',' << r.exact16 << ',' << r.exact32 << ',' << r.exact64 << '\n';
        t_engine += r.engine_ms; t8 += r.avx8_ms; t16 += r.avx16_ms;
        t32 += r.avx32_ms; t64 += r.avx64_ms;
        all_exact = all_exact && r.exact8 && r.exact16 && r.exact32 && r.exact64;
    }
    std::cout << "TOTAL," << t_engine << ',' << t8 << ',' << t16 << ',' << t32 << ',' << t64
              << ",exact=" << (all_exact ? "PASS" : "FAIL") << '\n';

    FILE* f = std::fopen(json_out.c_str(), "w");
    if (!f) return 5;
    std::fprintf(f, "{\n  \"model_dir\": \"%s\",\n  \"repeats\": %zu,\n  \"all_exact\": %s,\n  \"paths\": [\n",
                 model_dir.c_str(), repeats, all_exact ? "true" : "false");
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        std::fprintf(f,
            "    {\"name\": \"%s\", \"M\": %zu, \"K\": %zu, \"bytes\": %llu, "
            "\"engine_ms\": %.6f, \"avx8_ms\": %.6f, \"avx16_ms\": %.6f, "
            "\"avx32_ms\": %.6f, \"avx64_ms\": %.6f, "
            "\"exact8\": %s, \"exact16\": %s, \"exact32\": %s, \"exact64\": %s}%s\n",
            r.name.c_str(), r.M, r.K, static_cast<unsigned long long>(r.bytes),
            r.engine_ms, r.avx8_ms, r.avx16_ms, r.avx32_ms, r.avx64_ms,
            r.exact8 ? "true" : "false", r.exact16 ? "true" : "false",
            r.exact32 ? "true" : "false", r.exact64 ? "true" : "false",
            i + 1 == results.size() ? "" : ",");
    }
    std::fprintf(f, "  ]\n}\n");
    std::fclose(f);
    return all_exact ? 0 : 6;
}
