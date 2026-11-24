#include "littlebit/binary_packer.h"
#include "littlebit/safetensors.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

// Frozen baseline implementation from commit 87c2048. This file never changes;
// it is the parity oracle for any optimized kernel.
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

// Candidate A: flip the IEEE-754 sign bit directly. Operation order is
// unchanged: every sum += (+/-val) happens in the same k order as reference.
void candidate_xor_sign(
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
                uint32_t sign = (word >> b) & 1u;
                uint32_t bits;
                std::memcpy(&bits, &val, sizeof(bits));
                bits ^= sign << 31;
                std::memcpy(&val, &bits, sizeof(val));
                sum += val;
            }
        }
        y[i] = sum;
    }
}

// Candidate B: multiply by exact +/-1.0f (also bit-exact sign flip for finite
// values) and keep the same accumulation sequence.
void candidate_mul_sign(
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
                float sign = 1.0f - 2.0f * static_cast<float>((word >> b) & 1u);
                sum += x[k++] * sign;
            }
        }
        y[i] = sum;
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

struct PathResult {
    std::string name;
    size_t M = 0;
    size_t K = 0;
    uint64_t bytes = 0;
    bool bit_exact = false;
    bool xor_exact = false;
    bool mul_exact = false;
    bool avx2_exact = false;
    double engine_ms = 0;
    double reference_ms = 0;
    double xor_ms = 0;
    double mul_ms = 0;
    double avx2_ms = 0;
};

PathResult measure(
    const std::string& name,
    const std::vector<int32_t>& packed,
    size_t M,
    size_t K,
    size_t repeats
) {
    PathResult r;
    r.name = name;
    r.M = M;
    r.K = K;
    r.bytes = static_cast<uint64_t>(packed.size()) * sizeof(int32_t);
    if (packed.size() != M * ((K + 31) / 32)) {
        std::cerr << "Shape mismatch for " << name << '\n';
        std::abort();
    }

    std::vector<float> x = make_x(K);
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64)
    std::vector<float> y_ref(M, 0.0f), y_opt(M, 0.0f), y_xor(M, 0.0f), y_mul(M, 0.0f), y_avx2(M, 0.0f);
#else
    std::vector<float> y_ref(M, 0.0f), y_opt(M, 0.0f), y_xor(M, 0.0f), y_mul(M, 0.0f);
#endif
    reference_binary_matvec_mult(packed.data(), x.data(), y_ref.data(), M, K);
    littlebit::binary_matvec_mult(packed.data(), x.data(), y_opt.data(), M, K);
    candidate_xor_sign(packed.data(), x.data(), y_xor.data(), M, K);
    candidate_mul_sign(packed.data(), x.data(), y_mul.data(), M, K);
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64)
    littlebit::binary_matvec_mult_avx2(packed.data(), x.data(), y_avx2.data(), M, K);
#endif
    r.bit_exact = true;
    r.xor_exact = true;
    r.mul_exact = true;
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64)
    r.avx2_exact = true;
#endif
    for (size_t i = 0; i < M; ++i) {
        if (!same_float_bits(y_ref[i], y_opt[i])) r.bit_exact = false;
        if (!same_float_bits(y_ref[i], y_xor[i])) r.xor_exact = false;
        if (!same_float_bits(y_ref[i], y_mul[i])) r.mul_exact = false;
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64)
        if (!same_float_bits(y_ref[i], y_avx2[i])) r.avx2_exact = false;
#endif
    }
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64)
    if (!r.bit_exact || !r.xor_exact || !r.mul_exact || !r.avx2_exact) {
        std::cerr << "BIT MISMATCH in " << name
                  << " engine=" << r.bit_exact << " xor=" << r.xor_exact
                  << " mul=" << r.mul_exact << " avx2=" << r.avx2_exact << '\n';
    }
#else
    if (!r.bit_exact || !r.xor_exact || !r.mul_exact) {
        std::cerr << "BIT MISMATCH in " << name
                  << " engine=" << r.bit_exact << " xor=" << r.xor_exact
                  << " mul=" << r.mul_exact << '\n';
    }
#endif

    // Warm all implementations, then time them identically.
    for (int i = 0; i < 2; ++i) {
        reference_binary_matvec_mult(packed.data(), x.data(), y_ref.data(), M, K);
        littlebit::binary_matvec_mult(packed.data(), x.data(), y_opt.data(), M, K);
        candidate_xor_sign(packed.data(), x.data(), y_xor.data(), M, K);
        candidate_mul_sign(packed.data(), x.data(), y_mul.data(), M, K);
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64)
        littlebit::binary_matvec_mult_avx2(packed.data(), x.data(), y_avx2.data(), M, K);
#endif
    }
    volatile float sink = 0.0f;

    auto a = Clock::now();
    for (size_t rep = 0; rep < repeats; ++rep) {
        littlebit::binary_matvec_mult(packed.data(), x.data(), y_opt.data(), M, K);
        sink += y_opt[0];
    }
    auto b = Clock::now();
    r.engine_ms = std::chrono::duration<double, std::milli>(b - a).count() / repeats;

    a = Clock::now();
    for (size_t rep = 0; rep < repeats; ++rep) {
        reference_binary_matvec_mult(packed.data(), x.data(), y_ref.data(), M, K);
        sink += y_ref[0];
    }
    b = Clock::now();
    r.reference_ms = std::chrono::duration<double, std::milli>(b - a).count() / repeats;

    a = Clock::now();
    for (size_t rep = 0; rep < repeats; ++rep) {
        candidate_xor_sign(packed.data(), x.data(), y_xor.data(), M, K);
        sink += y_xor[0];
    }
    b = Clock::now();
    r.xor_ms = std::chrono::duration<double, std::milli>(b - a).count() / repeats;

    a = Clock::now();
    for (size_t rep = 0; rep < repeats; ++rep) {
        candidate_mul_sign(packed.data(), x.data(), y_mul.data(), M, K);
        sink += y_mul[0];
    }
    b = Clock::now();
    r.mul_ms = std::chrono::duration<double, std::milli>(b - a).count() / repeats;

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64)
    a = Clock::now();
    for (size_t rep = 0; rep < repeats; ++rep) {
        littlebit::binary_matvec_mult_avx2(packed.data(), x.data(), y_avx2.data(), M, K);
        sink += y_avx2[0];
    }
    b = Clock::now();
    r.avx2_ms = std::chrono::duration<double, std::milli>(b - a).count() / repeats;
#endif
    (void)sink;
    return r;
}

std::string quote(const std::string& s) {
    return "\"" + s + "\"";
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: binary_matvec_probe <model_dir> <json_out> [repeats]\n";
        return 2;
    }
    const std::string model_dir = argv[1];
    const std::string json_out = argv[2];
    const size_t repeats = (argc > 3) ? static_cast<size_t>(std::stoul(argv[3])) : 5;

    littlebit::SafeTensorsFile st;
    if (!st.open(model_dir + "/model.safetensors")) return 3;

    const char* prefixes[] = {
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.mlp.gate_proj",
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

    std::ofstream out(json_out);
    out << std::fixed << std::setprecision(6);
    out << "{\n"
        << "  \"reference\": \"branchy scalar implementation at 87c2048\",\n"
        << "  \"model_dir\": " << quote(model_dir) << ",\n"
        << "  \"repeats\": " << repeats << ",\n"
        << "  \"all_bit_exact\": " << (std::all_of(results.begin(), results.end(),
                                                  [](const PathResult& r) { return r.bit_exact; }) ? "true" : "false")
        << ",\n  \"all_avx2_bit_exact\": " << (std::all_of(results.begin(), results.end(),
                                                          [](const PathResult& r) { return r.avx2_exact; }) ? "true" : "false")
        << ",\n  \"paths\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        out << "    {\"name\": " << quote(r.name)
            << ", \"M\": " << r.M
            << ", \"K\": " << r.K
            << ", \"bytes\": " << r.bytes
            << ", \"bit_exact\": " << (r.bit_exact ? "true" : "false")
            << ", \"engine_ms_per_call\": " << r.engine_ms
            << ", \"reference_ms_per_call\": " << r.reference_ms
            << ", \"xor_ms_per_call\": " << r.xor_ms
            << ", \"mul_ms_per_call\": " << r.mul_ms
            << ", \"xor_bit_exact\": " << (r.xor_exact ? "true" : "false")
            << ", \"mul_bit_exact\": " << (r.mul_exact ? "true" : "false")
            << ", \"avx2_ms_per_call\": " << r.avx2_ms
            << ", \"avx2_bit_exact\": " << (r.avx2_exact ? "true" : "false")
            << ", \"avx2_speedup_vs_reference\": " << (r.avx2_ms > 0 ? r.reference_ms / r.avx2_ms : 0.0)
            << ", \"speedup_vs_reference\": " << (r.engine_ms > 0 ? r.reference_ms / r.engine_ms : 0.0)
            << "}" << (i + 1 == results.size() ? "\n" : ",\n");
    }
    out << "  ]\n}\n";

    std::cout << "[MATVEC] wrote " << json_out << '\n';
    for (const auto& r : results) {
        std::cout << "  " << r.name << " M=" << r.M << " K=" << r.K
                  << " engine=" << r.engine_ms << "ms ref=" << r.reference_ms
                  << "ms xor=" << r.xor_ms << "ms mul=" << r.mul_ms
                  << "ms avx2=" << r.avx2_ms << "ms"
                  << " exact=" << (r.bit_exact ? "yes" : "NO")
                  << " avx2_exact=" << (r.avx2_exact ? "yes" : "NO") << '\n';
    }
    return std::all_of(results.begin(), results.end(), [](const PathResult& r) { return r.bit_exact; }) ? 0 : 5;
}
