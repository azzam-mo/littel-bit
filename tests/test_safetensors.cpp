#include "littlebit/safetensors.h"
#include <iostream>
#include <fstream>
#include <cassert>
#include <vector>
#include <cstring>
#include <cmath>

// Helper to write a mini valid safetensors file
void create_dummy_safetensors(const std::string& filename) {
    // We will store:
    // 1. "test.U_packed": I32, shape [2, 1], 2 * 4 = 8 bytes
    // 2. "test.V_packed": I32, shape [3, 1], 3 * 4 = 12 bytes
    // 3. "test.u1": BF16, shape [2], 2 * 2 = 4 bytes
    // 4. "test.u2": BF16, shape [3], 3 * 2 = 6 bytes
    // 5. "test.v1": BF16, shape [3], 3 * 2 = 6 bytes
    // 6. "test.v2": BF16, shape [4], 4 * 2 = 8 bytes

    size_t offset = 0;
    size_t u_pack_start = offset; offset += 8;
    size_t u_pack_end = offset;

    size_t v_pack_start = offset; offset += 12;
    size_t v_pack_end = offset;

    size_t u1_start = offset; offset += 4;
    size_t u1_end = offset;

    size_t u2_start = offset; offset += 6;
    size_t u2_end = offset;

    size_t v1_start = offset; offset += 6;
    size_t v1_end = offset;

    size_t v2_start = offset; offset += 8;
    size_t v2_end = offset;

    std::string json_header = "{"
        "\"test.U_packed\":{\"dtype\":\"I32\",\"shape\":[2,1],\"data_offsets\":[" + std::to_string(u_pack_start) + "," + std::to_string(u_pack_end) + "]},"
        "\"test.V_packed\":{\"dtype\":\"I32\",\"shape\":[3,1],\"data_offsets\":[" + std::to_string(v_pack_start) + "," + std::to_string(v_pack_end) + "]},"
        "\"test.u1\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[" + std::to_string(u1_start) + "," + std::to_string(u1_end) + "]},"
        "\"test.u2\":{\"dtype\":\"BF16\",\"shape\":[3],\"data_offsets\":[" + std::to_string(u2_start) + "," + std::to_string(u2_end) + "]},"
        "\"test.v1\":{\"dtype\":\"BF16\",\"shape\":[3],\"data_offsets\":[" + std::to_string(v1_start) + "," + std::to_string(v1_end) + "]},"
        "\"test.v2\":{\"dtype\":\"BF16\",\"shape\":[4],\"data_offsets\":[" + std::to_string(v2_start) + "," + std::to_string(v2_end) + "]}"
    "}";

    uint64_t header_len = json_header.size();

    std::ofstream out(filename, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&header_len), 8);
    out.write(json_header.data(), header_len);

    // Write binary data:
    // U_packed: 2 ints
    int32_t u_data[2] = {12345, 67890};
    out.write(reinterpret_cast<const char*>(u_data), sizeof(u_data));

    // V_packed: 3 ints
    int32_t v_data[3] = {111, 222, 333};
    out.write(reinterpret_cast<const char*>(v_data), sizeof(v_data));

    // u1: 2 bf16 floats (e.g. 1.5f, 2.5f)
    uint16_t u1_data[2] = {littlebit::fp32_to_bf16(1.5f), littlebit::fp32_to_bf16(2.5f)};
    out.write(reinterpret_cast<const char*>(u1_data), sizeof(u1_data));

    // u2: 3 bf16 floats (0.5f, 1.0f, 1.5f)
    uint16_t u2_data[3] = {littlebit::fp32_to_bf16(0.5f), littlebit::fp32_to_bf16(1.0f), littlebit::fp32_to_bf16(1.5f)};
    out.write(reinterpret_cast<const char*>(u2_data), sizeof(u2_data));

    // v1: 3 bf16 floats (2.0f, 3.0f, 4.0f)
    uint16_t v1_data[3] = {littlebit::fp32_to_bf16(2.0f), littlebit::fp32_to_bf16(3.0f), littlebit::fp32_to_bf16(4.0f)};
    out.write(reinterpret_cast<const char*>(v1_data), sizeof(v1_data));

    // v2: 4 bf16 floats (0.1f, 0.2f, 0.3f, 0.4f)
    uint16_t v2_data[4] = {littlebit::fp32_to_bf16(0.1f), littlebit::fp32_to_bf16(0.2f), littlebit::fp32_to_bf16(0.3f), littlebit::fp32_to_bf16(0.4f)};
    out.write(reinterpret_cast<const char*>(v2_data), sizeof(v2_data));

    out.close();
}

int main() {
    std::cout << "[TEST] Running SafeTensorsFile unit tests..." << std::endl;

    std::string test_path = "/tmp/test_dummy.safetensors";
    create_dummy_safetensors(test_path);

    littlebit::SafeTensorsFile file;
    bool ok = file.open(test_path);
    assert(ok && "Failed to open safetensors file");
    assert(file.is_open());

    assert(file.has_tensor("test.U_packed"));
    assert(file.has_tensor("test.V_packed"));
    assert(file.has_tensor("test.u1"));
    assert(file.has_tensor("test.u2"));
    assert(file.has_tensor("test.v1"));
    assert(file.has_tensor("test.v2"));

    // Check tensor data
    std::vector<int32_t> u_pack;
    assert(file.load_tensor_to_i32("test.U_packed", u_pack));
    assert(u_pack.size() == 2);
    assert(u_pack[0] == 12345);
    assert(u_pack[1] == 67890);
    std::cout << "  ✓ I32 tensor loaded correctly." << std::endl;

    std::vector<float> u1_floats;
    assert(file.load_tensor_to_float("test.u1", u1_floats));
    assert(u1_floats.size() == 2);
    assert(std::fabs(u1_floats[0] - 1.5f) < 1e-3f);
    assert(std::fabs(u1_floats[1] - 2.5f) < 1e-3f);
    std::cout << "  ✓ BF16 tensor converted to FP32 correctly." << std::endl;

    // Check layer loading
    littlebit::LittleBitLinear layer;
    assert(file.load_littlebit_linear("test", layer));
    assert(layer.in_features == 4);
    assert(layer.out_features == 2);
    assert(layer.split_dim == 3);
    assert(layer.U_packed[0] == 12345);
    assert(layer.V_packed[0] == 111);
    assert(std::fabs(layer.u1[0] - 1.5f) < 1e-3f);
    assert(std::fabs(layer.v2[3] - 0.4f) < 1e-2f);
    std::cout << "  ✓ load_littlebit_linear loaded layer correctly." << std::endl;

    file.close();
    assert(!file.is_open());

    std::remove(test_path.c_str());
    std::cout << "All SafeTensorsFile unit tests passed successfully!" << std::endl;
    return 0;
}
