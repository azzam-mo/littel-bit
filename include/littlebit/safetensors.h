#pragma once

#include "littlebit/common.h"
#include "littlebit/linear.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace littlebit {

enum class DType {
    BF16,
    F16,
    F32,
    I32,
    I64,
    U8,
    UNKNOWN
};

struct SafeTensorInfo {
    std::string name;
    DType dtype = DType::UNKNOWN;
    std::vector<int64_t> shape;
    size_t offset_begin = 0;
    size_t offset_end = 0;
    const uint8_t* data_ptr = nullptr;

    size_t byte_size() const {
        return offset_end - offset_begin;
    }

    size_t numel() const {
        if (shape.empty()) return 0;
        size_t n = 1;
        for (auto d : shape) n *= static_cast<size_t>(d);
        return n;
    }
};

class SafeTensorsFile {
public:
    SafeTensorsFile() = default;
    ~SafeTensorsFile();

    // Disallow copy, allow move
    SafeTensorsFile(const SafeTensorsFile&) = delete;
    SafeTensorsFile& operator=(const SafeTensorsFile&) = delete;
    SafeTensorsFile(SafeTensorsFile&& other) noexcept;
    SafeTensorsFile& operator=(SafeTensorsFile&& other) noexcept;

    // Open and mmap file
    bool open(const std::string& filepath);
    void close();

    bool is_open() const { return mmap_addr_ != nullptr; }
    size_t file_size() const { return file_size_; }

    bool has_tensor(const std::string& name) const;
    const SafeTensorInfo& get_tensor_info(const std::string& name) const;

    // Load tensor data directly into float vector (automatically converts BF16/FP16/FP32 to float)
    bool load_tensor_to_float(const std::string& name, std::vector<float>& out) const;

    // Load tensor data directly into int32 vector
    bool load_tensor_to_i32(const std::string& name, std::vector<int32_t>& out) const;

    // Load a complete LittleBitLinear layer from safetensors
    bool load_littlebit_linear(const std::string& prefix, LittleBitLinear& layer) const;

    const std::unordered_map<std::string, SafeTensorInfo>& all_tensors() const {
        return tensors_;
    }

private:
    int fd_ = -1;
    void* mmap_addr_ = nullptr;
    size_t file_size_ = 0;
    size_t header_size_ = 0;
    const uint8_t* data_offset_base_ = nullptr;

    std::unordered_map<std::string, SafeTensorInfo> tensors_;
};

} // namespace littlebit
