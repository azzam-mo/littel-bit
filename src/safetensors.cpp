#include "littlebit/safetensors.h"
#include "nlohmann/json.hpp"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <iostream>

namespace littlebit {

SafeTensorsFile::~SafeTensorsFile() {
    close();
}

SafeTensorsFile::SafeTensorsFile(SafeTensorsFile&& other) noexcept
    : fd_(other.fd_),
      mmap_addr_(other.mmap_addr_),
      file_size_(other.file_size_),
      header_size_(other.header_size_),
      data_offset_base_(other.data_offset_base_),
      tensors_(std::move(other.tensors_)) {
    other.fd_ = -1;
    other.mmap_addr_ = nullptr;
    other.file_size_ = 0;
    other.header_size_ = 0;
    other.data_offset_base_ = nullptr;
}

SafeTensorsFile& SafeTensorsFile::operator=(SafeTensorsFile&& other) noexcept {
    if (this != &other) {
        close();
        fd_ = other.fd_;
        mmap_addr_ = other.mmap_addr_;
        file_size_ = other.file_size_;
        header_size_ = other.header_size_;
        data_offset_base_ = other.data_offset_base_;
        tensors_ = std::move(other.tensors_);

        other.fd_ = -1;
        other.mmap_addr_ = nullptr;
        other.file_size_ = 0;
        other.header_size_ = 0;
        other.data_offset_base_ = nullptr;
    }
    return *this;
}

void SafeTensorsFile::close() {
    if (mmap_addr_ != nullptr && mmap_addr_ != MAP_FAILED) {
        munmap(mmap_addr_, file_size_);
        mmap_addr_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    file_size_ = 0;
    header_size_ = 0;
    data_offset_base_ = nullptr;
    tensors_.clear();
}

bool SafeTensorsFile::open(const std::string& filepath) {
    close();

    fd_ = ::open(filepath.c_str(), O_RDONLY);
    if (fd_ < 0) {
        std::cerr << "SafeTensorsFile: Failed to open file: " << filepath << std::endl;
        return false;
    }

    struct stat sb;
    if (fstat(fd_, &sb) < 0) {
        std::cerr << "SafeTensorsFile: Failed to stat file: " << filepath << std::endl;
        close();
        return false;
    }
    file_size_ = static_cast<size_t>(sb.st_size);

    if (file_size_ < 8) {
        std::cerr << "SafeTensorsFile: File is too small (< 8 bytes)" << std::endl;
        close();
        return false;
    }

    mmap_addr_ = ::mmap(nullptr, file_size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (mmap_addr_ == MAP_FAILED) {
        std::cerr << "SafeTensorsFile: mmap failed" << std::endl;
        mmap_addr_ = nullptr;
        close();
        return false;
    }

    // Read 8 bytes header length (little-endian uint64)
    const uint8_t* byte_ptr = static_cast<const uint8_t*>(mmap_addr_);
    uint64_t h_len = 0;
    std::memcpy(&h_len, byte_ptr, sizeof(uint64_t));
    header_size_ = static_cast<size_t>(h_len);

    if (8 + header_size_ > file_size_) {
        std::cerr << "SafeTensorsFile: Corrupt file, header_size exceeds file size" << std::endl;
        close();
        return false;
    }

    data_offset_base_ = byte_ptr + 8 + header_size_;

    // Parse JSON header
    std::string header_json_str(reinterpret_cast<const char*>(byte_ptr + 8), header_size_);
    nlohmann::json header_json;
    try {
        header_json = nlohmann::json::parse(header_json_str);
    } catch (const std::exception& e) {
        std::cerr << "SafeTensorsFile: JSON parse error: " << e.what() << std::endl;
        close();
        return false;
    }

    for (auto it = header_json.begin(); it != header_json.end(); ++it) {
        if (it.key() == "__metadata__") continue;

        const auto& val = it.value();
        SafeTensorInfo info;
        info.name = it.key();

        std::string dt_str = val.value("dtype", "");
        if (dt_str == "BF16") info.dtype = DType::BF16;
        else if (dt_str == "F16") info.dtype = DType::F16;
        else if (dt_str == "F32") info.dtype = DType::F32;
        else if (dt_str == "I32") info.dtype = DType::I32;
        else if (dt_str == "I64") info.dtype = DType::I64;
        else if (dt_str == "U8") info.dtype = DType::U8;
        else info.dtype = DType::UNKNOWN;

        if (val.contains("shape") && val["shape"].is_array()) {
            for (const auto& s : val["shape"]) {
                info.shape.push_back(s.get<int64_t>());
            }
        }

        if (val.contains("data_offsets") && val["data_offsets"].is_array() && val["data_offsets"].size() == 2) {
            info.offset_begin = val["data_offsets"][0].get<size_t>();
            info.offset_end = val["data_offsets"][1].get<size_t>();

            // Reject corrupt headers instead of handing out pointers outside the mmap.
            // This is especially important for files imported from the iOS Files app.
            size_t data_size = file_size_ - (8 + header_size_);
            if (info.offset_begin > info.offset_end || info.offset_end > data_size) {
                std::cerr << "SafeTensorsFile: Invalid data_offsets for tensor: " << info.name << std::endl;
                close();
                return false;
            }
            info.data_ptr = data_offset_base_ + info.offset_begin;
        }

        tensors_[info.name] = info;
    }

    return true;
}

bool SafeTensorsFile::has_tensor(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
}

const SafeTensorInfo& SafeTensorsFile::get_tensor_info(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("SafeTensorsFile: Tensor not found: " + name);
    }
    return it->second;
}

bool SafeTensorsFile::load_tensor_to_float(const std::string& name, std::vector<float>& out) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) return false;

    const auto& info = it->second;
    size_t count = info.numel();
    out.resize(count);

    if (info.dtype == DType::BF16) {
        const uint16_t* ptr = reinterpret_cast<const uint16_t*>(info.data_ptr);
        for (size_t i = 0; i < count; ++i) {
            out[i] = bf16_to_fp32(ptr[i]);
        }
        return true;
    } else if (info.dtype == DType::F16) {
        const uint16_t* ptr = reinterpret_cast<const uint16_t*>(info.data_ptr);
        for (size_t i = 0; i < count; ++i) {
            out[i] = fp16_to_fp32(ptr[i]);
        }
        return true;
    } else if (info.dtype == DType::F32) {
        const float* ptr = reinterpret_cast<const float*>(info.data_ptr);
        std::memcpy(out.data(), ptr, count * sizeof(float));
        return true;
    }

    return false;
}

bool SafeTensorsFile::load_tensor_to_i32(const std::string& name, std::vector<int32_t>& out) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) return false;

    const auto& info = it->second;
    size_t count = info.numel();
    out.resize(count);

    if (info.dtype == DType::I32) {
        std::memcpy(out.data(), info.data_ptr, count * sizeof(int32_t));
        return true;
    }
    return false;
}

bool SafeTensorsFile::load_littlebit_linear(const std::string& prefix, LittleBitLinear& layer) const {
    std::string u_pack_name = prefix + ".U_packed";
    std::string v_pack_name = prefix + ".V_packed";
    if (!has_tensor(u_pack_name) || !has_tensor(v_pack_name)) {
        return false;
    }

    const auto& u_info = get_tensor_info(u_pack_name);
    const auto& v_info = get_tensor_info(v_pack_name);

    size_t out_feat = static_cast<size_t>(u_info.shape[0]);
    size_t split_d = static_cast<size_t>(v_info.shape[0]);

    // Read v2 to find in_features
    std::string v2_name = prefix + ".v2";
    if (!has_tensor(v2_name)) return false;
    const auto& v2_info = get_tensor_info(v2_name);
    size_t in_feat = v2_info.numel();

    bool has_res = has_tensor(prefix + ".U_R_packed");

    layer.initialize(in_feat, out_feat, split_d, has_res);

    // Load Primary Path
    load_tensor_to_i32(u_pack_name, layer.U_packed);
    load_tensor_to_i32(v_pack_name, layer.V_packed);
    load_tensor_to_float(prefix + ".u1", layer.u1);
    load_tensor_to_float(prefix + ".u2", layer.u2);
    load_tensor_to_float(prefix + ".v1", layer.v1);
    load_tensor_to_float(prefix + ".v2", layer.v2);

    // Load Residual Path
    if (has_res) {
        load_tensor_to_i32(prefix + ".U_R_packed", layer.U_R_packed);
        load_tensor_to_i32(prefix + ".V_R_packed", layer.V_R_packed);
        load_tensor_to_float(prefix + ".u1_R", layer.u1_R);
        load_tensor_to_float(prefix + ".u2_R", layer.u2_R);
        load_tensor_to_float(prefix + ".v1_R", layer.v1_R);
        load_tensor_to_float(prefix + ".v2_R", layer.v2_R);
    }

    // Optional bias
    std::string bias_name = prefix + ".bias";
    if (has_tensor(bias_name)) {
        load_tensor_to_float(bias_name, layer.bias);
    }

    return true;
}

} // namespace littlebit
