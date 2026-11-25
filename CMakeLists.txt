cmake_minimum_required(VERSION 3.16)
project(littlebit_loader LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-g -O0 -Wall -Wextra)
else()
    add_compile_options(-O3 -Wall -Wextra)
endif()

# Bit-exact kernels must not be reassociated or contracted into FMAs.
add_compile_options(-fno-fast-math -ffp-contract=off)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Engine Core Library
set(ENGINE_SOURCES
    src/binary_packer.cpp
    src/scalar_kernels.cpp
    src/linear.cpp
    src/safetensors.cpp
    src/tokenizer.cpp
    src/ops.cpp
    src/attention.cpp
    src/mlp.cpp
    src/transformer_block.cpp
    src/model.cpp
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    list(APPEND ENGINE_SOURCES
        src/binary_packer_neon.cpp
        src/lm_head_scalar.cpp
    )
else()
    list(APPEND ENGINE_SOURCES
        src/binary_packer_avx2.cpp
        src/lm_head_avx2.cpp
    )
endif()

add_library(littlebit_engine STATIC ${ENGINE_SOURCES})

# Unit Tests
enable_testing()

add_executable(test_binary_packer tests/test_binary_packer.cpp)
target_link_libraries(test_binary_packer PRIVATE littlebit_engine)
add_test(NAME test_binary_packer COMMAND test_binary_packer)

add_executable(test_littlebit_linear tests/test_littlebit_linear.cpp)
target_link_libraries(test_littlebit_linear PRIVATE littlebit_engine)
add_test(NAME test_littlebit_linear COMMAND test_littlebit_linear)

add_executable(test_safetensors tests/test_safetensors.cpp)
target_link_libraries(test_safetensors PRIVATE littlebit_engine)
add_test(NAME test_safetensors COMMAND test_safetensors)

add_executable(test_tokenizer tests/test_tokenizer.cpp)
target_link_libraries(test_tokenizer PRIVATE littlebit_engine)
add_test(NAME test_tokenizer COMMAND test_tokenizer)

add_executable(test_ops tests/test_ops.cpp)
target_link_libraries(test_ops PRIVATE littlebit_engine)
add_test(NAME test_ops COMMAND test_ops)

add_executable(test_transformer_block tests/test_transformer_block.cpp)
target_link_libraries(test_transformer_block PRIVATE littlebit_engine)
add_test(NAME test_transformer_block COMMAND test_transformer_block)

# Full inference-path smoke test with a generated tiny Qwen3-shaped model.
find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(TINY_MODEL_DIR "${CMAKE_CURRENT_BINARY_DIR}/tiny-model")
add_custom_command(
    OUTPUT "${TINY_MODEL_DIR}/model.safetensors"
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tests/make_tiny_model.py" "${TINY_MODEL_DIR}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/make_tiny_model.py"
    COMMENT "Generating tiny synthetic model fixture"
    VERBATIM
)
add_custom_target(tiny_model_fixture DEPENDS "${TINY_MODEL_DIR}/model.safetensors")

# Structural check of the generated Xcode project (no macOS/Xcode required).
add_test(NAME check_xcode_project
         COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_xcode_project.py")

add_executable(test_model_inference tests/test_model_inference.cpp)
add_dependencies(test_model_inference tiny_model_fixture)
target_link_libraries(test_model_inference PRIVATE littlebit_engine)
add_test(NAME test_model_inference COMMAND test_model_inference "${TINY_MODEL_DIR}")

# In-process repeated load/prefill/cache-clear/scope-exit memory probe.
add_executable(memory_cycle_probe tests/memory_cycle_probe.cpp)
add_dependencies(memory_cycle_probe tiny_model_fixture)
target_link_libraries(memory_cycle_probe PRIVATE littlebit_engine)
add_test(NAME memory_cycle_probe COMMAND memory_cycle_probe "${TINY_MODEL_DIR}")

# Standalone probes for validating real weights / tokenizer against independent references.
add_executable(model_probe tests/model_probe.cpp)
target_link_libraries(model_probe PRIVATE littlebit_engine)

add_executable(tokenizer_probe tests/tokenizer_probe.cpp)
target_link_libraries(tokenizer_probe PRIVATE littlebit_engine)

add_executable(logits_probe tests/logits_probe.cpp)
target_link_libraries(logits_probe PRIVATE littlebit_engine)

add_executable(real_model_smoke tests/real_model_smoke.cpp)
target_link_libraries(real_model_smoke PRIVATE littlebit_engine)

add_executable(perf_benchmark tests/perf_benchmark.cpp)
target_link_libraries(perf_benchmark PRIVATE littlebit_engine)

add_executable(binary_matvec_probe tests/binary_matvec_probe.cpp)
target_link_libraries(binary_matvec_probe PRIVATE littlebit_engine)
