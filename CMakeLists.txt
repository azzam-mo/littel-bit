cmake_minimum_required(VERSION 3.16)
project(littlebit_loader LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-g -O0 -Wall -Wextra)
else()
    add_compile_options(-O3 -march=native -Wall -Wextra)
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Engine Core Library
add_library(littlebit_engine STATIC
    src/binary_packer.cpp
    src/linear.cpp
    src/safetensors.cpp
    src/tokenizer.cpp
    src/ops.cpp
    src/attention.cpp
    src/mlp.cpp
    src/transformer_block.cpp
    src/model.cpp
)

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
