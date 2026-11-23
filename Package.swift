// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "LittleBitLoader",
    platforms: [
        .iOS(.v16),
        .macOS(.v13)
    ],
    products: [
        .library(
            name: "LittleBitCore",
            targets: ["LittleBitCore"]
        )
    ],
    targets: [
        .target(
            name: "LittleBitCore",
            path: ".",
            sources: [
                "src/binary_packer.cpp",
                "src/linear.cpp",
                "src/safetensors.cpp",
                "src/tokenizer.cpp",
                "src/ops.cpp",
                "src/attention.cpp",
                "src/mlp.cpp",
                "src/transformer_block.cpp",
                "src/model.cpp"
            ],
            publicHeadersPath: "include",
            cxxSettings: [
                .headerSearchPath("include"),
                .unsafeFlags(["-std=c++17", "-O3"])
            ]
        )
    ],
    cxxLanguageStandard: .cxx17
)
