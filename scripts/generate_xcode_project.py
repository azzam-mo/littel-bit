#!/usr/bin/env python3
"""Generate LittleBitLoader.xcodeproj without needing XcodeGen.

Runs on any platform. The generated project is plain text and can be opened
directly in Xcode on macOS:

    python3 scripts/generate_xcode_project.py
    open LittleBitLoader.xcodeproj

Set DEVELOPMENT_TEAM=<your Apple Team ID> in the environment to embed the
signing team; otherwise choose it manually in Xcode before running on device.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

APP_SWIFT = [
    "ios/LittleBitLoader/LittleBitLoaderApp.swift",
    "ios/LittleBitLoader/Bridge/LittleBitEngine.swift",
    "ios/LittleBitLoader/Views/HomeView.swift",
    "ios/LittleBitLoader/Views/ModelManagerView.swift",
    "ios/LittleBitLoader/Views/ChatView.swift",
    "ios/LittleBitLoader/Views/PerformanceMonitorView.swift",
]
BRIDGE_MM = ["ios/LittleBitLoader/Bridge/LittleBitEngineBridge.mm"]
ENGINE_CPP = [
    "src/binary_packer.cpp", "src/scalar_kernels.cpp",
    "src/binary_packer_neon.cpp", "src/lm_head_scalar.cpp",
    "src/linear.cpp", "src/safetensors.cpp",
    "src/tokenizer.cpp", "src/ops.cpp", "src/attention.cpp", "src/mlp.cpp",
    "src/transformer_block.cpp", "src/model.cpp",
]
ENGINE_HEADERS = [
    "include/littlebit/binary_packer.h", "include/littlebit/common.h",
    "include/littlebit/scalar_kernels.h",
    "include/littlebit/linear.h", "include/littlebit/safetensors.h",
    "include/littlebit/tokenizer.h", "include/littlebit/ops.h",
    "include/littlebit/attention.h", "include/littlebit/mlp.h",
    "include/littlebit/transformer_block.h", "include/littlebit/model.h",
    "include/littlebit/unicode_ranges.h", "include/nlohmann/json.hpp",
]
BRIDGE_HEADERS = [
    "ios/LittleBitLoader/Bridge/LittleBitEngineBridge.h",
    "ios/LittleBitLoader/LittleBitLoader-Bridging-Header.h",
]
INFO_PLIST = "ios/LittleBitLoader/Info.plist"
TEST_SWIFT = ["ios/LittleBitLoaderTests/EngineSmokeTests.swift"]

_counter = 0


def uid():
    global _counter
    _counter += 1
    return f"{_counter:024X}"


def q(value):
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'


def file_type(path):
    if path.endswith(".swift"):
        return "sourcecode.swift"
    if path.endswith(".mm"):
        return "sourcecode.cpp.objcpp"
    if path.endswith(".cpp"):
        return "sourcecode.cpp.cpp"
    if path.endswith(".h"):
        return "sourcecode.c.h"
    if path.endswith(".plist"):
        return "text.plist.xml"
    return "text"


def group(children, name, path=""):
    gid = uid()
    entries = " ".join(f"{c} /* {n} */," for c, n in children)
    path_line = f"\n\t\t\tpath = {q(path)};" if path else ""
    return gid, (
        f"\t\t{gid} /* {name} */ = {{\n"
        f"\t\t\tisa = PBXGroup;\n"
        f"\t\t\tchildren = (\n\t\t\t\t{entries}\n\t\t\t);\n"
        f"\t\t\tsourceTree = \"<group>\";{path_line}\n"
        f"\t\t}};\n"
    )


def main():
    missing = [p for p in APP_SWIFT + BRIDGE_MM + ENGINE_CPP + ENGINE_HEADERS +
               BRIDGE_HEADERS + [INFO_PLIST] + TEST_SWIFT if not os.path.isfile(os.path.join(ROOT, p))]
    if missing:
        print("Missing source files:", *missing, sep="\n  ")
        return 1

    # File references
    file_refs = {}
    lines = []
    for path in dict.fromkeys(APP_SWIFT + BRIDGE_MM + ENGINE_CPP + ENGINE_HEADERS +
                              BRIDGE_HEADERS + [INFO_PLIST] + TEST_SWIFT):
        fid = uid()
        file_refs[path] = fid
        name = os.path.basename(path)
        src_tree = '<group>' if path != INFO_PLIST or True else '<group>'
        lines.append(
            f"\t\t{fid} /* {name} */ = {{isa = PBXFileReference; "
            f"lastKnownFileType = {file_type(path)}; "
            f"path = {q(path)}; sourceTree = {q(src_tree)}; }};\n"
        )

    foundation_ref = uid()
    lines.append(
        f"\t\t{foundation_ref} /* Foundation.framework */ = {{isa = PBXFileReference; "
        f"lastKnownFileType = wrapper.framework; name = Foundation.framework; "
        f"path = System/Library/Frameworks/Foundation.framework; sourceTree = SDKROOT; }};\n"
    )
    xctest_ref = uid()
    lines.append(
        f"\t\t{xctest_ref} /* XCTest.framework */ = {{isa = PBXFileReference; "
        f"lastKnownFileType = wrapper.framework; name = XCTest.framework; "
        f"path = System/Library/Frameworks/XCTest.framework; sourceTree = SDKROOT; }};\n"
    )
    app_product_ref = uid()
    lines.append(
        f"\t\t{app_product_ref} /* LittleBitLoader.app */ = {{isa = PBXFileReference; "
        f"explicitFileType = wrapper.application; includeInIndex = 0; "
        f"path = LittleBitLoader.app; sourceTree = BUILT_PRODUCTS_DIR; }};\n"
    )
    test_product_ref = uid()
    lines.append(
        f"\t\t{test_product_ref} /* LittleBitLoaderTests.xctest */ = {{isa = PBXFileReference; "
        f"explicitFileType = wrapper.cfbundle; includeInIndex = 0; "
        f"path = LittleBitLoaderTests.xctest; sourceTree = BUILT_PRODUCTS_DIR; }};\n"
    )

    # Groups
    def ref_node(path):
        return (file_refs[path], os.path.basename(path))

    app_group, app_group_def = group([ref_node(p) for p in APP_SWIFT + BRIDGE_MM + BRIDGE_HEADERS + [INFO_PLIST]],
                                     "App", "ios/LittleBitLoader")
    engine_group, engine_group_def = group([ref_node(p) for p in ENGINE_CPP + ENGINE_HEADERS], "Engine", ".")
    tests_group, tests_group_def = group([ref_node(p) for p in TEST_SWIFT], "Tests", "ios/LittleBitLoaderTests")
    frameworks_group, frameworks_group_def = group(
        [(foundation_ref, "Foundation.framework"), (xctest_ref, "XCTest.framework")], "Frameworks")
    products_group, products_group_def = group(
        [(app_product_ref, "LittleBitLoader.app"), (test_product_ref, "LittleBitLoaderTests.xctest")], "Products")
    main_group, main_group_def = group(
        [(app_group, "App"), (engine_group, "Engine"), (tests_group, "Tests"),
         (frameworks_group, "Frameworks"), (products_group, "Products")], "")

    # Build files
    build_files = {}

    def build_file(path, target):
        key = (path, target)
        if key in build_files:
            return build_files[key]
        bid = uid()
        build_files[key] = bid
        lines.append(
            f"\t\t{bid} /* {os.path.basename(path)} in {target} Sources */ = "
            f"{{isa = PBXBuildFile; fileRef = {file_refs[path]} /* {os.path.basename(path)} */; }};\n"
        )
        return bid

    def framework_build_file(ref, name, target):
        bid = uid()
        lines.append(
            f"\t\t{bid} /* {name} in {target} Frameworks */ = {{isa = PBXBuildFile; "
            f"fileRef = {ref} /* {name} */; }};\n"
        )
        return bid

    fw_app = framework_build_file(foundation_ref, "Foundation.framework", "app")
    fw_test_f = framework_build_file(foundation_ref, "Foundation.framework", "tests")
    fw_test_x = framework_build_file(xctest_ref, "XCTest.framework", "tests")

    def source_phase(target, paths):
        pid = uid()
        entries = "\n".join(f"\t\t\t\t{build_file(p, target)} /* {os.path.basename(p)} in Sources */,"
                            for p in paths)
        return pid, (
            f"\t\t{pid} /* Sources */ = {{\n"
            f"\t\t\tisa = PBXSourcesBuildPhase;\n"
            f"\t\t\tbuildActionMask = 2147483647;\n"
            f"\t\t\tfiles = (\n{entries}\n\t\t\t);\n"
            f"\t\t\trunOnlyForDeploymentPostprocessing = 0;\n"
            f"\t\t}};\n"
        )

    def framework_phase(target, refs):
        pid = uid()
        entries = "\n".join(f"\t\t\t\t{r} /* in Frameworks */," for r in refs)
        return pid, (
            f"\t\t{pid} /* Frameworks */ = {{\n"
            f"\t\t\tisa = PBXFrameworksBuildPhase;\n"
            f"\t\t\tbuildActionMask = 2147483647;\n"
            f"\t\t\tfiles = (\n{entries}\n\t\t\t);\n"
            f"\t\t\trunOnlyForDeploymentPostprocessing = 0;\n"
            f"\t\t}};\n"
        )

    def resources_phase():
        pid = uid()
        return pid, (
            f"\t\t{pid} /* Resources */ = {{\n"
            f"\t\t\tisa = PBXResourcesBuildPhase;\n"
            f"\t\t\tbuildActionMask = 2147483647;\n"
            f"\t\t\tfiles = (\n\t\t\t);\n"
            f"\t\t\trunOnlyForDeploymentPostprocessing = 0;\n"
            f"\t\t}};\n"
        )

    app_src_phase, app_src_def = source_phase("app", APP_SWIFT + BRIDGE_MM + ENGINE_CPP)
    app_fw_phase, app_fw_def = framework_phase("app", [fw_app])
    app_res_phase, app_res_def = resources_phase()
    test_src_phase, test_src_def = source_phase(
        "tests", [p for p in APP_SWIFT if "App.swift" not in p] +
        TEST_SWIFT + BRIDGE_MM + ENGINE_CPP)
    test_fw_phase, test_fw_def = framework_phase("tests", [fw_test_f, fw_test_x])
    test_res_phase, test_res_def = resources_phase()

    development_team = os.environ.get("DEVELOPMENT_TEAM", "").strip()
    common_settings = [
        "CLANG_CXX_LANGUAGE_STANDARD = \"c++17\";",
        "CLANG_CXX_LIBRARY = \"libc++\";",
        "CLANG_ENABLE_MODULES = YES;",
        "CLANG_ENABLE_OBJC_ARC = YES;",
        "GCC_PRECOMPILE_PREFIX_HEADER = NO;",
        "HEADER_SEARCH_PATHS = ( \"$(inherited)\", \"$(SRCROOT)/include\", \"$(SRCROOT)\", \"$(SRCROOT)/ios/LittleBitLoader/Bridge\" );",
        "IPHONEOS_DEPLOYMENT_TARGET = 16.0;",
        "OTHER_CPLUSPLUSFLAGS = ( \"$(inherited)\", \"-O3\", \"-fno-fast-math\", \"-ffp-contract=off\" );",
        "SDKROOT = iphoneos;",
        "SWIFT_OBJC_BRIDGING_HEADER = \"$(SRCROOT)/ios/LittleBitLoader/LittleBitLoader-Bridging-Header.h\";",
        "SWIFT_VERSION = 5.0;",
        *([f"DEVELOPMENT_TEAM = {development_team};"] if development_team else []),
        "TARGETED_DEVICE_FAMILY = \"1,2\";",
    ]

    app_config, app_config_def = uid(), None
    app_config_def = (
        f"\t\t{app_config} /* Debug */ = {{\n"
        f"\t\t\tisa = XCBuildConfiguration;\n"
        f"\t\t\tbuildSettings = {{\n"
        + "".join(f"\t\t\t\t{s}\n" for s in [
            *common_settings,
            "\"ARCHS[sdk=iphoneos*]\" = arm64;",
            "CODE_SIGN_STYLE = Automatic;",
            "ENABLE_TESTABILITY = YES;",
            f"INFOPLIST_FILE = {q(INFO_PLIST)};",
            "PRODUCT_BUNDLE_IDENTIFIER = com.littlebit.loader;",
            "PRODUCT_NAME = LittleBitLoader;",
        ])
        + f"\t\t\t}};\n\t\t\tname = Debug;\n\t\t}};\n"
    )
    app_config_rel, app_config_rel_def = uid(), None
    app_config_rel_def = app_config_def.replace(app_config, app_config_rel, 1).replace("name = Debug;", "name = Release;")

    test_config, test_config_def = uid(), None
    test_config_def = (
        f"\t\t{test_config} /* Debug */ = {{\n"
        f"\t\t\tisa = XCBuildConfiguration;\n"
        f"\t\t\tbuildSettings = {{\n"
        + "".join(f"\t\t\t\t{s}\n" for s in [
            *common_settings,
            "CODE_SIGN_STYLE = Automatic;",
            "PRODUCT_BUNDLE_IDENTIFIER = com.littlebit.loader.tests;",
            "PRODUCT_NAME = LittleBitLoaderTests;",
        ])
        + f"\t\t\t}};\n\t\t\tname = Debug;\n\t\t}};\n"
    )
    test_config_rel, test_config_rel_def = uid(), None
    test_config_rel_def = test_config_def.replace(test_config, test_config_rel, 1).replace("name = Debug;", "name = Release;")

    app_target, app_target_def = uid(), None
    app_target_def = (
        f"\t\t{app_target} /* LittleBitLoader */ = {{\n"
        f"\t\t\tisa = PBXNativeTarget;\n"
        f"\t\t\tbuildConfigurationList = PLACEHOLDER_APP_LIST;\n"
        f"\t\t\tbuildPhases = ( {app_src_phase} /* Sources */, {app_fw_phase} /* Frameworks */, {app_res_phase} /* Resources */ );\n"
        f"\t\t\tbuildRules = ( );\n"
        f"\t\t\tdependencies = ( );\n"
        f"\t\t\tname = LittleBitLoader;\n"
        f"\t\t\tproductName = LittleBitLoader;\n"
        f"\t\t\tproductReference = {app_product_ref} /* LittleBitLoader.app */;\n"
        f"\t\t\tproductType = \"com.apple.product-type.application\";\n"
        f"\t\t}};\n"
    )
    test_target, test_target_def = uid(), None
    dependency = uid()
    test_target_def = (
        f"\t\t{test_target} /* LittleBitLoaderTests */ = {{\n"
        f"\t\t\tisa = PBXNativeTarget;\n"
        f"\t\t\tbuildConfigurationList = PLACEHOLDER_TEST_LIST;\n"
        f"\t\t\tbuildPhases = ( {test_src_phase} /* Sources */, {test_fw_phase} /* Frameworks */, {test_res_phase} /* Resources */ );\n"
        f"\t\t\tbuildRules = ( );\n"
        f"\t\t\tdependencies = ( {dependency} /* PBXTargetDependency */ );\n"
        f"\t\t\tname = LittleBitLoaderTests;\n"
        f"\t\t\tproductName = LittleBitLoaderTests;\n"
        f"\t\t\tproductReference = {test_product_ref} /* LittleBitLoaderTests.xctest */;\n"
        f"\t\t\tproductType = \"com.apple.product-type.bundle.unit-test\";\n"
        f"\t\t}};\n"
    )
    dep_def = (
        f"\t\t{dependency} /* PBXTargetDependency */ = {{\n"
        f"\t\t\tisa = PBXTargetDependency;\n"
        f"\t\t\ttarget = {app_target} /* LittleBitLoader */;\n"
        f"\t\t\ttargetProxy = PLACEHOLDER_PROXY;\n"
        f"\t\t}};\n"
    )
    proxy = uid()
    proxy_def = (
        f"\t\t{proxy} /* PBXContainerItemProxy */ = {{\n"
        f"\t\t\tisa = PBXContainerItemProxy;\n"
        f"\t\t\tcontainerPortal = PROJECT_ID;\n"
        f"\t\t\tproxyType = 1;\n"
        f"\t\t\tremoteGlobalIDString = {app_target};\n"
        f"\t\t\tremoteInfo = LittleBitLoader;\n"
        f"\t\t}};\n"
    )
    app_list, app_list_def = uid(), None
    app_list_def = (
        f"\t\t{app_list} /* Build configuration list for PBXNativeTarget \"LittleBitLoader\" */ = {{\n"
        f"\t\t\tisa = XCConfigurationList;\n"
        f"\t\t\tbuildConfigurations = ( {app_config} /* Debug */, {app_config_rel} /* Release */ );\n"
        f"\t\t\tdefaultConfigurationIsVisible = 0;\n"
        f"\t\t\tdefaultConfigurationName = Release;\n"
        f"\t\t}};\n"
    )
    test_list, test_list_def = uid(), None
    test_list_def = (
        f"\t\t{test_list} /* Build configuration list for PBXNativeTarget \"LittleBitLoaderTests\" */ = {{\n"
        f"\t\t\tisa = XCConfigurationList;\n"
        f"\t\t\tbuildConfigurations = ( {test_config} /* Debug */, {test_config_rel} /* Release */ );\n"
        f"\t\t\tdefaultConfigurationIsVisible = 0;\n"
        f"\t\t\tdefaultConfigurationName = Release;\n"
        f"\t\t}};\n"
    )
    project_id = uid()
    project_def = (
        f"\t\t{project_id} /* Project object */ = {{\n"
        f"\t\t\tisa = PBXProject;\n"
        f"\t\t\tattributes = {{ LastUpgradeCheck = 1500; }};\n"
        f"\t\t\tbuildConfigurationList = PLACEHOLDER_PROJ_LIST;\n"
        f"\t\t\tcompatibilityVersion = \"Xcode 14.0\";\n"
        f"\t\t\tdevelopmentRegion = en;\n"
        f"\t\t\thasScannedForEncodings = 0;\n"
        f"\t\t\tknownRegions = ( en, ar );\n"
        f"\t\t\tmainGroup = {main_group};\n"
        f"\t\t\tproductRefGroup = {products_group} /* Products */;\n"
        f"\t\t\tprojectDirPath = \"\";\n"
        f"\t\t\tprojectRoot = \"\";\n"
        f"\t\t\ttargets = ( {app_target} /* LittleBitLoader */, {test_target} /* LittleBitLoaderTests */ );\n"
        f"\t\t}};\n"
    )
    proj_list, proj_list_def = uid(), None
    proj_list_def = (
        f"\t\t{proj_list} /* Build configuration list for PBXProject \"LittleBitLoader\" */ = {{\n"
        f"\t\t\tisa = XCConfigurationList;\n"
        f"\t\t\tbuildConfigurations = ( PROJ_DEBUG, PROJ_RELEASE );\n"
        f"\t\t\tdefaultConfigurationIsVisible = 0;\n"
        f"\t\t\tdefaultConfigurationName = Release;\n"
        f"\t\t}};\n"
    )
    proj_debug, proj_debug_def = uid(), None
    proj_debug_def = (
        f"\t\t{proj_debug} /* Debug */ = {{\n\t\t\tisa = XCBuildConfiguration;\n"
        f"\t\t\tbuildSettings = {{ ALWAYS_SEARCH_USER_PATHS = NO; SDKROOT = iphoneos; }};\n"
        f"\t\t\tname = Debug;\n\t\t}};\n"
    )
    proj_release, proj_release_def = uid(), None
    proj_release_def = (
        f"\t\t{proj_release} /* Release */ = {{\n\t\t\tisa = XCBuildConfiguration;\n"
        f"\t\t\tbuildSettings = {{ ALWAYS_SEARCH_USER_PATHS = NO; SDKROOT = iphoneos; }};\n"
        f"\t\t\tname = Release;\n\t\t}};\n"
    )

    body = "".join(lines) + "".join([
        app_group_def, engine_group_def, tests_group_def, frameworks_group_def,
        products_group_def, main_group_def,
        app_src_def, app_fw_def, app_res_def, test_src_def, test_fw_def, test_res_def,
        app_target_def, test_target_def, dep_def, proxy_def,
        app_config_def, app_config_rel_def, test_config_def, test_config_rel_def,
        app_list_def, test_list_def, project_def, proj_list_def, proj_debug_def, proj_release_def,
    ])
    body = (body
            .replace("PLACEHOLDER_APP_LIST", app_list)
            .replace("PLACEHOLDER_TEST_LIST", test_list)
            .replace("PLACEHOLDER_PROXY", proxy)
            .replace("PROJECT_ID", project_id)
            .replace("PLACEHOLDER_PROJ_LIST", proj_list)
            .replace("PROJ_DEBUG", proj_debug)
            .replace("PROJ_RELEASE", proj_release))

    out_dir = os.path.join(ROOT, "LittleBitLoader.xcodeproj")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "project.pbxproj")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("// !$*UTF8*$!\n{\n\tarchiveVersion = 1;\n\tclasses = {\n\t};\n"
                 f"\tobjectVersion = 56;\n\tobjects = {{\n{body}\t}};\n"
                 f"\trootObject = {project_id} /* Project object */;\n}}\n")
    print("Generated", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
