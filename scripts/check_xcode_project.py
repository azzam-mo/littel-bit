#!/usr/bin/env python3
"""Structurally validate the generated Xcode project on any platform.

This does NOT build or sign the app. It checks the generated pbxproj, resolves
every PBXFileReference through its PBXGroup chain the way Xcode does, verifies
build-setting paths, and checks source references and Info.plist contents.

Usage: python3 scripts/check_xcode_project.py [path/to/project.pbxproj]
"""
import os
import plistlib
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from validate_pbxproj import Parser, validate_file

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PBX = ROOT / "LittleBitLoader.xcodeproj" / "project.pbxproj"
PLIST = ROOT / "ios" / "LittleBitLoader" / "Info.plist"
ENTITLEMENTS = ROOT / "ios" / "LittleBitLoader" / "LittleBitLoader.entitlements"

SKIP_SOURCE_TREES = {"SDKROOT", "BUILT_PRODUCTS_DIR", "DEVELOPER_DIR"}


def _labels(text: str) -> Dict[str, str]:
    """Read human-readable comments for object IDs from pbxproj text."""
    return dict(re.findall(r"^\t\t([0-9A-F]{24}) /\* (.*?) \*/ = \{", text, re.M))


def _expand(value: str) -> str:
    """Expand the Xcode root variables used by this generator."""
    for name in ("SRCROOT", "PROJECT_DIR"):
        value = value.replace(f"$({name})", str(ROOT))
        value = value.replace(f"${{{name}}}", str(ROOT))
    return value


def _as_path(value: str) -> Path:
    path = Path(_expand(value))
    return path if path.is_absolute() else ROOT / path


def _resolve_group(
    objects: Dict[str, Any],
    parent_of: Dict[str, str],
    group_id: str,
    stack: Tuple[str, ...] = (),
) -> Tuple[Optional[Path], str]:
    """Resolve a PBXGroup to an absolute directory, accumulating group paths."""
    if group_id in stack:
        return None, f"PBXGroup cycle: {' -> '.join(stack + (group_id,))}"
    group = objects.get(group_id)
    if not isinstance(group, dict) or group.get("isa") != "PBXGroup":
        return None, f"{group_id} is not a PBXGroup"

    source_tree = group.get("sourceTree")
    path_value = group.get("path", "")
    if source_tree == "<group>":
        parent_id = parent_of.get(group_id)
        if parent_id:
            base, error = _resolve_group(objects, parent_of, parent_id, stack + (group_id,))
            if error:
                return None, error
        else:
            base = ROOT
        if not path_value or path_value == ".":
            return base, ""
        path = Path(path_value)
        return (path if path.is_absolute() else base / path), ""

    if source_tree == "SOURCE_ROOT":
        return (_as_path(path_value) if path_value else ROOT), ""
    if source_tree == "absolute":
        return Path(path_value), ""
    return None, f"unsupported PBXGroup sourceTree {source_tree!r}"


def _resolve_file_ref(
    objects: Dict[str, Any],
    parent_of: Dict[str, str],
    ref_id: str,
) -> Tuple[Optional[Path], str]:
    """Resolve one PBXFileReference to an absolute filesystem path."""
    ref = objects.get(ref_id)
    if not isinstance(ref, dict) or ref.get("isa") != "PBXFileReference":
        return None, f"{ref_id} is not a PBXFileReference"

    source_tree = ref.get("sourceTree")
    if source_tree in SKIP_SOURCE_TREES:
        return None, ""

    path_value = ref.get("path", ref.get("name", ""))
    if not path_value:
        return None, "empty path"

    if source_tree == "<group>":
        group_id = parent_of.get(ref_id)
        if not group_id:
            return None, "PBXFileReference with sourceTree <group> has no parent PBXGroup"
        base, error = _resolve_group(objects, parent_of, group_id)
        if error:
            return None, error
        path = Path(path_value)
        return (path if path.is_absolute() else base / path), ""

    if source_tree == "SOURCE_ROOT":
        return _as_path(path_value), ""
    if source_tree == "absolute":
        return Path(path_value), ""

    return None, f"unsupported PBXFileReference sourceTree {source_tree!r}"


def check_resolved_paths(text: str, doc: Any) -> List[str]:
    """Resolve and verify all file references and path-based build settings."""
    failures: List[str] = []
    root = doc.root
    objects = root.get("objects", {})
    label_of = _labels(text)

    parent_of: Dict[str, str] = {}
    for group_id, group in objects.items():
        if group.get("isa") != "PBXGroup":
            continue
        for child in group.get("children", []):
            parent_of[child] = group_id

    def add_missing(label: str, resolved: Path) -> None:
        failures.append(f"{label} resolves to missing file {resolved}")

    for ref_id, ref in objects.items():
        if ref.get("isa") != "PBXFileReference":
            continue
        resolved, error = _resolve_file_ref(objects, parent_of, ref_id)
        if error:
            if error != "":
                failures.append(f"{label_of.get(ref_id, ref_id)}: {error}")
            continue
        if resolved is None:
            continue
        if not resolved.is_file():
            add_missing(f"PBXFileReference {label_of.get(ref_id, ref_id)}", resolved)

    for config in objects.values():
        if config.get("isa") != "XCBuildConfiguration":
            continue
        settings = config.get("buildSettings", {})
        for key in ("INFOPLIST_FILE", "SWIFT_OBJC_BRIDGING_HEADER", "CODE_SIGN_ENTITLEMENTS"):
            value = settings.get(key)
            if not value:
                continue
            resolved = _as_path(value)
            if not resolved.is_file():
                add_missing(f"build setting {key}", resolved)

        search_paths = settings.get("HEADER_SEARCH_PATHS", [])
        if isinstance(search_paths, str):
            search_paths = [search_paths]
        for value in search_paths:
            if value == "$(inherited)":
                continue
            resolved = _as_path(value)
            if not resolved.is_dir():
                add_missing(f"build setting HEADER_SEARCH_PATHS entry {value!r}", resolved)

    # Preserve order while removing duplicate messages from Debug/Release.
    return list(dict.fromkeys(failures))


def main(pbx_path: Path = PBX) -> int:
    if not os.path.isfile(pbx_path):
        print("Missing", pbx_path)
        return 1
    valid, message = validate_file(str(pbx_path))
    if not valid:
        print("[xcode] FAIL")
        print(" -", message)
        return 1
    text = Path(pbx_path).read_text(encoding="utf-8")
    doc = Parser(text).parse()
    failures: List[str] = []

    defined = set(re.findall(r"^\t\t([0-9A-F]{24}) ", text, re.M))
    refs = set(re.findall(r"\b([0-9A-F]{24})\b", text))
    if refs - defined:
        failures.append(f"undefined object refs: {sorted(refs - defined)[:5]}")
    if re.search(r"PLACEHOLDER_\w+|PROJ_(?:DEBUG|RELEASE)", text):
        failures.append("unresolved placeholders in pbxproj")
    if text.count("{") != text.count("}") or text.count("(") != text.count(")"):
        failures.append("unbalanced braces/parentheses")

    paths = re.findall(r'path = "([^"]+)";', text)
    failures.extend(check_resolved_paths(text, doc))

    required_settings = [
        "IPHONEOS_DEPLOYMENT_TARGET = 16.0;",
        "CODE_SIGN_STYLE = Automatic;",
        "CODE_SIGN_ENTITLEMENTS = \"ios/LittleBitLoader/LittleBitLoader.entitlements\";",
        "SWIFT_OBJC_BRIDGING_HEADER = \"$(SRCROOT)/ios/LittleBitLoader/LittleBitLoader-Bridging-Header.h\";",
        "CLANG_CXX_LANGUAGE_STANDARD = \"c++17\";",
        "\"ARCHS[sdk=iphoneos*]\" = arm64;",
        "-fno-fast-math",
        "-ffp-contract=off",
    ]
    for setting in required_settings:
        if setting not in text:
            failures.append(f"missing build setting: {setting}")
    if "-march=native" in text:
        failures.append("illegal -march=native in Xcode settings")
    if "DEVELOPMENT_TEAM" not in text:
        print("[signing] DEVELOPMENT_TEAM is not embedded; set it in Xcode or regenerate with env var")

    plist_text = PLIST.read_text(encoding="utf-8")
    for key in ["UIFileSharingEnabled", "LSSupportsOpeningDocumentsInPlace", "UILaunchScreen",
                "UIRequiredDeviceCapabilities"]:
        if f"<key>{key}</key>" not in plist_text:
            failures.append(f"Info.plist missing {key}")
    if "arm64" not in plist_text:
        failures.append("Info.plist does not require arm64")

    try:
        entitlements = plistlib.loads(ENTITLEMENTS.read_bytes())
    except (OSError, ValueError) as exc:
        failures.append(f"invalid app entitlements plist {ENTITLEMENTS}: {exc}")
    else:
        capability = "com.apple.developer.kernel.increased-memory-limit"
        if entitlements.get(capability) is not True:
            failures.append(f"app entitlements missing {capability} = true")

    app_sources = [
        "LittleBitEngineBridge.mm", "model.cpp", "LittleBitEngine.swift",
        "binary_packer.cpp", "binary_packer_neon.cpp", "scalar_kernels.cpp",
        "lm_head_scalar.cpp",
    ]
    for name in app_sources:
        if not any(p.endswith(name) for p in paths):
            failures.append(f"app target missing source {name}")
    if not any(p.endswith("EngineSmokeTests.swift") for p in paths):
        failures.append("test target missing EngineSmokeTests.swift")

    if failures:
        print("[xcode] FAIL")
        for failure in failures:
            print(" -", failure)
        return 1
    print("[xcode] PASS: structure, resolved paths, sources, deployment target, bridging header and entitlements")
    print("[xcode] NOTE: signing/build/test still require macOS, Xcode and a device")
    return 0


if __name__ == "__main__":
    selected = Path(sys.argv[1]) if len(sys.argv) > 1 else PBX
    raise SystemExit(main(selected))
