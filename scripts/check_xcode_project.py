#!/usr/bin/env python3
"""Structurally validate the generated Xcode project on any platform.

This does NOT build or sign the app. It checks the generated pbxproj, source
references and Info.plist so macOS-specific gaps are explicit.

Usage: python3 scripts/check_xcode_project.py
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PBX = os.path.join(ROOT, "LittleBitLoader.xcodeproj", "project.pbxproj")
PLIST = os.path.join(ROOT, "ios", "LittleBitLoader", "Info.plist")


def main():
    if not os.path.isfile(PBX):
        print("Missing", PBX)
        return 1
    text = open(PBX, encoding="utf-8").read()
    failures = []

    defined = set(re.findall(r"^\t\t([0-9A-F]{24}) ", text, re.M))
    refs = set(re.findall(r"\b([0-9A-F]{24})\b", text))
    if refs - defined:
        failures.append(f"undefined object refs: {sorted(refs - defined)[:5]}")
    if re.search(r"PLACEHOLDER_\w+|PROJ_(?:DEBUG|RELEASE)", text):
        failures.append("unresolved placeholders in pbxproj")
    if text.count("{") != text.count("}") or text.count("(") != text.count(")"):
        failures.append("unbalanced braces/parentheses")

    paths = re.findall(r'path = "([^"]+)";', text)
    missing = [p for p in paths if not os.path.exists(os.path.join(ROOT, p)) and
               not p.endswith((".app", ".xctest")) and
               not p.startswith("System/Library/Frameworks")]
    if missing:
        failures.append(f"missing source files: {missing[:5]}")

    required_settings = [
        "IPHONEOS_DEPLOYMENT_TARGET = 16.0;",
        "CODE_SIGN_STYLE = Automatic;",
        "SWIFT_OBJC_BRIDGING_HEADER = \"$(SRCROOT)/ios/LittleBitLoader/LittleBitLoader-Bridging-Header.h\";",
        "CLANG_CXX_LANGUAGE_STANDARD = c++17;",
        "ARCHS[sdk=iphoneos*] = arm64;",
    ]
    for setting in required_settings:
        if setting not in text:
            failures.append(f"missing build setting: {setting}")
    if "-march=native" in text:
        failures.append("illegal -march=native in Xcode settings")
    if "DEVELOPMENT_TEAM" not in text:
        print("[signing] DEVELOPMENT_TEAM is not embedded; set it in Xcode or regenerate with env var")

    plist_text = open(PLIST, encoding="utf-8").read()
    for key in ["UIFileSharingEnabled", "LSSupportsOpeningDocumentsInPlace", "UILaunchScreen",
                "UIRequiredDeviceCapabilities"]:
        if f"<key>{key}</key>" not in plist_text:
            failures.append(f"Info.plist missing {key}")
    if "arm64" not in plist_text:
        failures.append("Info.plist does not require arm64")

    app_sources = ["LittleBitEngineBridge.mm", "model.cpp", "LittleBitEngine.swift"]
    for name in app_sources:
        if not any(p.endswith(name) for p in paths):
            failures.append(f"app target missing source {name}")
    if not any(p.endswith("EngineSmokeTests.swift") for p in paths):
        failures.append("test target missing EngineSmokeTests.swift")

    if failures:
        print("[xcode] FAIL")
        for f in failures:
            print(" -", f)
        return 1
    print("[xcode] PASS: structure, sources, deployment target and bridging header")
    print("[xcode] NOTE: signing/build/test still require macOS, Xcode and a device")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
