#!/usr/bin/env python3
"""Standalone path-resolution tests for scripts/check_xcode_project.py.

Not registered with CTest so the existing x86 CTest count stays at 8.
"""
import contextlib
import io
import plistlib
import tempfile
import unittest
from pathlib import Path

from check_xcode_project import ENTITLEMENTS, PBX, ROOT, main

FIXTURE = Path(__file__).resolve().parent / "pbxproj_fixtures" / "doubled_path.pbxproj"


class XcodeProjectPaths(unittest.TestCase):
    def test_doubled_path_fixture_is_rejected(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            rc = main(FIXTURE)
        self.assertNotEqual(rc, 0)
        self.assertIn("ios/LittleBitLoader/ios/LittleBitLoader/LittleBitLoaderApp.swift", output.getvalue())

    def test_generated_project_paths_resolve(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            rc = main(PBX)
        self.assertEqual(rc, 0, output.getvalue())
        self.assertIn("[xcode] PASS: structure, resolved paths", output.getvalue())

    def test_app_entitlements_are_wired_and_valid(self):
        text = PBX.read_text(encoding="utf-8")
        self.assertIn(
            'CODE_SIGN_ENTITLEMENTS = "ios/LittleBitLoader/LittleBitLoader.entitlements";',
            text,
        )
        entitlements = plistlib.loads(ENTITLEMENTS.read_bytes())
        self.assertIs(entitlements["com.apple.developer.kernel.increased-memory-limit"], True)

    def test_missing_entitlements_path_is_rejected(self):
        original = PBX.read_text(encoding="utf-8")
        changed = original.replace(
            'CODE_SIGN_ENTITLEMENTS = "ios/LittleBitLoader/LittleBitLoader.entitlements";',
            'CODE_SIGN_ENTITLEMENTS = "ios/LittleBitLoader/Missing.entitlements";',
        )
        self.assertNotEqual(changed, original)
        with tempfile.TemporaryDirectory(prefix="littlebit-entitlement-check-") as temp_dir:
            candidate = Path(temp_dir) / "project.pbxproj"
            candidate.write_text(changed, encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                rc = main(candidate)
        self.assertNotEqual(rc, 0)
        self.assertIn("ios/LittleBitLoader/Missing.entitlements", output.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
