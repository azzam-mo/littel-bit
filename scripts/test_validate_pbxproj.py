#!/usr/bin/env python3
"""Fixture tests for scripts/validate_pbxproj.py.

This is intentionally standalone (not registered with CTest) so the existing
x86 CTest suite stays at 8 tests. Run it with:

    python3 scripts/test_validate_pbxproj.py
"""

from __future__ import annotations

import pathlib
import sys
import unittest

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
FIXTURES = SCRIPT_DIR / "pbxproj_fixtures"
sys.path.insert(0, str(SCRIPT_DIR))

from validate_pbxproj import validate_file  # noqa: E402


class ValidatePbxprojFixtures(unittest.TestCase):
    def test_valid_project_passes(self) -> None:
        ok, message = validate_file(str(FIXTURES / "valid.pbxproj"))
        self.assertTrue(ok, message)
        self.assertIn("OK (strict OpenStep plist and project structure)", message)

    def test_rejects_each_rule(self) -> None:
        cases = {
            "bad_header.pbxproj": "invalid pbxproj header",
            "bad_bare_key.pbxproj": "invalid character '[' in bare key 'ARCHS[sdk=iphoneos*]'",
            "bad_bare_value.pbxproj": "invalid character '+' in bare value 'c++17'",
            "bad_bare_at_sign.pbxproj": "invalid character '@' in bare value '@executable_path'",
            "duplicate_id.pbxproj": "duplicate object ID 'AAAAAAAAAAAAAAAAAAAAAAAA'",
            "missing_root_object.pbxproj": "missing required top-level key 'rootObject'",
            "dangling_reference.pbxproj": "dangling object reference 'FFFFFFFFFFFFFFFFFFFFFFFF'",
            "wrong_type.pbxproj": "objects must be a dictionary",
        }
        for name, expected in cases.items():
            with self.subTest(fixture=name):
                ok, message = validate_file(str(FIXTURES / name))
                self.assertFalse(ok, f"{name} unexpectedly passed: {message}")
                self.assertIn(expected, message)


if __name__ == "__main__":
    unittest.main(verbosity=2)
