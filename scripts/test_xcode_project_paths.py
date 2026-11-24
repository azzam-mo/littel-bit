#!/usr/bin/env python3
"""Standalone path-resolution tests for scripts/check_xcode_project.py.

Not registered with CTest so the existing x86 CTest count stays at 8.
"""
import contextlib
import io
import unittest
from pathlib import Path

from check_xcode_project import PBX, main

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
