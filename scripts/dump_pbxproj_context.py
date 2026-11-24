#!/usr/bin/env python3
"""Print pbxproj context for a failed `xcodebuild -list` run.

Usage:

    python3 scripts/dump_pbxproj_context.py XCODEBUILD_LOG PROJECT_PBXPROJ

This helper is diagnostic only: it never builds anything and always exits 0
unless an argument is missing.
"""

from __future__ import annotations

import pathlib
import re
import sys
from typing import List


def line_numbers_from_log(log_text: str, max_line: int) -> List[int]:
    """Collect 1-based line numbers mentioned as `line N` or `:N`."""
    matches = re.findall(r"(?:[Ll]ine\s*|:)\s*(\d+)", log_text)
    numbers = {int(value) for value in matches}
    return sorted(number for number in numbers if 1 <= number <= max_line)


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print("usage: dump_pbxproj_context.py XCODEBUILD_LOG PROJECT_PBXPROJ", file=sys.stderr)
        return 2

    log_path = pathlib.Path(argv[0])
    pbx_path = pathlib.Path(argv[1])
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    lines = pbx_path.read_text(encoding="utf-8").splitlines()

    print("### first 60 lines of project.pbxproj")
    for number, line in enumerate(lines[:60], start=1):
        print(f"{number:5}: {line}")

    numbers = line_numbers_from_log(log_text, len(lines))
    if not numbers:
        print("### no pbxproj line numbers found in xcodebuild output")
        return 0

    for number in numbers:
        print(f"### context around line {number}")
        start = max(1, number - 5)
        end = min(len(lines), number + 5)
        for index in range(start, end + 1):
            print(f"{index:5}: {lines[index - 1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
