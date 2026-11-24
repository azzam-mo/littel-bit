#!/usr/bin/env python3
"""Strict old-style/OpenStep plist validator for project.pbxproj files.

This parser is intentionally small and uses only the Python standard library.
It does not build anything and does not require Xcode or plutil.

Rules enforced here:

* dictionaries are ``{ key = value; ... }`` and require semicolons;
* arrays are ``( value, ... )`` and require commas;
* strings may be quoted or bare;
* ``//`` and ``/* ... */`` comments are accepted;
* braces and parentheses must balance;
* the whole file must be consumed as one dictionary.

Exit status is 0 on success and 1 on the first parse error.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


class PbxprojParseError(Exception):
    """A strict parse failure with an exact source location."""

    def __init__(self, message: str, line: int, column: int) -> None:
        super().__init__(message)
        self.message = message
        self.line = line
        self.column = column

    def __str__(self) -> str:
        return f"line {self.line}, column {self.column}: {self.message}"


@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    line: int
    column: int


# Bare dictionary keys in Apple project files commonly contain conditional
# suffixes such as ARCHS[sdk=iphoneos*]. Values are deliberately stricter:
# plus signs are not valid in old-style bare values, which is the exact class
# of bug this validator is meant to catch ("c++17", "libc++").
KEY_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "_$/:.-[]=*"
)
VALUE_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "_$/:.-"
)


class Lexer:
    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0
        self.line = 1
        self.column = 1

    def error(self, message: str) -> PbxprojParseError:
        return PbxprojParseError(message, self.line, self.column)

    def peek(self, offset: int = 0) -> str:
        index = self.pos + offset
        if index >= len(self.text):
            return ""
        return self.text[index]

    def advance(self) -> str:
        char = self.text[self.pos]
        self.pos += 1
        if char == "\n":
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return char

    def skip_ignored(self) -> None:
        while self.pos < len(self.text):
            char = self.peek()
            if char.isspace():
                self.advance()
            elif char == "/" and self.peek(1) == "/":
                while self.pos < len(self.text) and self.peek() != "\n":
                    self.advance()
            elif char == "/" and self.peek(1) == "*":
                self.advance()
                self.advance()
                while self.pos < len(self.text):
                    if self.peek() == "*" and self.peek(1) == "/":
                        self.advance()
                        self.advance()
                        break
                    self.advance()
                else:
                    raise self.error("unterminated block comment")
            else:
                return

    def read_quoted(self) -> Token:
        line, column = self.line, self.column
        self.advance()  # opening quote
        value: List[str] = []
        while self.pos < len(self.text):
            char = self.peek()
            if char == '"':
                self.advance()
                return Token("STRING", "".join(value), line, column)
            if char == "\\":
                self.advance()
                if self.pos >= len(self.text):
                    raise self.error("unterminated escape sequence in quoted string")
                escaped = self.advance()
                mapping = {
                    "n": "\n",
                    "r": "\r",
                    "t": "\t",
                    "b": "\b",
                    "f": "\f",
                    '"': '"',
                    "'": "'",
                    "\\": "\\",
                }
                value.append(mapping.get(escaped, escaped))
                continue
            if char == "\n":
                raise self.error("newline in quoted string")
            value.append(self.advance())
        raise self.error("unterminated quoted string")

    def next_token(self) -> Token | None:
        self.skip_ignored()
        if self.pos >= len(self.text):
            return None

        line, column = self.line, self.column
        char = self.peek()

        if char in "{}(),;=":
            self.advance()
            return Token(char, char, line, column)

        if char == '"':
            return self.read_quoted()

        # Bare tokens are scanned broadly here. The parser decides whether the
        # characters are legal for a key or a value.
        allowed = KEY_CHARS | VALUE_CHARS | frozenset("+")
        if char in allowed:
            value: List[str] = []
            while self.pos < len(self.text) and self.peek() in allowed:
                value.append(self.advance())
            return Token("WORD", "".join(value), line, column)

        raise self.error(f"unexpected character {char!r}")


class Parser:
    def __init__(self, text: str) -> None:
        self.lexer = Lexer(text)
        self.tokens: List[Token] = []
        while True:
            token = self.lexer.next_token()
            if token is None:
                break
            self.tokens.append(token)
        self.index = 0

    def current(self) -> Token | None:
        if self.index >= len(self.tokens):
            return None
        return self.tokens[self.index]

    def kind(self) -> str:
        token = self.current()
        return "" if token is None else token.kind

    def error(self, message: str, token: Token | None = None) -> PbxprojParseError:
        if token is None:
            token = self.current()
        if token is None:
            previous = self.tokens[-1] if self.tokens else Token("EOF", "", 1, 1)
            return PbxprojParseError(message, previous.line, previous.column + len(previous.value))
        return PbxprojParseError(message, token.line, token.column)

    def take(self, kind: str, expected: str) -> Token:
        token = self.current()
        if token is None:
            raise self.error(f"expected {expected}, found end of file")
        if token.kind != kind:
            raise self.error(f"expected {expected}, found {token.value!r}", token)
        self.index += 1
        return token

    def parse_bare(self, token: Token, *, is_key: bool) -> None:
        allowed = KEY_CHARS if is_key else VALUE_CHARS
        role = "key" if is_key else "value"
        for offset, char in enumerate(token.value):
            if char not in allowed:
                raise PbxprojParseError(
                    f"invalid character {char!r} in bare {role} {token.value!r}",
                    token.line,
                    token.column + offset,
                )

    def parse_value(self) -> None:
        token = self.current()
        if token is None:
            raise self.error("expected value, found end of file")

        if token.kind == "{":
            self.parse_dictionary()
            return
        if token.kind == "(":
            self.parse_array()
            return
        if token.kind == "STRING":
            self.index += 1
            return
        if token.kind == "WORD":
            self.parse_bare(token, is_key=False)
            self.index += 1
            return
        raise self.error(f"expected value, found {token.value!r}", token)

    def parse_dictionary(self) -> None:
        self.take("{", "dictionary opening brace")
        while True:
            token = self.current()
            if token is None:
                raise self.error("unterminated dictionary; expected '}'")
            if token.kind == "}":
                self.index += 1
                return

            if token.kind == "STRING":
                self.index += 1
            elif token.kind == "WORD":
                self.parse_bare(token, is_key=True)
                self.index += 1
            else:
                raise self.error(f"expected dictionary key, found {token.value!r}", token)

            self.take("=", "'=' after dictionary key")
            self.parse_value()
            self.take(";", "';' after dictionary value")

    def parse_array(self) -> None:
        self.take("(", "array opening parenthesis")
        if self.kind() == ")":
            self.index += 1
            return

        while True:
            self.parse_value()
            if self.kind() == ",":
                self.index += 1
                # Apple's generator emits trailing commas in arrays; Xcode
                # accepts them, so allow a comma immediately before ')'.
                if self.kind() == ")":
                    self.index += 1
                    return
                continue
            if self.kind() == ")":
                self.index += 1
                return
            token = self.current()
            if token is None:
                raise self.error("unterminated array; expected ')'")
            raise self.error("expected ',' or ')' after array value", token)

    def parse(self) -> Dict[str, Any]:
        self.parse_dictionary()
        extra = self.current()
        if extra is not None:
            raise self.error(f"unexpected trailing token {extra.value!r}", extra)
        return {"status": "ok"}


def validate_file(path: str) -> Tuple[bool, str]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as exc:
        return False, f"cannot read {path}: {exc}"

    try:
        Parser(text).parse()
    except PbxprojParseError as exc:
        return False, f"{path}:{exc.line}:{exc.column}: {exc.message}"
    return True, f"{path}: OK (strict OpenStep plist)"


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="project.pbxproj file to validate")
    args = parser.parse_args(argv)

    ok, message = validate_file(args.path)
    print(("OK: " if ok else "ERROR: ") + message)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
