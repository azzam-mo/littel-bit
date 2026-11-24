#!/usr/bin/env python3
"""Strict old-style/OpenStep plist validator for project.pbxproj files.

This parser is intentionally small and uses only the Python standard library.
It does not build anything and does not require Xcode or plutil.

Rules enforced here:

* the file starts with the exact Xcode header ``// !$*UTF8*$!\n`` (no BOM, no
  CRLF variant, no leading whitespace);
* dictionaries are ``{ key = value; ... }`` and require semicolons;
* arrays are ``( value, ... )`` and require commas (one trailing comma is
  accepted, matching Xcode's own generator);
* bare dictionary keys and bare values may contain only
  ``[A-Za-z0-9_$/:.-]``; everything else must be quoted;
* ``//`` and ``/* ... */`` comments are accepted;
* braces and parentheses must balance and the whole file must be one
  dictionary;
* ``archiveVersion``, ``classes``, ``objectVersion``, ``objects`` and
  ``rootObject`` are present with the expected types;
* object IDs are unique 24-character uppercase hex strings, ``rootObject``
  points to a PBXProject object, every ID-shaped value references a defined
  object, and known object fields have the expected dictionary/list/ID/scalar
  types.

Exit status is 0 on success and 1 on the first parse or structure error.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
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


EXPECTED_HEADER = b"// !$*UTF8*$!\n"
ID_RE = re.compile(r"^[0-9A-F]{24}$")

# CoreFoundation's old-style plist parser accepts this character set in bare
# tokens. Anything outside it must be quoted. This applies to dictionary keys
# as well as values: "ARCHS[sdk=iphoneos*]" is legal only when quoted.
SAFE_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "_$/:.-"
)

# Scan a superset of SAFE_CHARS so common mistakes such as "c++17", "libc++",
# "ARCHS[sdk=iphoneos*]" and "@executable_path" can be reported precisely as
# invalid bare tokens instead of as generic lexer failures.
LEXER_CHARS = SAFE_CHARS | frozenset("+[]=*@%!?&~^'#")

REQUIRED_TOP_LEVEL = (
    "archiveVersion",
    "classes",
    "objectVersion",
    "objects",
    "rootObject",
)

# Required fields and their expected types for the object kinds emitted by
# scripts/generate_xcode_project.py. "id" means a 24-character uppercase hex
# reference that must resolve to an object defined in the objects section.
OBJECT_FIELD_TYPES: Dict[str, Dict[str, str]] = {
    "PBXBuildFile": {"fileRef": "id"},
    "PBXContainerItemProxy": {
        "containerPortal": "id",
        "proxyType": "scalar",
        "remoteGlobalIDString": "id",
        "remoteInfo": "scalar",
    },
    "PBXFileReference": {},
    "PBXFrameworksBuildPhase": {
        "buildActionMask": "scalar",
        "files": "list",
        "runOnlyForDeploymentPostprocessing": "scalar",
    },
    "PBXGroup": {"children": "list", "sourceTree": "scalar"},
    "PBXNativeTarget": {
        "buildConfigurationList": "id",
        "buildPhases": "list",
        "buildRules": "list",
        "dependencies": "list",
        "name": "scalar",
        "productName": "scalar",
        "productReference": "id",
        "productType": "scalar",
    },
    "PBXProject": {
        "attributes": "dict",
        "buildConfigurationList": "id",
        "compatibilityVersion": "scalar",
        "developmentRegion": "scalar",
        "hasScannedForEncodings": "scalar",
        "knownRegions": "list",
        "mainGroup": "id",
        "productRefGroup": "id",
        "projectDirPath": "scalar",
        "projectRoot": "scalar",
        "targets": "list",
    },
    "PBXResourcesBuildPhase": {
        "buildActionMask": "scalar",
        "files": "list",
        "runOnlyForDeploymentPostprocessing": "scalar",
    },
    "PBXSourcesBuildPhase": {
        "buildActionMask": "scalar",
        "files": "list",
        "runOnlyForDeploymentPostprocessing": "scalar",
    },
    "PBXTargetDependency": {"target": "id", "targetProxy": "id"},
    "XCBuildConfiguration": {"buildSettings": "dict", "name": "scalar"},
    "XCConfigurationList": {
        "buildConfigurations": "list",
        "defaultConfigurationIsVisible": "scalar",
        "defaultConfigurationName": "scalar",
    },
}


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

        if char in LEXER_CHARS:
            value: List[str] = []
            while self.pos < len(self.text) and self.peek() in LEXER_CHARS:
                value.append(self.advance())
            return Token("WORD", "".join(value), line, column)

        raise self.error(f"unexpected character {char!r}")


@dataclass
class ParsedDocument:
    """Parse tree plus source locations needed for structural diagnostics."""

    root: Dict[str, Any]
    bare_tokens: List[Tuple[str, Token]] = field(default_factory=list)
    top_keys: Dict[str, Token] = field(default_factory=dict)
    object_key_tokens: Dict[str, Token] = field(default_factory=dict)


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
        self.bare_tokens: List[Tuple[str, Token]] = []
        self.top_keys: Dict[str, Token] = {}
        self.object_key_tokens: Dict[str, Token] = {}

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
        role = "key" if is_key else "value"
        for offset, char in enumerate(token.value):
            if char not in SAFE_CHARS:
                raise PbxprojParseError(
                    f"invalid character {char!r} in bare {role} {token.value!r}",
                    token.line,
                    token.column + offset,
                )

    def parse_value(self, *, depth: int, in_objects: bool = False) -> Any:
        token = self.current()
        if token is None:
            raise self.error("expected value, found end of file")

        if token.kind == "{":
            return self.parse_dictionary(depth=depth, in_objects=in_objects)
        if token.kind == "(":
            return self.parse_array(depth=depth)
        if token.kind == "STRING":
            self.index += 1
            return token.value
        if token.kind == "WORD":
            self.parse_bare(token, is_key=False)
            self.bare_tokens.append(("value", token))
            self.index += 1
            return token.value
        raise self.error(f"expected value, found {token.value!r}", token)

    def parse_dictionary(self, *, depth: int, in_objects: bool = False) -> Dict[str, Any]:
        self.take("{", "dictionary opening brace")
        result: Dict[str, Any] = {}
        while True:
            token = self.current()
            if token is None:
                raise self.error("unterminated dictionary; expected '}'")
            if token.kind == "}":
                self.index += 1
                return result

            if token.kind == "STRING":
                self.index += 1
                key = token.value
            elif token.kind == "WORD":
                self.parse_bare(token, is_key=True)
                self.bare_tokens.append(("key", token))
                self.index += 1
                key = token.value
            else:
                raise self.error(f"expected dictionary key, found {token.value!r}", token)

            if key in result:
                kind = "duplicate object ID" if in_objects and ID_RE.fullmatch(key) else "duplicate dictionary key"
                raise PbxprojParseError(f"{kind} {key!r}", token.line, token.column)

            if depth == 0:
                self.top_keys[key] = token
            if in_objects and ID_RE.fullmatch(key):
                self.object_key_tokens[key] = token

            self.take("=", "'=' after dictionary key")
            result[key] = self.parse_value(
                depth=depth + 1,
                in_objects=depth == 0 and key == "objects",
            )
            self.take(";", "';' after dictionary value")

    def parse_array(self, *, depth: int) -> List[Any]:
        self.take("(", "array opening parenthesis")
        result: List[Any] = []
        if self.kind() == ")":
            self.index += 1
            return result

        while True:
            result.append(self.parse_value(depth=depth + 1))
            if self.kind() == ",":
                self.index += 1
                # Xcode's own generator emits one trailing comma in arrays.
                if self.kind() == ")":
                    self.index += 1
                    return result
                continue
            if self.kind() == ")":
                self.index += 1
                return result
            token = self.current()
            if token is None:
                raise self.error("unterminated array; expected ')'")
            raise self.error("expected ',' or ')' after array value", token)

    def parse(self) -> ParsedDocument:
        root = self.parse_dictionary(depth=0)
        extra = self.current()
        if extra is not None:
            raise self.error(f"unexpected trailing token {extra.value!r}", extra)
        return ParsedDocument(
            root=root,
            bare_tokens=self.bare_tokens,
            top_keys=self.top_keys,
            object_key_tokens=self.object_key_tokens,
        )


def _is_scalar(value: Any) -> bool:
    # The parser represents both bare and quoted scalars as strings.
    return isinstance(value, str)


def _require_scalar(value: Any, label: str, token: Token | None) -> None:
    if not _is_scalar(value):
        line, column = (token.line, token.column) if token else (1, 1)
        raise PbxprojParseError(f"{label} must be a scalar", line, column)


def _require_dict(value: Any, label: str, token: Token | None) -> None:
    if not isinstance(value, dict):
        line, column = (token.line, token.column) if token else (1, 1)
        raise PbxprojParseError(f"{label} must be a dictionary", line, column)


def _require_list(value: Any, label: str, token: Token | None) -> None:
    if not isinstance(value, list):
        line, column = (token.line, token.column) if token else (1, 1)
        raise PbxprojParseError(f"{label} must be an array", line, column)


def validate_document(doc: ParsedDocument) -> None:
    root = doc.root
    fallback = Token("EOF", "", 1, 1)

    for key in REQUIRED_TOP_LEVEL:
        if key not in root:
            token = doc.top_keys.get(key, fallback)
            raise PbxprojParseError(f"missing required top-level key {key!r}", token.line, token.column)

    _require_scalar(root["archiveVersion"], "archiveVersion", doc.top_keys.get("archiveVersion"))
    _require_dict(root["classes"], "classes", doc.top_keys.get("classes"))
    _require_scalar(root["objectVersion"], "objectVersion", doc.top_keys.get("objectVersion"))
    _require_dict(root["objects"], "objects", doc.top_keys.get("objects"))

    objects: Dict[str, Any] = root["objects"]
    if not objects:
        token = doc.top_keys.get("objects", fallback)
        raise PbxprojParseError("objects section contains no object IDs", token.line, token.column)

    defined_ids = set(objects)
    if len(defined_ids) != len(objects):
        token = fallback
        raise PbxprojParseError("duplicate object ID in objects section", token.line, token.column)

    root_id = root["rootObject"]
    _require_scalar(root_id, "rootObject", doc.top_keys.get("rootObject"))
    if not ID_RE.fullmatch(root_id):
        token = doc.top_keys.get("rootObject", fallback)
        raise PbxprojParseError(f"rootObject is not a valid object ID: {root_id!r}", token.line, token.column)

    root_token = doc.top_keys.get("rootObject", fallback)
    if root_id not in defined_ids:
        raise PbxprojParseError(f"rootObject references missing object ID {root_id!r}", root_token.line, root_token.column)

    root_object_token = doc.object_key_tokens.get(root_id, fallback)
    _require_dict(objects[root_id], f"root object {root_id}", root_object_token)
    if objects[root_id].get("isa") != "PBXProject":
        raise PbxprojParseError(
            f"rootObject {root_id!r} must be a PBXProject object",
            root_object_token.line,
            root_object_token.column,
        )

    # Any 24-character uppercase hex bare value is an object reference. This
    # catches references hidden in arrays (children, files, targets, etc.).
    for role, token in doc.bare_tokens:
        if role == "value" and ID_RE.fullmatch(token.value) and token.value not in defined_ids:
            raise PbxprojParseError(
                f"dangling object reference {token.value!r}",
                token.line,
                token.column,
            )

    for object_id, obj in objects.items():
        token = doc.object_key_tokens.get(object_id, fallback)
        _require_dict(obj, f"object {object_id}", token)
        isa = obj.get("isa")
        if not _is_scalar(isa):
            raise PbxprojParseError(f"object {object_id!r} has a non-scalar isa", token.line, token.column)
        if isa not in OBJECT_FIELD_TYPES:
            raise PbxprojParseError(f"object {object_id!r} has unsupported isa {isa!r}", token.line, token.column)

        for field_name, expected in OBJECT_FIELD_TYPES[isa].items():
            if field_name not in obj:
                raise PbxprojParseError(
                    f"object {object_id!r} ({isa}) is missing field {field_name!r}",
                    token.line,
                    token.column,
                )
            value = obj[field_name]
            label = f"object {object_id!r} ({isa}) field {field_name!r}"
            if expected == "dict":
                _require_dict(value, label, token)
            elif expected == "list":
                _require_list(value, label, token)
            elif expected == "scalar":
                _require_scalar(value, label, token)
            elif expected == "id":
                if not _is_scalar(value) or not ID_RE.fullmatch(value):
                    raise PbxprojParseError(
                        f"{label} must be a valid object ID",
                        token.line,
                        token.column,
                    )
                if value not in defined_ids:
                    raise PbxprojParseError(
                        f"{label} references missing object ID {value!r}",
                        token.line,
                        token.column,
                    )


def validate_file(path: str) -> Tuple[bool, str]:
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError as exc:
        return False, f"cannot read {path}: {exc}"

    if not raw.startswith(EXPECTED_HEADER):
        return False, f"{path}:1:1: invalid pbxproj header; expected exact first line {EXPECTED_HEADER!r}"

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return False, f"{path}:1:1: pbxproj is not valid UTF-8: {exc}"

    try:
        document = Parser(text).parse()
        validate_document(document)
    except PbxprojParseError as exc:
        return False, f"{path}:{exc.line}:{exc.column}: {exc.message}"
    return True, f"{path}: OK (strict OpenStep plist and project structure)"


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="project.pbxproj file to validate")
    args = parser.parse_args(argv)

    ok, message = validate_file(args.path)
    print(("OK: " if ok else "ERROR: ") + message)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
