#!/usr/bin/env python3
"""Compare the C++ tokenizer with the official Hugging Face `tokenizers` library.

Usage:
  python3 scripts/validate_tokenizer.py <model_dir> [tokenizer_probe]

The reference is the model's own tokenizer.json, not an approximation.
"""
import json
import os
import subprocess
import sys

from tokenizers import Tokenizer


def chat_prompt(message):
    return f"<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    model_dir = sys.argv[1]
    probe = sys.argv[2] if len(sys.argv) > 2 else "build/tokenizer_probe"
    vocab_dir = sys.argv[3] if len(sys.argv) > 3 else model_dir
    tok_json = os.path.join(model_dir, "tokenizer.json")
    vocab = os.path.join(vocab_dir, "vocab.json")
    merges = os.path.join(vocab_dir, "merges.txt")
    if not all(os.path.isfile(p) for p in [tok_json, vocab, merges]):
        print("Missing tokenizer files in", model_dir)
        return 2

    ref = Tokenizer.from_file(tok_json)
    samples = [
        "hello",
        "Qwen3 runs locally on iPhone.",
        "مرحبا بالعالم",
        "مرحبا، كيف حالك؟ ١٢٣",
        "12345 3.14159 2026-11-23",
        "!@#$%^&*()[]{}<>?/\\|~`+-=",
        "Mixed مرحبا 123 — 🌟",
        "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n",
    ]
    samples += [chat_prompt(s) for s in ["hello", "مرحبا", "123"]]

    passed = True
    for text in samples:
        expected = ref.encode(text, add_special_tokens=False).ids
        out = subprocess.check_output(
            [probe, vocab, merges, text, tok_json], text=True
        ).strip()
        got = [int(x) for x in out.split()] if out else []
        ok = got == expected
        passed &= ok
        label = text if len(text) <= 48 else text[:45] + "..."
        if ok:
            print(f"[MATCH] {label!r} -> {len(got)} tokens")
        else:
            first = next((i for i, (a, b) in enumerate(zip(got, expected)) if a != b),
                         min(len(got), len(expected)))
            print(f"[MISMATCH] {label!r} len {len(got)} vs {len(expected)}, first diff @{first}")
            print(f"  got     {got[max(0, first-3):first+5]}")
            print(f"  expected {expected[max(0, first-3):first+5]}")
    print("RESULT:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
