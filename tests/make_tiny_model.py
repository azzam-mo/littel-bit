#!/usr/bin/env python3
"""Generate a deterministic tiny Qwen3-shaped model for inference smoke tests.

This is NOT a real model: it only exists to exercise safetensors mmap loading,
LittleBitLinear decoding, GQA/KV-cache, streaming callbacks and early stopping
on machines without model weights or macOS/iOS.
"""
import json
import math
import os
import struct
import sys


def f32_bits(value: float) -> bytes:
    return struct.pack("<I", struct.unpack("<I", struct.pack("<f", value))[0])


def to_bf16(value: float) -> bytes:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return struct.pack("<H", bits >> 16)


def i64_payload(*values: int) -> bytes:
    return struct.pack("<%dq" % len(values), *values)


def byte_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    mapping = {}
    for b, code in zip(bs, cs):
        if code <= 0x7F:
            s = chr(code)
        elif code <= 0x7FF:
            s = chr(0xC0 | (code >> 6)) + chr(0x80 | (code & 0x3F))
        else:
            s = chr(0xE0 | (code >> 12)) + chr(0x80 | ((code >> 6) & 0x3F)) + chr(0x80 | (code & 0x3F))
        mapping[b] = s
    return mapping


def main() -> int:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "tiny-model"
    os.makedirs(out_dir, exist_ok=True)

    hidden = 32
    layers = 1
    q_heads = 4
    kv_heads = 2
    head_dim = 8
    inter = 64
    vocab = 256
    seq = 32
    split = 16
    scale = 0.01

    config = {
        "model_type": "qwen3",
        "vocab_size": vocab,
        "hidden_size": hidden,
        "num_hidden_layers": layers,
        "num_attention_heads": q_heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "intermediate_size": inter,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": seq,
        "eff_bit": 0.55,
        "split_dim": split,
        "residual": False,
        "kv_factor": 1.0,
        "min_split_dim": 8,
        "quant_mod": "LittleBitLinear",
        "quant_func": "SmoothSign",
        "use_itq": False,
        "itq_n_iter": 0,
    }
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(config, fh)

    littlebit_config = {
        "quant_func": "SmoothSign",
        "eff_bit": 0.55,
        "split_dim": split,
        "residual": False,
        "kv_factor": 1.0,
        "min_split_dim": 8,
        "quant_mod": "LittleBitLinear",
        "use_itq": False,
        "itq_n_iter": 0,
    }
    with open(os.path.join(out_dir, "littlebit_config.json"), "w", encoding="utf-8") as fh:
        json.dump(littlebit_config, fh)

    # Single-character byte-level vocabulary, enough for a smoke test.
    vocab_map = {ch: idx for idx, ch in enumerate(byte_to_unicode().values())}
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as fh:
        json.dump(vocab_map, fh, ensure_ascii=False)
    with open(os.path.join(out_dir, "merges.txt"), "w", encoding="utf-8") as fh:
        fh.write("# version: 0.2\n")

    tensors = {}

    def add(name, dtype, shape, payload):
        tensors[name] = {"dtype": dtype, "shape": shape, "payload": payload}

    # Embeddings / head / norms.
    add("model.embed_tokens.weight", "BF16", [vocab, hidden],
        b"".join(to_bf16(0.01 * math.sin(i * 0.07)) for i in range(vocab * hidden)))
    add("lm_head.weight", "BF16", [vocab, hidden],
        b"".join(to_bf16(0.01 * math.cos(i * 0.05)) for i in range(vocab * hidden)))
    for name, size in [
        ("model.norm.weight", hidden),
        ("model.layers.0.input_layernorm.weight", hidden),
        ("model.layers.0.post_attention_layernorm.weight", hidden),
        ("model.layers.0.self_attn.q_norm.weight", head_dim),
        ("model.layers.0.self_attn.k_norm.weight", head_dim),
    ]:
        add(name, "F32", [size], b"".join(f32_bits(1.0) for _ in range(size)))

    linears = [
        ("model.layers.0.self_attn.q_proj", hidden, hidden),
        ("model.layers.0.self_attn.k_proj", hidden, kv_heads * head_dim),
        ("model.layers.0.self_attn.v_proj", hidden, kv_heads * head_dim),
        ("model.layers.0.self_attn.o_proj", hidden, hidden),
        ("model.layers.0.mlp.gate_proj", hidden, inter),
        ("model.layers.0.mlp.up_proj", hidden, inter),
        ("model.layers.0.mlp.down_proj", inter, hidden),
    ]
    for prefix, in_feat, out_feat in linears:
        u_words = (split + 31) // 32
        v_words = (in_feat + 31) // 32
        add(prefix + ".U_shape", "I64", [2], i64_payload(out_feat, split))
        add(prefix + ".V_shape", "I64", [2], i64_payload(split, in_feat))
        add(prefix + "._split_dim_final", "I64", [], i64_payload(split))
        add(prefix + ".U_packed", "I32", [out_feat, u_words],
            struct.pack("<%di" % (out_feat * u_words), *([0x55555555] * (out_feat * u_words))))
        add(prefix + ".V_packed", "I32", [split, v_words],
            struct.pack("<%di" % (split * v_words), *([0x55555555] * (split * v_words))))
        for suffix, size in [("u1", out_feat), ("u2", split), ("v1", split), ("v2", in_feat)]:
            add(prefix + "." + suffix, "F32", [size], b"".join(f32_bits(scale) for _ in range(size)))

    # Serialise safetensors: 8-byte header length + JSON header + aligned data.
    header = {}
    data = bytearray()
    for name, meta in tensors.items():
        begin = len(data)
        data.extend(meta["payload"])
        while len(data) % 4:
            data.append(0)
        header[name] = {
            "dtype": meta["dtype"],
            "shape": meta["shape"],
            "data_offsets": [begin, len(data)],
        }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * ((-len(header_bytes)) % 8)
    with open(os.path.join(out_dir, "model.safetensors"), "wb") as fh:
        fh.write(struct.pack("<Q", len(header_bytes)))
        fh.write(header_bytes)
        fh.write(data)

    print("Wrote tiny synthetic model to", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
