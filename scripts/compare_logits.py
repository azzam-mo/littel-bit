#!/usr/bin/env python3
"""Compare C++ engine logits with an independent NumPy reference on REAL weights.

The reference follows Hugging Face transformers v4.51.3 modeling_qwen3.py and
SamsungLabs LittleBit-2 quantization/modules/littlebit.py. It uses the model's
actual Safetensors data; no synthetic weights are used.

Usage:
  python3 scripts/compare_logits.py <model_dir> <logits_probe> <tokens_csv> [num_layers]
"""
import json
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

SHIFTS = np.arange(32, dtype=np.int32)


class SafeTensors:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            self.header = json.loads(fh.read(n))
        self.header.pop("__metadata__", None)
        self.base = 8 + n
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    def _bytes(self, name, start=None, end=None):
        a, b = self.header[name]["data_offsets"]
        if start is not None:
            a += start
        if end is not None:
            b = a + (end - start) if start is not None else end
        return self.mm[self.base + a:self.base + b].tobytes()

    def raw(self, name, dtype):
        return np.frombuffer(self._bytes(name), dtype=dtype)

    def f32(self, name):
        return (self.raw(name, np.uint16).astype(np.uint32) << 16).view(np.float32)

    def f32_range(self, name, row_start, row_end, row_width=1):
        a, _ = self.header[name]["data_offsets"]
        lo = a + row_start * row_width * 2
        hi = a + row_end * row_width * 2
        return (np.frombuffer(self.mm[self.base + lo:self.base + hi].tobytes(), dtype=np.uint16)
                .astype(np.uint32) << 16).view(np.float32)

    def scalar_i64(self, name):
        return int(self.raw(name, np.int64)[0])

    def unpack(self, packed_name, shape_name):
        words = self.raw(packed_name, np.int32)
        rows, cols = [int(x) for x in self.raw(shape_name, np.int64)]
        wpr = words.size // rows
        words = words.reshape(rows, wpr)
        bits = ((words[:, :, None] >> SHIFTS) & 1).astype(np.int8)
        return (1 - 2 * bits).reshape(rows, wpr * 32)[:, :cols]


class LittleBitLinear:
    def __init__(self, st, prefix):
        self.st = st
        self.prefix = prefix
        self.v2 = st.f32(prefix + ".v2").reshape(-1)
        self.v1 = st.f32(prefix + ".v1").reshape(-1)
        self.u2 = st.f32(prefix + ".u2").reshape(-1)
        self.u1 = st.f32(prefix + ".u1").reshape(-1)
        self.residual = st.has(prefix + ".U_R_packed") if hasattr(st, "has") else (prefix + ".U_R_packed" in st.header)

    def _path(self, suffix, x):
        if suffix:
            # Safetensors stores residual scales as u1_R etc., not u1.R.
            scale = lambda name: self.st.f32(self.prefix + "." + name + suffix).reshape(-1)
            V = self.st.unpack(self.prefix + ".V" + suffix + "_packed",
                               self.prefix + ".V" + suffix + "_shape")
            U = self.st.unpack(self.prefix + ".U" + suffix + "_packed",
                               self.prefix + ".U" + suffix + "_shape")
        else:
            scale = lambda name: self.st.f32(self.prefix + "." + name).reshape(-1)
            V = self.st.unpack(self.prefix + ".V_packed", self.prefix + ".V_shape")
            U = self.st.unpack(self.prefix + ".U_packed", self.prefix + ".U_shape")
        v2, v1, u2, u1 = (scale("v2"), scale("v1"), scale("u2"), scale("u1"))
        h = (x * v2) @ V.T
        h *= v1 * u2
        h = h @ U.T
        return h * u1

    def forward(self, x):
        y = self._path("", x)
        if self.residual:
            y = y + self._path("_R", x)
        return y


def rmsnorm(x, w, eps):
    x32 = x.astype(np.float32)
    var = np.mean(x32 * x32, dtype=np.float32)
    return (w * (x32 / np.sqrt(var + eps))).astype(np.float32)


def rope_table(cfg, seq_len):
    half = cfg["head_dim"] // 2
    inv = 1.0 / np.power(np.float32(cfg["rope_theta"]),
                         np.arange(half, dtype=np.float32) * 2 / cfg["head_dim"])
    pos = np.arange(seq_len, dtype=np.float32)[:, None]
    freqs = pos * inv[None, :]
    cos = np.cos(freqs).astype(np.float32)
    sin = np.sin(freqs).astype(np.float32)
    return cos, sin


def rotate_half(x):
    half = x.shape[-1] // 2
    return np.concatenate((-x[..., half:], x[..., :half]), axis=-1)


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e, dtype=np.float32)


def reference_logits(st, cfg, tokens, n_layers):
    hidden = cfg["hidden_size"]
    kv_heads = cfg["num_key_value_heads"]
    q_heads = cfg["num_attention_heads"]
    head_dim = cfg["head_dim"]
    group = q_heads // kv_heads
    vocab = cfg["vocab_size"]
    eps = np.float32(cfg["rms_norm_eps"])
    cos, sin = rope_table(cfg, len(tokens))

    layers = []
    for l in range(n_layers):
        p = f"model.layers.{l}"
        layers.append({
            "in_norm": st.f32(p + ".input_layernorm.weight"),
            "post_norm": st.f32(p + ".post_attention_layernorm.weight"),
            "q_norm": st.f32(p + ".self_attn.q_norm.weight"),
            "k_norm": st.f32(p + ".self_attn.k_norm.weight"),
            "q": LittleBitLinear(st, p + ".self_attn.q_proj"),
            "k": LittleBitLinear(st, p + ".self_attn.k_proj"),
            "v": LittleBitLinear(st, p + ".self_attn.v_proj"),
            "o": LittleBitLinear(st, p + ".self_attn.o_proj"),
            "gate": LittleBitLinear(st, p + ".mlp.gate_proj"),
            "up": LittleBitLinear(st, p + ".mlp.up_proj"),
            "down": LittleBitLinear(st, p + ".mlp.down_proj"),
        })

    k_cache = np.zeros((n_layers, len(tokens), kv_heads, head_dim), dtype=np.float32)
    v_cache = np.zeros_like(k_cache)
    result = np.zeros((len(tokens), vocab), dtype=np.float32)

    for pos, token in enumerate(tokens):
        a, b = st.header["model.embed_tokens.weight"]["data_offsets"]
        row = st.f32_range("model.embed_tokens.weight",
                           token, token + 1, hidden)
        h = row.astype(np.float32)

        for l, lw in enumerate(layers):
            x = rmsnorm(h, lw["in_norm"], eps)
            q = lw["q"].forward(x).reshape(q_heads, head_dim)
            k = lw["k"].forward(x).reshape(kv_heads, head_dim)
            v = lw["v"].forward(x).reshape(kv_heads, head_dim)
            q = np.stack([rmsnorm(head, lw["q_norm"], eps) for head in q])
            k = np.stack([rmsnorm(head, lw["k_norm"], eps) for head in k])
            c, s = cos[pos], sin[pos]
            c_full = np.concatenate((c, c))
            s_full = np.concatenate((s, s))
            q = q * c_full + rotate_half(q) * s_full
            k = k * c_full + rotate_half(k) * s_full
            k_cache[l, pos] = k
            v_cache[l, pos] = v

            k_ctx = np.repeat(k_cache[l, :pos + 1], group, axis=1)
            v_ctx = np.repeat(v_cache[l, :pos + 1], group, axis=1)
            scores = np.einsum("qd,cqd->qc", q, k_ctx) / np.float32(head_dim ** 0.5)
            weights = np.stack([softmax(row) for row in scores])
            attn = np.einsum("qc,cqd->qd", weights, v_ctx).reshape(hidden)
            h = h + lw["o"].forward(attn)

            x2 = rmsnorm(h, lw["post_norm"], eps)
            gate = lw["gate"].forward(x2)
            up = lw["up"].forward(x2)
            silu = gate / (1.0 + np.exp(-gate))
            h = h + lw["down"].forward(silu * up)

        h = rmsnorm(h, st.f32("model.norm.weight"), eps)
        for start in range(0, vocab, 4096):
            end = min(start + 4096, vocab)
            rows = st.f32_range("lm_head.weight", start, end, hidden).reshape(end - start, hidden)
            result[pos, start:end] = rows @ h
        print(f"  reference: completed token {pos + 1}/{len(tokens)}", file=sys.stderr)
    return result


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    model_dir, probe, tokens_csv = sys.argv[1:4]
    n_layers = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    tokens = [int(x) for x in tokens_csv.split(",") if x]
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    if n_layers == 0:
        n_layers = cfg["num_hidden_layers"]
    st = SafeTensors(os.path.join(model_dir, "model.safetensors"))

    with tempfile.TemporaryDirectory() as td:
        binary = os.path.join(td, "logits.bin")
        out = subprocess.check_output(
            [probe, model_dir, ",".join(map(str, tokens)), binary, str(n_layers)], text=True
        )
        got = np.fromfile(binary, dtype=np.float32).reshape(len(tokens), cfg["vocab_size"])

    ref = reference_logits(st, cfg, tokens, n_layers)
    diff = np.abs(got - ref)
    scale = max(1e-8, float(np.max(np.abs(ref))))
    max_rel = float(np.max(diff)) / scale
    mean_rel = float(np.mean(diff)) / scale
    top_got = np.argmax(got[-1])
    top_ref = np.argmax(ref[-1])
    top5_got = set(np.argsort(got[-1])[-5:].tolist())
    top5_ref = set(np.argsort(ref[-1])[-5:].tolist())
    ok = max_rel < 5e-3 and top_got == top_ref
    print(f"[logits] layers={n_layers} tokens={len(tokens)} "
          f"max_abs={np.max(diff):.6g} max_rel={max_rel:.6g} mean_rel={mean_rel:.6g}")
    print(f"[top1] got={top_got} ref={top_ref} {'MATCH' if top_got == top_ref else 'MISMATCH'}")
    print(f"[top5] overlap={len(top5_got & top5_ref)}/5")
    print(f"[probe] {out.strip()}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
