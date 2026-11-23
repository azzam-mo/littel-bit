#!/usr/bin/env python3
"""Validate a REAL littlebit-qwen3 model against the C++ engine on Linux.

Checks (no fake weights):
  1. config.json vs ModelConfig defaults.
  2. Every tensor name/shape the loader needs actually exists.
  3. One LittleBitLinear output from the C++ engine vs an independent NumPy
     implementation of the official SamsungLabs LittleBit-2 math.

Usage:
  python3 scripts/validate_real_model.py <model_dir> [probe_binary]
"""
import json
import math
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

DT_ITEMSIZE = {"BF16": 2, "F16": 2, "F32": 4, "I32": 4, "I64": 8, "U8": 1}


class SafeTensors:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            self.header = json.loads(fh.read(n))
        self.header.pop("__metadata__", None)
        self.base = 8 + n
        with open(path, "rb") as fh:
            self.mm = np.memmap(fh, dtype=np.uint8, mode="r")

    def info(self, name):
        return self.header[name]

    def has(self, name):
        return name in self.header

    def raw(self, name):
        info = self.header[name]
        a, b = info["data_offsets"]
        dt = info["dtype"]
        return np.frombuffer(self.mm[self.base + a:self.base + b].tobytes(),
                             dtype=np.int32 if dt == "I32" else
                             np.int64 if dt == "I64" else np.uint16)

    def f32(self, name):
        u16 = self.raw(name).astype(np.uint32) << 16
        return u16.view(np.float32)

    def unpack(self, packed_name, shape_name):
        words = self.raw(packed_name).astype(np.int32)
        rows, cols = [int(x) for x in self.raw(shape_name)]
        wpr = words.size // rows
        words = words.reshape(rows, wpr)
        bits = ((words[:, :, None] >> np.arange(32, dtype=np.int32)) & 1).astype(np.int8)
        return (1 - 2 * bits).reshape(rows, wpr * 32)[:, :cols]


def check_config(model_dir):
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    expected = {
        "model_type": "qwen3", "vocab_size": 151936, "hidden_size": 4096,
        "num_hidden_layers": 36, "num_attention_heads": 32,
        "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 12288,
        "rms_norm_eps": 1e-6, "rope_theta": 1000000,
    }
    bad = {k: (cfg.get(k), v) for k, v in expected.items() if cfg.get(k) != v}
    print("[config]", "MATCH" if not bad else f"MISMATCH {bad}")
    return cfg, not bad


def check_structure(st, cfg):
    required = ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"]
    for layer in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{layer}"
        required += [f"{p}.input_layernorm.weight", f"{p}.post_attention_layernorm.weight",
                     f"{p}.self_attn.q_norm.weight", f"{p}.self_attn.k_norm.weight"]
        for linear in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                       "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
            prefix = f"{p}.{linear}"
            required += [f"{prefix}.{s}" for s in [
                "U_packed", "V_packed", "U_shape", "V_shape", "_split_dim_final",
                "u1", "u2", "v1", "v2"]]
            if cfg.get("residual"):
                required += [f"{prefix}.{s}" for s in [
                    "U_R_packed", "V_R_packed", "U_R_shape", "V_R_shape",
                    "u1_R", "u2_R", "v1_R", "v2_R"]]
    missing = [n for n in required if not st.has(n)]
    print("[tensors]", f"{len(required)} required, missing {len(missing)}", missing[:5])
    if missing:
        return False

    emb = st.info("model.embed_tokens.weight")
    head = st.info("lm_head.weight")
    shape_ok = (emb["shape"] == [cfg["vocab_size"], cfg["hidden_size"]] and
                emb["dtype"] == "BF16" and
                head["shape"] == emb["shape"] and head["dtype"] == "BF16")
    print("[embeddings]", "MATCH" if shape_ok else f"MISMATCH {emb['shape']} {head['shape']}")
    return shape_ok


def numpy_reference(st, prefix, x):
    def path(suffix):
        V = st.unpack(prefix + "V" + suffix + "_packed", prefix + "V" + suffix + "_shape")
        U = st.unpack(prefix + "U" + suffix + "_packed", prefix + "U" + suffix + "_shape")
        v2 = st.f32(prefix + "v2" + suffix).reshape(-1)
        v1 = st.f32(prefix + "v1" + suffix).reshape(-1)
        u2 = st.f32(prefix + "u2" + suffix).reshape(-1)
        u1 = st.f32(prefix + "u1" + suffix).reshape(-1)
        assert V.shape[1] == x.shape[0] == v2.shape[0]
        return ((((x * v2) @ V.T) * (v1 * u2)) @ U.T) * u1

    y = path("")
    if st.has(prefix + "U_R_packed"):
        y = y + path("_R")
    return y


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    model_dir = sys.argv[1]
    probe = sys.argv[2] if len(sys.argv) > 2 else "build/model_probe"
    cfg, cfg_ok = check_config(model_dir)
    st = SafeTensors(os.path.join(model_dir, "model.safetensors"))
    struct_ok = check_structure(st, cfg)

    prefix = "model.layers.0.self_attn.q_proj."
    in_feat = int(st.info(prefix + "v2")["shape"][-1])
    x = np.sin(np.arange(in_feat, dtype=np.float32) * 0.013).astype(np.float32)
    ref = numpy_reference(st, prefix, x)

    with tempfile.TemporaryDirectory() as td:
        xf, yf = os.path.join(td, "x.bin"), os.path.join(td, "y.bin")
        x.tofile(xf)
        out = subprocess.check_output([probe, os.path.join(model_dir, "model.safetensors"),
                                       prefix.rstrip("."), xf, yf], text=True)
        got = np.fromfile(yf, dtype=np.float32)

    if got.shape != ref.shape:
        print("[layer] SHAPE MISMATCH", got.shape, ref.shape)
        return 1
    max_abs = float(np.max(np.abs(got - ref)))
    rel = max_abs / max(1e-8, float(np.max(np.abs(ref))))
    ok = rel < 2e-3
    print(f"[layer] {prefix.rstrip('.')} in={in_feat} out={got.shape[0]} "
          f"max_abs={max_abs:.6g} rel={rel:.6g} {'MATCH' if ok else 'MISMATCH'}")
    print(f"[probe] {out.strip()}")
    print("RESULT:", "PASS" if (cfg_ok and struct_ok and ok) else "FAIL")
    return 0 if (cfg_ok and struct_ok and ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
