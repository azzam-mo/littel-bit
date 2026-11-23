#!/usr/bin/env python3
"""Calculate real memory requirements of a littlebit-qwen3 checkpoint.

Usage:
  python3 scripts/memory_budget.py <model_dir> [observed_rss_mb]

Prints measured file/header sizes plus theoretical KV-cache cost for several
context lengths. No weights are loaded into RAM by this script.
"""
import json
import os
import struct
import sys
from collections import Counter


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    model_dir = sys.argv[1]
    observed = int(sys.argv[2]) if len(sys.argv) > 2 else None
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    path = os.path.join(model_dir, "model.safetensors")
    file_bytes = os.path.getsize(path)
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    header.pop("__metadata__", None)
    dtype_bytes = Counter()
    for info in header.values():
        a, b = info["data_offsets"]
        dtype_bytes[info["dtype"]] += b - a

    kv_per_token = (cfg["num_hidden_layers"] * 2 * cfg["num_key_value_heads"] *
                    cfg["head_dim"] * 4)
    print(f"[file] {file_bytes / 2**30:.2f} GiB, header {n} bytes, tensors {len(header)}")
    for dtype, size in dtype_bytes.most_common():
        print(f"  {dtype}: {size / 2**20:.1f} MiB")
    if observed is not None:
        print(f"[observed] engine RSS after full load: {observed} MiB")
    print(f"[kv] {kv_per_token / 1024:.0f} KiB per token "
          f"({cfg['num_hidden_layers']} layers, {cfg['num_key_value_heads']} KV heads, "
          f"head_dim={cfg['head_dim']})")
    print("[kv] context budget:")
    for seq in [512, 1024, 2048, 4096, 8192, cfg["max_position_embeddings"]]:
        mib = kv_per_token * seq / 2**20
        print(f"  {seq:>6} tokens: {mib:>9.1f} MiB")
    print("[note] mmap file pages are backed by disk but become resident when touched;")
    print("       lm_head logits scan touches every embedding row on each token.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
