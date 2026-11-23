#!/usr/bin/env python3
"""Reproducible Phase 9 profiling runner for the REAL Qwen3/LittleBit checkpoint.

It runs tests/perf_benchmark.cpp with a fixed prompt and fixed decode token IDs,
records the full environment, and writes both JSON and a Markdown summary.

Usage:
  python3 scripts/benchmark_phase9.py [profile_binary] [model_dir] [tag]

Defaults are pinned to the baseline used in docs/phase9-profiling-baseline.md.
No engine source is modified by this script.
"""
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "/home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts"
DEFAULT_BIN = "/tmp/opencode/lbit-p9/perf_benchmark"
PROMPT = "مرحبا"
DECODE_TOKENS = "262,220,16"  # sampled from the real-model smoke run at commit 62ffde8
MAX_SEQ = "512"
DETAIL = "1"
LM_REPEATS = "3"


def sh(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=20).strip()
    except Exception as e:  # environment must be recorded, never silently omitted
        return f"UNAVAILABLE: {e}"


def collect_env():
    env = {
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "nproc": os.cpu_count(),
        "avx2_mode": "scalar-forced" if os.environ.get("LITTLEBIT_DISABLE_AVX2") not in (None, "", "0") else "avx2-enabled",
    }
    env["uname"] = sh(["uname", "-a"])
    env["cpu_model"] = sh(["bash", "-lc", "lscpu | grep -m1 'Model name' | cut -d: -f2- | xargs"])
    env["cpu_flags"] = sh(["bash", "-lc", "grep -m1 '^flags' /proc/cpuinfo | cut -d: -f2- | tr ' ' '\\n' | grep -E '^(avx2|avx512f|sse4_2|bmi2|popcnt)$' | sort | tr '\\n' ' '"])
    env["cpu_governor"] = sh(["bash", "-lc", "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown"])
    env["mem_total"] = sh(["bash", "-lc", "grep MemTotal /proc/meminfo"])
    env["mem_available"] = sh(["bash", "-lc", "grep MemAvailable /proc/meminfo"])
    env["affinity"] = sh(["bash", "-lc", "taskset -pc $$ 2>/dev/null || echo unknown"])
    return env


def fmt_ms(v):
    return f"{v:.3f} ms"


def fmt_gbs(v):
    return f"{v:.3f} GB/s"


def main():
    binary = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BIN
    model_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    tag = sys.argv[3] if len(sys.argv) > 3 else "baseline"
    if not Path(binary).is_file():
        print("Missing benchmark binary:", binary)
        return 2
    if not Path(model_dir, "model.safetensors").is_file():
        print("Missing real model:", model_dir)
        return 2

    out_dir = ROOT / "docs" / "phase9-profiles"
    out_dir.mkdir(exist_ok=True)
    json_path = out_dir / f"{tag}.json"
    md_path = out_dir / f"{tag}.md"
    raw_path = out_dir / f"{tag}.stdout.log"

    taskset = shutil.which("taskset")
    cores = f"0-{os.cpu_count() - 1}" if os.cpu_count() else None
    prefix = [taskset, "-c", cores] if taskset and cores else []
    cmd = prefix + [
        str(binary), str(model_dir), PROMPT, DECODE_TOKENS, MAX_SEQ, DETAIL,
        str(LM_REPEATS), str(json_path),
    ]
    reuse = os.environ.get("PHASE9_REUSE_JSON") == "1" and json_path.is_file()
    if reuse:
        print("[profile] reusing existing raw JSON:", json_path)
    else:
        print("[profile] running:", " ".join(cmd))
        cp = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        raw_path.write_text(cp.stdout, encoding="utf-8")
        print(cp.stdout.strip())
        if cp.returncode != 0 or not json_path.is_file():
            print("[profile] FAILED: benchmark did not produce JSON")
            return cp.returncode or 1

    data = json.loads(json_path.read_text(encoding="utf-8"))
    if "environment" not in data:
        env = collect_env()
        # The raw JSON was written at the end of the measurement; use its mtime
        # instead of report-generation time when regenerating a report.
        env["date_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(json_path.stat().st_mtime))
        data["environment"] = env
    data["command"] = " ".join(cmd)
    data["commit"] = sh(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"])

    f = lambda k: float(data[k])
    proj_total = sum(f(k) for k in [
        "q_proj_ms", "k_plus_v_proj_ms", "o_proj_ms",
        "gate_proj_ms", "up_proj_ms", "down_proj_ms",
    ])
    groups = {
        "LittleBit binary projections (all 7 per layer)": proj_total,
        "lm_head (full vocab dot product)": f("lm_head_ms"),
        "Self-attention core (scores + softmax + V)": f("attn_core_ms"),
        "Top-p sampling (sort 151936 logits)": f("top_p_sample_ms"),
        "QK RMSNorm": f("qk_norm_ms"),
        "Input + post + final RMSNorm": f("input_norm_ms") + f("post_norm_ms") + f("final_norm_ms"),
        "SwiGLU activation": f("swiglu_ms"),
        "KV cache write": f("kv_write_ms"),
        "Rotary embedding": f("rope_ms"),
        "Residual adds": f("attn_residual_ms") + f("mlp_residual_ms"),
        "Embedding lookup": f("embed_ms"),
        "Greedy sampling": f("greedy_sample_ms"),
    }
    replay_total = sum(groups.values())
    ranked = sorted(groups.items(), key=lambda x: x[1], reverse=True)
    calls = max(1, int(data["replay_calls"]))

    # Recompute bandwidth from raw byte/time fields. Older JSON files may have
    # stale derived values from before the unit fix; raw measurements are authoritative.
    proj_ms_per_call = proj_total / calls
    data["projection_GB_per_s"] = int(data["total_projection_bytes_per_token"]) / max(1e-9, proj_ms_per_call) / 1e6
    data["lm_head_GB_per_s"] = int(data["lm_head_bytes_per_call"]) / max(1e-9, f("lm_head_ms") / calls) / 1e6
    data["lm_head_isolated_GB_per_s"] = int(data["lm_head_bytes_per_call"]) / max(1e-9, f("lm_head_isolated_ms_per_call")) / 1e6
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    layers = [float(x) for x in data["layer_ms_csv"].strip('"').split(",") if x]
    positions = [float(x) for x in data["layers_ms_by_position_csv"].strip('"').split(",") if x]
    hot_layers = sorted(enumerate(layers), key=lambda x: x[1], reverse=True)[:5]
    proj_bytes = [int(data[f"proj_bytes_{n}"]) for n in ["q", "k", "v", "o", "gate", "up", "down"]]
    proj_names = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    lines = [
        f"# Phase 9 Profiling — {tag}",
        "",
        f"- Commit: `{data['commit']}`",
        f"- Generated (UTC): `{data['environment']['date_utc']}`",
        f"- Command: `{data['command']}`",
        f"- Environment: `{data['environment']['cpu_model']}`; nproc={data['environment']['nproc']}; "
        f"flags={data['environment']['cpu_flags'] or 'none reported'}; governor={data['environment']['cpu_governor']}",
        f"- Engine threads: `{data['threads']}`; CPU utilization: `{f('cpu_util_pct_of_one_core'):.1f}%` of one core "
        f"(~{f('cpu_util_pct_of_one_core') / max(1, int(data['environment']['nproc'])):.1f}% of the 4-core machine)",
        f"- Real model: `{model_dir}`; layers={data['num_layers']}; hidden={data['hidden_size']}; vocab={data['vocab_size']}",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "# build",
        "cmake -S . -B /tmp/opencode/lbit-p9 -DCMAKE_BUILD_TYPE=Release",
        "cmake --build /tmp/opencode/lbit-p9 -j4 --target perf_benchmark",
        "# profile",
        data["command"],
        "```",
        "",
        "The benchmark replays `Qwen3Model::forward_token` with timers and checks its logits against the "
        f"engine on every replayed call: max abs diff = `{data['replay_logit_max_abs']}`. "
        "No engine source was changed for the baseline.",
        "",
        "## End-to-end timing",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Model load | {fmt_ms(f('load_ms'))} |",
        f"| Prompt tokenization | {fmt_ms(f('tokenize_ms'))} |",
        f"| Prefill ({data['prompt_tokens']} tokens) | {fmt_ms(f('prefill_engine_ms'))} |",
        f"| TTFT (tokenize + prefill) | {fmt_ms(f('ttft_ms'))} |",
        f"| Decode ({data['decode_tokens']} tokens) | {fmt_ms(f('decode_engine_ms'))} |",
        f"| Decode per token | {fmt_ms(f('decode_ms_per_token'))} |",
        f"| RSS after profiling | {data['rss_mb']} MB |",
        "",
        "## Top 3 bottlenecks (measured, not assumed)",
        "",
        "| Rank | Component | Total replay ms | ms / forward | Share |",
        "|---:|---|---:|---:|---:|",
    ]
    for i, (name, ms) in enumerate(ranked[:3], 1):
        lines.append(f"| {i} | {name} | {ms:.1f} | {ms / calls:.3f} | {100 * ms / replay_total:.1f}% |")
    lines += [
        "",
        "## Full component breakdown",
        "",
        "| Component | ms / forward | Share |",
        "|---|---:|---:|",
    ]
    for name, ms in ranked:
        lines.append(f"| {name} | {ms / calls:.3f} | {100 * ms / replay_total:.1f}% |")

    lines += [
        "",
        "## Per-layer cost",
        "",
        f"- Mean per layer per forward: `{sum(layers) / max(1, len(layers)) / calls:.3f} ms`",
        f"- Slowest 5 layers (total over {calls} replayed forwards): " +
        ", ".join(f"L{i}={ms:.1f} ms" for i, ms in hot_layers),
        f"- Total transformer time by forward index (prefill then decode): `{', '.join(f'{x:.1f}' for x in positions)}` ms",
        "",
        "## Projection weight reads and bandwidth",
        "",
        "| Projection | Packed+scale bytes (all layers) |",
        "|---|---:|",
    ]
    for name, nbytes in zip(proj_names, proj_bytes):
        lines.append(f"| {name} | {nbytes / 2**20:.1f} MiB |")
    lines += [
        f"| **total** | {int(data['total_projection_bytes_per_token']) / 2**20:.1f} MiB / token |",
        "",
        f"- Projection effective bandwidth: `{fmt_gbs(f('projection_GB_per_s'))}`",
        f"- `lm_head` bytes per call: `{int(data['lm_head_bytes_per_call']) / 2**30:.3f} GiB`",
        f"- `lm_head` effective bandwidth (in-forward): `{fmt_gbs(f('lm_head_GB_per_s'))}`",
        f"- `lm_head` isolated bandwidth ({LM_REPEATS} repeats): `{fmt_gbs(f('lm_head_isolated_GB_per_s'))}`",
        f"- KV data read by attention core: `{int(data['kv_read_bytes']) / 2**20:.1f} MiB` over {calls} forwards",
        "",
        "## Interpretation",
        "",
        "The numbers above are the baseline before any optimization. Top-3 proposals will be written in "
        "`docs/phase9-profiling-baseline.md` after review; this generated file contains raw measurements only.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[profile] wrote {json_path}")
    print(f"[profile] wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
