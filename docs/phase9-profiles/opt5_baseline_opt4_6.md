# Phase 9 Profiling — opt5_baseline_opt4_6

- Commit: `a056a5d`
- Generated (UTC): `2026-11-24T03:10:43Z`
- Command: `/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt4-baseline/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt5_baseline_opt4_6.json`
- Environment: `AMD EPYC 9354P 32-Core Processor`; nproc=4; flags=avx2 avx512f bmi2 popcnt sse4_2; governor=unknown
- Engine threads: `1`; CPU utilization: `99.8%` of one core (~24.9% of the 4-core machine)
- Real model: `/home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts`; layers=36; hidden=4096; vocab=151936

## Reproducibility

```bash
# build
cmake -S . -B /tmp/opencode/lbit-p9 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/lbit-p9 -j4 --target perf_benchmark
# profile
/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt4-baseline/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt5_baseline_opt4_6.json
```

The benchmark replays `Qwen3Model::forward_token` with timers and checks its logits against the engine on every replayed call: max abs diff = `0.0`. No engine source was changed for the baseline.

## End-to-end timing

| Metric | Value |
|---|---:|
| Model load | 981.179 ms |
| Prompt tokenization | 0.154 ms |
| Prefill (11 tokens) | 11607.693 ms |
| TTFT (tokenize + prefill) | 11607.847 ms |
| Decode (3 tokens) | 3128.799 ms |
| Decode per token | 1042.933 ms |
| RSS after profiling | 2326 MB |

## Top 3 bottlenecks (measured, not assumed)

| Rank | Component | Total replay ms | ms / forward | Share |
|---:|---|---:|---:|---:|
| 1 | LittleBit binary projections (all 7 per layer) | 9765.6 | 697.545 | 65.2% |
| 2 | lm_head (full vocab dot product) | 5120.4 | 365.741 | 34.2% |
| 3 | Top-p sampling (sort 151936 logits) | 41.8 | 2.985 | 0.3% |

## Full component breakdown

| Component | ms / forward | Share |
|---|---:|---:|
| LittleBit binary projections (all 7 per layer) | 697.545 | 65.2% |
| lm_head (full vocab dot product) | 365.741 | 34.2% |
| Top-p sampling (sort 151936 logits) | 2.985 | 0.3% |
| SwiGLU activation | 1.803 | 0.2% |
| Self-attention core (scores + softmax + V) | 1.109 | 0.1% |
| Input + post + final RMSNorm | 0.428 | 0.0% |
| QK RMSNorm | 0.233 | 0.0% |
| Residual adds | 0.171 | 0.0% |
| Rotary embedding | 0.087 | 0.0% |
| KV cache write | 0.086 | 0.0% |
| Greedy sampling | 0.026 | 0.0% |
| Embedding lookup | 0.003 | 0.0% |

## Per-layer cost

- Mean per layer per forward: `19.498 ms`
- Slowest 5 layers (total over 14 replayed forwards): L5=292.1 ms, L20=284.0 ms, L19=282.6 ms, L33=282.4 ms, L18=280.3 ms
- Total transformer time by forward index (prefill then decode): `722.8, 736.5, 704.3, 681.0, 693.3, 672.6, 755.6, 690.1, 691.0, 688.8, 718.1, 694.9, 686.4, 691.6` ms

## Projection weight reads and bandwidth

| Projection | Packed+scale bytes (all layers) |
|---|---:|
| q_proj | 40.8 MiB |
| k_proj | 10.8 MiB |
| v_proj | 10.8 MiB |
| o_proj | 40.8 MiB |
| gate_proj | 121.7 MiB |
| up_proj | 121.7 MiB |
| down_proj | 121.1 MiB |
| **total** | 467.7 MiB / token |

- Projection effective bandwidth: `0.703 GB/s`
- `lm_head` bytes per call: `1.159 GiB`
- `lm_head` effective bandwidth (in-forward): `3.403 GB/s`
- `lm_head` isolated bandwidth (3 repeats): `3.544 GB/s`
- KV data read by attention core: `29.5 MiB` over 14 forwards

## Interpretation

The numbers above are the baseline before any optimization. Top-3 proposals will be written in `docs/phase9-profiling-baseline.md` after review; this generated file contains raw measurements only.
