# Phase 9 Profiling — opt5_baseline_opt4_5

- Commit: `a056a5d`
- Generated (UTC): `2026-11-24T03:10:10Z`
- Command: `/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt4-baseline/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt5_baseline_opt4_5.json`
- Environment: `AMD EPYC 9354P 32-Core Processor`; nproc=4; flags=avx2 avx512f bmi2 popcnt sse4_2; governor=unknown
- Engine threads: `1`; CPU utilization: `99.4%` of one core (~24.8% of the 4-core machine)
- Real model: `/home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts`; layers=36; hidden=4096; vocab=151936

## Reproducibility

```bash
# build
cmake -S . -B /tmp/opencode/lbit-p9 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/lbit-p9 -j4 --target perf_benchmark
# profile
/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt4-baseline/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt5_baseline_opt4_5.json
```

The benchmark replays `Qwen3Model::forward_token` with timers and checks its logits against the engine on every replayed call: max abs diff = `0.0`. No engine source was changed for the baseline.

## End-to-end timing

| Metric | Value |
|---|---:|
| Model load | 875.922 ms |
| Prompt tokenization | 0.144 ms |
| Prefill (11 tokens) | 11066.063 ms |
| TTFT (tokenize + prefill) | 11066.207 ms |
| Decode (3 tokens) | 4987.222 ms |
| Decode per token | 1662.407 ms |
| RSS after profiling | 2326 MB |

## Top 3 bottlenecks (measured, not assumed)

| Rank | Component | Total replay ms | ms / forward | Share |
|---:|---|---:|---:|---:|
| 1 | LittleBit binary projections (all 7 per layer) | 10456.1 | 746.867 | 65.5% |
| 2 | lm_head (full vocab dot product) | 5388.7 | 384.905 | 33.8% |
| 3 | Top-p sampling (sort 151936 logits) | 42.9 | 3.067 | 0.3% |

## Full component breakdown

| Component | ms / forward | Share |
|---|---:|---:|
| LittleBit binary projections (all 7 per layer) | 746.867 | 65.5% |
| lm_head (full vocab dot product) | 384.905 | 33.8% |
| Top-p sampling (sort 151936 logits) | 3.067 | 0.3% |
| SwiGLU activation | 2.221 | 0.2% |
| Self-attention core (scores + softmax + V) | 1.351 | 0.1% |
| Input + post + final RMSNorm | 0.440 | 0.0% |
| QK RMSNorm | 0.352 | 0.0% |
| Residual adds | 0.159 | 0.0% |
| KV cache write | 0.082 | 0.0% |
| Rotary embedding | 0.075 | 0.0% |
| Greedy sampling | 0.024 | 0.0% |
| Embedding lookup | 0.002 | 0.0% |

## Per-layer cost

- Mean per layer per forward: `20.893 ms`
- Slowest 5 layers (total over 14 replayed forwards): L33=345.8 ms, L35=342.2 ms, L29=337.3 ms, L9=336.1 ms, L10=333.5 ms
- Total transformer time by forward index (prefill then decode): `685.5, 679.6, 660.9, 660.3, 681.6, 669.2, 695.1, 729.8, 676.9, 666.9, 664.7, 1223.0, 1136.0, 700.5` ms

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

- Projection effective bandwidth: `0.657 GB/s`
- `lm_head` bytes per call: `1.159 GiB`
- `lm_head` effective bandwidth (in-forward): `3.234 GB/s`
- `lm_head` isolated bandwidth (3 repeats): `3.695 GB/s`
- KV data read by attention core: `29.5 MiB` over 14 forwards

## Interpretation

The numbers above are the baseline before any optimization. Top-3 proposals will be written in `docs/phase9-profiling-baseline.md` after review; this generated file contains raw measurements only.
