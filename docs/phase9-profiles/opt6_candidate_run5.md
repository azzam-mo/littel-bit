# Phase 9 Profiling — opt6_candidate_run5

- Commit: `1f3ad54`
- Generated (UTC): `2026-11-24T04:16:30Z`
- Command: `/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt6/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt6_candidate_run5.json`
- Environment: `AMD EPYC 9354P 32-Core Processor`; nproc=4; flags=avx2 avx512f bmi2 popcnt sse4_2; governor=unknown
- Engine threads: `1`; CPU utilization: `99.9%` of one core (~25.0% of the 4-core machine)
- Real model: `/home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts`; layers=36; hidden=4096; vocab=151936

## Reproducibility

```bash
# build
cmake -S . -B /tmp/opencode/lbit-p9 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/lbit-p9 -j4 --target perf_benchmark
# profile
/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt6/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt6_candidate_run5.json
```

The benchmark replays `Qwen3Model::forward_token` with timers and checks its logits against the engine on every replayed call: max abs diff = `0.0`. No engine source was changed for the baseline.

## End-to-end timing

| Metric | Value |
|---|---:|
| Model load | 911.333 ms |
| Prompt tokenization | 0.134 ms |
| Prefill (11 tokens) | 5871.374 ms |
| TTFT (tokenize + prefill) | 5871.508 ms |
| Decode (3 tokens) | 1516.295 ms |
| Decode per token | 505.432 ms |
| RSS after profiling | 2326 MB |

## Top 3 bottlenecks (measured, not assumed)

| Rank | Component | Total replay ms | ms / forward | Share |
|---:|---|---:|---:|---:|
| 1 | LittleBit binary projections (all 7 per layer) | 5890.9 | 420.775 | 79.7% |
| 2 | lm_head (full vocab dot product) | 1407.9 | 100.562 | 19.1% |
| 3 | Top-p sampling (sort 151936 logits) | 35.2 | 2.517 | 0.5% |

## Full component breakdown

| Component | ms / forward | Share |
|---|---:|---:|
| LittleBit binary projections (all 7 per layer) | 420.775 | 79.7% |
| lm_head (full vocab dot product) | 100.562 | 19.1% |
| Top-p sampling (sort 151936 logits) | 2.517 | 0.5% |
| SwiGLU activation | 1.826 | 0.3% |
| Self-attention core (scores + softmax + V) | 1.140 | 0.2% |
| Input + post + final RMSNorm | 0.417 | 0.1% |
| QK RMSNorm | 0.227 | 0.0% |
| Residual adds | 0.155 | 0.0% |
| Rotary embedding | 0.096 | 0.0% |
| KV cache write | 0.078 | 0.0% |
| Greedy sampling | 0.024 | 0.0% |
| Embedding lookup | 0.003 | 0.0% |

## Per-layer cost

- Mean per layer per forward: `11.808 ms`
- Slowest 5 layers (total over 14 replayed forwards): L18=176.7 ms, L19=172.3 ms, L14=171.8 ms, L22=170.7 ms, L17=169.5 ms
- Total transformer time by forward index (prefill then decode): `430.0, 442.7, 423.3, 454.0, 417.6, 429.3, 421.9, 433.9, 423.2, 433.3, 414.8, 411.4, 387.2, 428.6` ms

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

- Projection effective bandwidth: `1.165 GB/s`
- `lm_head` bytes per call: `1.159 GiB`
- `lm_head` effective bandwidth (in-forward): `12.377 GB/s`
- `lm_head` isolated bandwidth (3 repeats): `10.339 GB/s`
- KV data read by attention core: `29.5 MiB` over 14 forwards

## Interpretation

The numbers above are the baseline before any optimization. Top-3 proposals will be written in `docs/phase9-profiling-baseline.md` after review; this generated file contains raw measurements only.
