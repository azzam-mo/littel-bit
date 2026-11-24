# Phase 9 Profiling — opt5_candidate_g4_4

- Commit: `a056a5d`
- Generated (UTC): `2026-11-24T03:11:08Z`
- Command: `/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt4/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt5_candidate_g4_4.json`
- Environment: `AMD EPYC 9354P 32-Core Processor`; nproc=4; flags=avx2 avx512f bmi2 popcnt sse4_2; governor=unknown
- Engine threads: `1`; CPU utilization: `99.9%` of one core (~25.0% of the 4-core machine)
- Real model: `/home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts`; layers=36; hidden=4096; vocab=151936

## Reproducibility

```bash
# build
cmake -S . -B /tmp/opencode/lbit-p9 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/lbit-p9 -j4 --target perf_benchmark
# profile
/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt4/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt5_candidate_g4_4.json
```

The benchmark replays `Qwen3Model::forward_token` with timers and checks its logits against the engine on every replayed call: max abs diff = `0.0`. No engine source was changed for the baseline.

## End-to-end timing

| Metric | Value |
|---|---:|
| Model load | 934.047 ms |
| Prompt tokenization | 0.156 ms |
| Prefill (11 tokens) | 8617.904 ms |
| TTFT (tokenize + prefill) | 8618.060 ms |
| Decode (3 tokens) | 2184.262 ms |
| Decode per token | 728.087 ms |
| RSS after profiling | 2326 MB |

## Top 3 bottlenecks (measured, not assumed)

| Rank | Component | Total replay ms | ms / forward | Share |
|---:|---|---:|---:|---:|
| 1 | LittleBit binary projections (all 7 per layer) | 5769.6 | 412.117 | 54.3% |
| 2 | lm_head (full vocab dot product) | 4775.9 | 341.132 | 44.9% |
| 3 | Top-p sampling (sort 151936 logits) | 35.3 | 2.522 | 0.3% |

## Full component breakdown

| Component | ms / forward | Share |
|---|---:|---:|
| LittleBit binary projections (all 7 per layer) | 412.117 | 54.3% |
| lm_head (full vocab dot product) | 341.132 | 44.9% |
| Top-p sampling (sort 151936 logits) | 2.522 | 0.3% |
| SwiGLU activation | 1.744 | 0.2% |
| Self-attention core (scores + softmax + V) | 1.065 | 0.1% |
| Input + post + final RMSNorm | 0.417 | 0.1% |
| QK RMSNorm | 0.224 | 0.0% |
| Residual adds | 0.163 | 0.0% |
| KV cache write | 0.085 | 0.0% |
| Rotary embedding | 0.076 | 0.0% |
| Greedy sampling | 0.026 | 0.0% |
| Embedding lookup | 0.003 | 0.0% |

## Per-layer cost

- Mean per layer per forward: `11.563 ms`
- Slowest 5 layers (total over 14 replayed forwards): L3=176.1 ms, L10=172.5 ms, L9=172.1 ms, L14=169.4 ms, L21=168.0 ms
- Total transformer time by forward index (prefill then decode): `432.2, 417.1, 431.3, 392.0, 424.9, 413.8, 397.5, 415.0, 393.0, 434.8, 489.1, 388.6, 402.7, 395.4` ms

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

- Projection effective bandwidth: `1.190 GB/s`
- `lm_head` bytes per call: `1.159 GiB`
- `lm_head` effective bandwidth (in-forward): `3.649 GB/s`
- `lm_head` isolated bandwidth (3 repeats): `3.909 GB/s`
- KV data read by attention core: `29.5 MiB` over 14 forwards

## Interpretation

The numbers above are the baseline before any optimization. Top-3 proposals will be written in `docs/phase9-profiling-baseline.md` after review; this generated file contains raw measurements only.
