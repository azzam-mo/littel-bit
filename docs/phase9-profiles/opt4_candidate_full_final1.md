# Phase 9 Profiling — opt4_candidate_full_final1

- Commit: `9605266`
- Generated (UTC): `2026-11-23T11:55:03Z`
- Command: `/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt4/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt4_candidate_full_final1.json`
- Environment: `AMD EPYC 9354P 32-Core Processor`; nproc=4; flags=avx2 avx512f bmi2 popcnt sse4_2; governor=unknown
- Engine threads: `1`; CPU utilization: `98.9%` of one core (~24.7% of the 4-core machine)
- Real model: `/home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts`; layers=36; hidden=4096; vocab=151936

## Reproducibility

```bash
# build
cmake -S . -B /tmp/opencode/lbit-p9 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/lbit-p9 -j4 --target perf_benchmark
# profile
/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt4/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt4_candidate_full_final1.json
```

The benchmark replays `Qwen3Model::forward_token` with timers and checks its logits against the engine on every replayed call: max abs diff = `0.0`. No engine source was changed for the baseline.

## End-to-end timing

| Metric | Value |
|---|---:|
| Model load | 972.455 ms |
| Prompt tokenization | 0.156 ms |
| Prefill (11 tokens) | 11311.846 ms |
| TTFT (tokenize + prefill) | 11312.002 ms |
| Decode (3 tokens) | 3123.508 ms |
| Decode per token | 1041.169 ms |
| RSS after profiling | 2328 MB |

## Top 3 bottlenecks (measured, not assumed)

| Rank | Component | Total replay ms | ms / forward | Share |
|---:|---|---:|---:|---:|
| 1 | LittleBit binary projections (all 7 per layer) | 9761.2 | 697.226 | 64.9% |
| 2 | lm_head (full vocab dot product) | 5129.6 | 366.401 | 34.1% |
| 3 | Top-p sampling (sort 151936 logits) | 98.0 | 7.001 | 0.7% |

## Full component breakdown

| Component | ms / forward | Share |
|---|---:|---:|
| LittleBit binary projections (all 7 per layer) | 697.226 | 64.9% |
| lm_head (full vocab dot product) | 366.401 | 34.1% |
| Top-p sampling (sort 151936 logits) | 7.001 | 0.7% |
| SwiGLU activation | 1.947 | 0.2% |
| Self-attention core (scores + softmax + V) | 1.100 | 0.1% |
| Input + post + final RMSNorm | 0.513 | 0.0% |
| QK RMSNorm | 0.237 | 0.0% |
| Residual adds | 0.163 | 0.0% |
| KV cache write | 0.081 | 0.0% |
| Rotary embedding | 0.077 | 0.0% |
| Greedy sampling | 0.025 | 0.0% |
| Embedding lookup | 0.002 | 0.0% |

## Per-layer cost

- Mean per layer per forward: `19.494 ms`
- Slowest 5 layers (total over 14 replayed forwards): L23=312.6 ms, L20=302.0 ms, L6=293.8 ms, L25=280.6 ms, L10=280.5 ms
- Total transformer time by forward index (prefill then decode): `685.6, 687.6, 672.4, 690.2, 694.8, 689.6, 673.3, 672.9, 671.9, 692.5, 678.5, 730.5, 705.6, 879.7` ms

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
- `lm_head` effective bandwidth (in-forward): `3.397 GB/s`
- `lm_head` isolated bandwidth (3 repeats): `2.120 GB/s`
- KV data read by attention core: `29.5 MiB` over 14 forwards

## Interpretation

The numbers above are the baseline before any optimization. Top-3 proposals will be written in `docs/phase9-profiling-baseline.md` after review; this generated file contains raw measurements only.
