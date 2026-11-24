# Phase 9 Profiling — opt4_candidate_full_final2

- Commit: `9605266`
- Generated (UTC): `2026-11-23T11:55:37Z`
- Command: `/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt4/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt4_candidate_full_final2.json`
- Environment: `AMD EPYC 9354P 32-Core Processor`; nproc=4; flags=avx2 avx512f bmi2 popcnt sse4_2; governor=unknown
- Engine threads: `1`; CPU utilization: `99.8%` of one core (~25.0% of the 4-core machine)
- Real model: `/home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts`; layers=36; hidden=4096; vocab=151936

## Reproducibility

```bash
# build
cmake -S . -B /tmp/opencode/lbit-p9 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/lbit-p9 -j4 --target perf_benchmark
# profile
/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt4/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt4_candidate_full_final2.json
```

The benchmark replays `Qwen3Model::forward_token` with timers and checks its logits against the engine on every replayed call: max abs diff = `0.0`. No engine source was changed for the baseline.

## End-to-end timing

| Metric | Value |
|---|---:|
| Model load | 1091.631 ms |
| Prompt tokenization | 0.136 ms |
| Prefill (11 tokens) | 11606.879 ms |
| TTFT (tokenize + prefill) | 11607.015 ms |
| Decode (3 tokens) | 3180.714 ms |
| Decode per token | 1060.238 ms |
| RSS after profiling | 2328 MB |

## Top 3 bottlenecks (measured, not assumed)

| Rank | Component | Total replay ms | ms / forward | Share |
|---:|---|---:|---:|---:|
| 1 | LittleBit binary projections (all 7 per layer) | 9710.0 | 693.571 | 66.6% |
| 2 | lm_head (full vocab dot product) | 4774.1 | 341.010 | 32.7% |
| 3 | Top-p sampling (sort 151936 logits) | 36.2 | 2.589 | 0.2% |

## Full component breakdown

| Component | ms / forward | Share |
|---|---:|---:|
| LittleBit binary projections (all 7 per layer) | 693.571 | 66.6% |
| lm_head (full vocab dot product) | 341.010 | 32.7% |
| Top-p sampling (sort 151936 logits) | 2.589 | 0.2% |
| SwiGLU activation | 1.840 | 0.2% |
| Self-attention core (scores + softmax + V) | 1.107 | 0.1% |
| Input + post + final RMSNorm | 0.451 | 0.0% |
| Rotary embedding | 0.242 | 0.0% |
| QK RMSNorm | 0.233 | 0.0% |
| Residual adds | 0.213 | 0.0% |
| KV cache write | 0.084 | 0.0% |
| Greedy sampling | 0.043 | 0.0% |
| Embedding lookup | 0.003 | 0.0% |

## Per-layer cost

- Mean per layer per forward: `19.395 ms`
- Slowest 5 layers (total over 14 replayed forwards): L20=285.5 ms, L8=283.5 ms, L9=280.8 ms, L23=279.5 ms, L16=279.4 ms
- Total transformer time by forward index (prefill then decode): `701.4, 710.8, 693.3, 692.1, 709.5, 707.7, 670.9, 731.2, 667.8, 695.2, 697.0, 692.9, 718.7, 686.5` ms

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

- Projection effective bandwidth: `0.707 GB/s`
- `lm_head` bytes per call: `1.159 GiB`
- `lm_head` effective bandwidth (in-forward): `3.650 GB/s`
- `lm_head` isolated bandwidth (3 repeats): `3.773 GB/s`
- KV data read by attention core: `29.5 MiB` over 14 forwards

## Interpretation

The numbers above are the baseline before any optimization. Top-3 proposals will be written in `docs/phase9-profiling-baseline.md` after review; this generated file contains raw measurements only.
