# Phase 9 Profiling — opt2_scalar_run1

- Commit: `87c2048`
- Generated (UTC): `2026-11-23T10:03:04Z`
- Command: `/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt2/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt2_scalar_run1.json`
- Environment: `AMD EPYC 9354P 32-Core Processor`; nproc=4; flags=avx2 avx512f bmi2 popcnt sse4_2; governor=unknown
- Engine threads: `1`; CPU utilization: `99.5%` of one core (~24.9% of the 4-core machine)
- Real model: `/home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts`; layers=36; hidden=4096; vocab=151936

## Reproducibility

```bash
# build
cmake -S . -B /tmp/opencode/lbit-p9 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/lbit-p9 -j4 --target perf_benchmark
# profile
/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt2/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt2_scalar_run1.json
```

The benchmark replays `Qwen3Model::forward_token` with timers and checks its logits against the engine on every replayed call: max abs diff = `0.0`. No engine source was changed for the baseline.

## End-to-end timing

| Metric | Value |
|---|---:|
| Model load | 1395.735 ms |
| Prompt tokenization | 0.161 ms |
| Prefill (11 tokens) | 154217.115 ms |
| TTFT (tokenize + prefill) | 154217.276 ms |
| Decode (3 tokens) | 40630.244 ms |
| Decode per token | 13543.415 ms |
| RSS after profiling | 2332 MB |

## Top 3 bottlenecks (measured, not assumed)

| Rank | Component | Total replay ms | ms / forward | Share |
|---:|---|---:|---:|---:|
| 1 | LittleBit binary projections (all 7 per layer) | 196399.2 | 14028.511 | 96.1% |
| 2 | lm_head (full vocab dot product) | 7840.8 | 560.055 | 3.8% |
| 3 | Top-p sampling (sort 151936 logits) | 36.5 | 2.608 | 0.0% |

## Full component breakdown

| Component | ms / forward | Share |
|---|---:|---:|
| LittleBit binary projections (all 7 per layer) | 14028.511 | 96.1% |
| lm_head (full vocab dot product) | 560.055 | 3.8% |
| Top-p sampling (sort 151936 logits) | 2.608 | 0.0% |
| SwiGLU activation | 2.072 | 0.0% |
| Self-attention core (scores + softmax + V) | 1.263 | 0.0% |
| Input + post + final RMSNorm | 0.464 | 0.0% |
| QK RMSNorm | 0.424 | 0.0% |
| Residual adds | 0.198 | 0.0% |
| KV cache write | 0.098 | 0.0% |
| Rotary embedding | 0.092 | 0.0% |
| Greedy sampling | 0.026 | 0.0% |
| Embedding lookup | 0.002 | 0.0% |

## Per-layer cost

- Mean per layer per forward: `389.836 ms`
- Slowest 5 layers (total over 14 replayed forwards): L27=7448.3 ms, L26=6942.1 ms, L6=6289.9 ms, L15=5915.7 ms, L14=5714.7 ms
- Total transformer time by forward index (prefill then decode): `13082.3, 12957.9, 13124.4, 15134.5, 12916.1, 14467.3, 13041.5, 18266.5, 14272.0, 14807.0, 13141.8, 14518.9, 13099.4, 13647.5` ms

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

- Projection effective bandwidth: `0.035 GB/s`
- `lm_head` bytes per call: `1.159 GiB`
- `lm_head` effective bandwidth (in-forward): `2.222 GB/s`
- `lm_head` isolated bandwidth (3 repeats): `1.738 GB/s`
- KV data read by attention core: `29.5 MiB` over 14 forwards

## Interpretation

The numbers above are the baseline before any optimization. Top-3 proposals will be written in `docs/phase9-profiling-baseline.md` after review; this generated file contains raw measurements only.
