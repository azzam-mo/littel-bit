# Phase 9 Profiling — opt2_avx2_run1

- Commit: `87c2048`
- Generated (UTC): `2026-11-23T10:17:07Z`
- Command: `/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt2/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt2_avx2_run1.json`
- Environment: `AMD EPYC 9354P 32-Core Processor`; nproc=4; flags=avx2 avx512f bmi2 popcnt sse4_2; governor=unknown
- Engine threads: `1`; CPU utilization: `99.6%` of one core (~24.9% of the 4-core machine)
- Real model: `/home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts`; layers=36; hidden=4096; vocab=151936

## Reproducibility

```bash
# build
cmake -S . -B /tmp/opencode/lbit-p9 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/lbit-p9 -j4 --target perf_benchmark
# profile
/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt2/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt2_avx2_run1.json
```

The benchmark replays `Qwen3Model::forward_token` with timers and checks its logits against the engine on every replayed call: max abs diff = `0.0`. No engine source was changed for the baseline.

## End-to-end timing

| Metric | Value |
|---|---:|
| Model load | 892.135 ms |
| Prompt tokenization | 0.146 ms |
| Prefill (11 tokens) | 13258.091 ms |
| TTFT (tokenize + prefill) | 13258.238 ms |
| Decode (3 tokens) | 4411.517 ms |
| Decode per token | 1470.506 ms |
| RSS after profiling | 2332 MB |

## Top 3 bottlenecks (measured, not assumed)

| Rank | Component | Total replay ms | ms / forward | Share |
|---:|---|---:|---:|---:|
| 1 | LittleBit binary projections (all 7 per layer) | 10328.5 | 737.753 | 58.2% |
| 2 | lm_head (full vocab dot product) | 7327.3 | 523.375 | 41.3% |
| 3 | Top-p sampling (sort 151936 logits) | 38.2 | 2.731 | 0.2% |

## Full component breakdown

| Component | ms / forward | Share |
|---|---:|---:|
| LittleBit binary projections (all 7 per layer) | 737.753 | 58.2% |
| lm_head (full vocab dot product) | 523.375 | 41.3% |
| Top-p sampling (sort 151936 logits) | 2.731 | 0.2% |
| SwiGLU activation | 1.832 | 0.1% |
| Self-attention core (scores + softmax + V) | 1.157 | 0.1% |
| Input + post + final RMSNorm | 0.433 | 0.0% |
| QK RMSNorm | 0.231 | 0.0% |
| Residual adds | 0.165 | 0.0% |
| KV cache write | 0.082 | 0.0% |
| Rotary embedding | 0.079 | 0.0% |
| Greedy sampling | 0.023 | 0.0% |
| Embedding lookup | 0.002 | 0.0% |

## Per-layer cost

- Mean per layer per forward: `20.617 ms`
- Slowest 5 layers (total over 14 replayed forwards): L14=396.4 ms, L15=354.9 ms, L17=348.4 ms, L16=345.8 ms, L7=328.6 ms
- Total transformer time by forward index (prefill then decode): `706.2, 676.8, 689.1, 661.6, 679.8, 672.4, 715.2, 705.5, 688.1, 687.5, 691.4, 768.0, 1231.1, 818.1` ms

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

- Projection effective bandwidth: `0.665 GB/s`
- `lm_head` bytes per call: `1.159 GiB`
- `lm_head` effective bandwidth (in-forward): `2.378 GB/s`
- `lm_head` isolated bandwidth (3 repeats): `2.299 GB/s`
- KV data read by attention core: `29.5 MiB` over 14 forwards

## Interpretation

The numbers above are the baseline before any optimization. Top-3 proposals will be written in `docs/phase9-profiling-baseline.md` after review; this generated file contains raw measurements only.
