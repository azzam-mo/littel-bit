# Phase 9 Profiling — opt2_scalar_run3

- Commit: `87c2048`
- Generated (UTC): `2026-11-23T10:16:27Z`
- Command: `/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt2/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt2_scalar_run3.json`
- Environment: `AMD EPYC 9354P 32-Core Processor`; nproc=4; flags=avx2 avx512f bmi2 popcnt sse4_2; governor=unknown
- Engine threads: `1`; CPU utilization: `99.6%` of one core (~24.9% of the 4-core machine)
- Real model: `/home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts`; layers=36; hidden=4096; vocab=151936

## Reproducibility

```bash
# build
cmake -S . -B /tmp/opencode/lbit-p9 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/lbit-p9 -j4 --target perf_benchmark
# profile
/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt2/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt2_scalar_run3.json
```

The benchmark replays `Qwen3Model::forward_token` with timers and checks its logits against the engine on every replayed call: max abs diff = `0.0`. No engine source was changed for the baseline.

## End-to-end timing

| Metric | Value |
|---|---:|
| Model load | 1184.320 ms |
| Prompt tokenization | 0.165 ms |
| Prefill (11 tokens) | 155522.800 ms |
| TTFT (tokenize + prefill) | 155522.965 ms |
| Decode (3 tokens) | 44118.939 ms |
| Decode per token | 14706.313 ms |
| RSS after profiling | 2332 MB |

## Top 3 bottlenecks (measured, not assumed)

| Rank | Component | Total replay ms | ms / forward | Share |
|---:|---|---:|---:|---:|
| 1 | LittleBit binary projections (all 7 per layer) | 188491.7 | 13463.696 | 95.9% |
| 2 | lm_head (full vocab dot product) | 7985.0 | 570.357 | 4.1% |
| 3 | Top-p sampling (sort 151936 logits) | 37.2 | 2.658 | 0.0% |

## Full component breakdown

| Component | ms / forward | Share |
|---|---:|---:|
| LittleBit binary projections (all 7 per layer) | 13463.696 | 95.9% |
| lm_head (full vocab dot product) | 570.357 | 4.1% |
| Top-p sampling (sort 151936 logits) | 2.658 | 0.0% |
| SwiGLU activation | 1.939 | 0.0% |
| Self-attention core (scores + softmax + V) | 1.134 | 0.0% |
| Input + post + final RMSNorm | 0.453 | 0.0% |
| QK RMSNorm | 0.268 | 0.0% |
| Residual adds | 0.184 | 0.0% |
| KV cache write | 0.093 | 0.0% |
| Rotary embedding | 0.089 | 0.0% |
| Greedy sampling | 0.025 | 0.0% |
| Embedding lookup | 0.002 | 0.0% |

## Per-layer cost

- Mean per layer per forward: `374.134 ms`
- Slowest 5 layers (total over 14 replayed forwards): L9=5660.1 ms, L25=5611.4 ms, L23=5570.5 ms, L0=5491.4 ms, L20=5442.4 ms
- Total transformer time by forward index (prefill then decode): `15579.4, 13612.1, 13041.0, 13178.9, 13770.4, 13038.9, 14539.9, 12673.9, 14843.1, 12811.8, 12803.2, 13273.7, 12639.8, 12757.2` ms

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

- Projection effective bandwidth: `0.036 GB/s`
- `lm_head` bytes per call: `1.159 GiB`
- `lm_head` effective bandwidth (in-forward): `2.182 GB/s`
- `lm_head` isolated bandwidth (3 repeats): `2.409 GB/s`
- KV data read by attention core: `29.5 MiB` over 14 forwards

## Interpretation

The numbers above are the baseline before any optimization. Top-3 proposals will be written in `docs/phase9-profiling-baseline.md` after review; this generated file contains raw measurements only.
