# Phase 9 Profiling — baseline

- Commit: `62ffde8`
- Generated (UTC): `2026-11-23T07:23:13Z`
- Command: `/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/baseline.json`
- Environment: `AMD EPYC 9354P 32-Core Processor`; nproc=4; flags=avx2 avx512f bmi2 popcnt sse4_2; governor=unknown
- Engine threads: `1`; CPU utilization: `99.7%` of one core (~24.9% of the 4-core machine)
- Real model: `/home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts`; layers=36; hidden=4096; vocab=151936

## Reproducibility

```bash
# build
cmake -S . -B /tmp/opencode/lbit-p9 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/lbit-p9 -j4 --target perf_benchmark
# profile
/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/baseline.json
```

The benchmark replays `Qwen3Model::forward_token` with timers and checks its logits against the engine on every replayed call: max abs diff = `0.0`. No engine source was changed for the baseline.

## End-to-end timing

| Metric | Value |
|---|---:|
| Model load | 927.225 ms |
| Prompt tokenization | 0.142 ms |
| Prefill (11 tokens) | 142572.132 ms |
| TTFT (tokenize + prefill) | 142572.274 ms |
| Decode (3 tokens) | 40180.990 ms |
| Decode per token | 13393.663 ms |
| RSS after profiling | 2332 MB |

## Top 3 bottlenecks (measured, not assumed)

| Rank | Component | Total replay ms | ms / forward | Share |
|---:|---|---:|---:|---:|
| 1 | LittleBit binary projections (all 7 per layer) | 175612.8 | 12543.771 | 96.0% |
| 2 | lm_head (full vocab dot product) | 7297.4 | 521.243 | 4.0% |
| 3 | Top-p sampling (sort 151936 logits) | 36.4 | 2.603 | 0.0% |

## Full component breakdown

| Component | ms / forward | Share |
|---|---:|---:|
| LittleBit binary projections (all 7 per layer) | 12543.771 | 96.0% |
| lm_head (full vocab dot product) | 521.243 | 4.0% |
| Top-p sampling (sort 151936 logits) | 2.603 | 0.0% |
| SwiGLU activation | 1.693 | 0.0% |
| Self-attention core (scores + softmax + V) | 1.020 | 0.0% |
| Input + post + final RMSNorm | 0.463 | 0.0% |
| Residual adds | 0.225 | 0.0% |
| QK RMSNorm | 0.209 | 0.0% |
| Rotary embedding | 0.095 | 0.0% |
| KV cache write | 0.087 | 0.0% |
| Greedy sampling | 0.025 | 0.0% |
| Embedding lookup | 0.003 | 0.0% |

## Per-layer cost

- Mean per layer per forward: `348.572 ms`
- Slowest 5 layers (total over 14 replayed forwards): L25=6546.0 ms, L27=5312.9 ms, L11=5239.7 ms, L26=5231.6 ms, L10=5000.2 ms
- Total transformer time by forward index (prefill then decode): `12116.7, 12145.9, 12209.0, 12223.9, 12374.9, 12134.3, 15757.6, 12367.2, 12405.1, 12064.6, 12070.1, 12299.8, 12107.1, 13403.8` ms

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

- Projection effective bandwidth: `0.039 GB/s`
- `lm_head` bytes per call: `1.159 GiB`
- `lm_head` effective bandwidth (in-forward): `2.388 GB/s`
- `lm_head` isolated bandwidth (3 repeats): `2.427 GB/s`
- KV data read by attention core: `29.5 MiB` over 14 forwards

## Interpretation

The numbers above are the baseline before any optimization. Top-3 proposals will be written in `docs/phase9-profiling-baseline.md` after review; this generated file contains raw measurements only.
