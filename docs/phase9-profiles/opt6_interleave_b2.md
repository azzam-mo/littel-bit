# Phase 9 Profiling — opt6_interleave_b2

- Commit: `1f3ad54`
- Generated (UTC): `2026-11-24T04:18:36Z`
- Command: `/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt4/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt6_interleave_b2.json`
- Environment: `AMD EPYC 9354P 32-Core Processor`; nproc=4; flags=avx2 avx512f bmi2 popcnt sse4_2; governor=unknown
- Engine threads: `1`; CPU utilization: `99.8%` of one core (~25.0% of the 4-core machine)
- Real model: `/home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts`; layers=36; hidden=4096; vocab=151936

## Reproducibility

```bash
# build
cmake -S . -B /tmp/opencode/lbit-p9 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/lbit-p9 -j4 --target perf_benchmark
# profile
/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt4/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt6_interleave_b2.json
```

The benchmark replays `Qwen3Model::forward_token` with timers and checks its logits against the engine on every replayed call: max abs diff = `0.0`. No engine source was changed for the baseline.

## End-to-end timing

| Metric | Value |
|---|---:|
| Model load | 912.953 ms |
| Prompt tokenization | 0.147 ms |
| Prefill (11 tokens) | 8380.523 ms |
| TTFT (tokenize + prefill) | 8380.670 ms |
| Decode (3 tokens) | 2151.708 ms |
| Decode per token | 717.236 ms |
| RSS after profiling | 2326 MB |

## Top 3 bottlenecks (measured, not assumed)

| Rank | Component | Total replay ms | ms / forward | Share |
|---:|---|---:|---:|---:|
| 1 | LittleBit binary projections (all 7 per layer) | 5715.9 | 408.276 | 54.0% |
| 2 | lm_head (full vocab dot product) | 4773.3 | 340.949 | 45.1% |
| 3 | Top-p sampling (sort 151936 logits) | 35.6 | 2.546 | 0.3% |

## Full component breakdown

| Component | ms / forward | Share |
|---|---:|---:|
| LittleBit binary projections (all 7 per layer) | 408.276 | 54.0% |
| lm_head (full vocab dot product) | 340.949 | 45.1% |
| Top-p sampling (sort 151936 logits) | 2.546 | 0.3% |
| SwiGLU activation | 1.777 | 0.2% |
| Self-attention core (scores + softmax + V) | 1.095 | 0.1% |
| Input + post + final RMSNorm | 0.417 | 0.1% |
| QK RMSNorm | 0.231 | 0.0% |
| Residual adds | 0.157 | 0.0% |
| KV cache write | 0.074 | 0.0% |
| Rotary embedding | 0.071 | 0.0% |
| Greedy sampling | 0.024 | 0.0% |
| Embedding lookup | 0.003 | 0.0% |

## Per-layer cost

- Mean per layer per forward: `11.457 ms`
- Slowest 5 layers (total over 14 replayed forwards): L26=167.5 ms, L33=167.1 ms, L4=164.8 ms, L8=164.7 ms, L9=164.3 ms
- Total transformer time by forward index (prefill then decode): `399.2, 423.7, 402.7, 429.7, 401.6, 424.9, 415.0, 409.2, 399.3, 409.3, 414.8, 412.9, 414.9, 417.3` ms

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

- Projection effective bandwidth: `1.201 GB/s`
- `lm_head` bytes per call: `1.159 GiB`
- `lm_head` effective bandwidth (in-forward): `3.651 GB/s`
- `lm_head` isolated bandwidth (3 repeats): `3.587 GB/s`
- KV data read by attention core: `29.5 MiB` over 14 forwards

## Interpretation

The numbers above are the baseline before any optimization. Top-3 proposals will be written in `docs/phase9-profiling-baseline.md` after review; this generated file contains raw measurements only.
