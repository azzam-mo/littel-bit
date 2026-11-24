# Phase 9 Profiling — opt6_candidate_run3

- Commit: `1f3ad54`
- Generated (UTC): `2026-11-24T04:15:52Z`
- Command: `/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt6/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt6_candidate_run3.json`
- Environment: `AMD EPYC 9354P 32-Core Processor`; nproc=4; flags=avx2 avx512f bmi2 popcnt sse4_2; governor=unknown
- Engine threads: `1`; CPU utilization: `99.6%` of one core (~24.9% of the 4-core machine)
- Real model: `/home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts`; layers=36; hidden=4096; vocab=151936

## Reproducibility

```bash
# build
cmake -S . -B /tmp/opencode/lbit-p9 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/lbit-p9 -j4 --target perf_benchmark
# profile
/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt6/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt6_candidate_run3.json
```

The benchmark replays `Qwen3Model::forward_token` with timers and checks its logits against the engine on every replayed call: max abs diff = `0.0`. No engine source was changed for the baseline.

## End-to-end timing

| Metric | Value |
|---|---:|
| Model load | 1109.840 ms |
| Prompt tokenization | 0.137 ms |
| Prefill (11 tokens) | 6208.823 ms |
| TTFT (tokenize + prefill) | 6208.960 ms |
| Decode (3 tokens) | 1854.498 ms |
| Decode per token | 618.166 ms |
| RSS after profiling | 2326 MB |

## Top 3 bottlenecks (measured, not assumed)

| Rank | Component | Total replay ms | ms / forward | Share |
|---:|---|---:|---:|---:|
| 1 | LittleBit binary projections (all 7 per layer) | 6239.4 | 445.672 | 79.5% |
| 2 | lm_head (full vocab dot product) | 1499.2 | 107.086 | 19.1% |
| 3 | Top-p sampling (sort 151936 logits) | 44.0 | 3.142 | 0.6% |

## Full component breakdown

| Component | ms / forward | Share |
|---|---:|---:|
| LittleBit binary projections (all 7 per layer) | 445.672 | 79.5% |
| lm_head (full vocab dot product) | 107.086 | 19.1% |
| Top-p sampling (sort 151936 logits) | 3.142 | 0.6% |
| SwiGLU activation | 2.021 | 0.4% |
| Self-attention core (scores + softmax + V) | 1.379 | 0.2% |
| Input + post + final RMSNorm | 0.537 | 0.1% |
| QK RMSNorm | 0.226 | 0.0% |
| Residual adds | 0.162 | 0.0% |
| KV cache write | 0.081 | 0.0% |
| Rotary embedding | 0.079 | 0.0% |
| Greedy sampling | 0.024 | 0.0% |
| Embedding lookup | 0.003 | 0.0% |

## Per-layer cost

- Mean per layer per forward: `12.516 ms`
- Slowest 5 layers (total over 14 replayed forwards): L21=194.9 ms, L22=183.6 ms, L15=183.4 ms, L13=180.5 ms, L6=180.3 ms
- Total transformer time by forward index (prefill then decode): `460.6, 434.0, 485.9, 452.3, 432.2, 460.1, 419.8, 484.7, 451.3, 435.8, 424.9, 447.1, 477.5, 441.5` ms

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

- Projection effective bandwidth: `1.100 GB/s`
- `lm_head` bytes per call: `1.159 GiB`
- `lm_head` effective bandwidth (in-forward): `11.623 GB/s`
- `lm_head` isolated bandwidth (3 repeats): `12.863 GB/s`
- KV data read by attention core: `29.5 MiB` over 14 forwards

## Interpretation

The numbers above are the baseline before any optimization. Top-3 proposals will be written in `docs/phase9-profiling-baseline.md` after review; this generated file contains raw measurements only.
