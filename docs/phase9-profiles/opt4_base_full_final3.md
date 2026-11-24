# Phase 9 Profiling — opt4_base_full_final3

- Commit: `9605266`
- Generated (UTC): `2026-11-23T11:54:29Z`
- Command: `/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt4-base/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt4_base_full_final3.json`
- Environment: `AMD EPYC 9354P 32-Core Processor`; nproc=4; flags=avx2 avx512f bmi2 popcnt sse4_2; governor=unknown
- Engine threads: `1`; CPU utilization: `99.7%` of one core (~24.9% of the 4-core machine)
- Real model: `/home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts`; layers=36; hidden=4096; vocab=151936

## Reproducibility

```bash
# build
cmake -S . -B /tmp/opencode/lbit-p9 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/lbit-p9 -j4 --target perf_benchmark
# profile
/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt4-base/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt4_base_full_final3.json
```

The benchmark replays `Qwen3Model::forward_token` with timers and checks its logits against the engine on every replayed call: max abs diff = `0.0`. No engine source was changed for the baseline.

## End-to-end timing

| Metric | Value |
|---|---:|
| Model load | 998.540 ms |
| Prompt tokenization | 0.148 ms |
| Prefill (11 tokens) | 13767.815 ms |
| TTFT (tokenize + prefill) | 13767.962 ms |
| Decode (3 tokens) | 3597.733 ms |
| Decode per token | 1199.244 ms |
| RSS after profiling | 2328 MB |

## Top 3 bottlenecks (measured, not assumed)

| Rank | Component | Total replay ms | ms / forward | Share |
|---:|---|---:|---:|---:|
| 1 | LittleBit binary projections (all 7 per layer) | 10777.8 | 769.845 | 59.5% |
| 2 | lm_head (full vocab dot product) | 7248.3 | 517.737 | 40.0% |
| 3 | Top-p sampling (sort 151936 logits) | 36.4 | 2.601 | 0.2% |

## Full component breakdown

| Component | ms / forward | Share |
|---|---:|---:|
| LittleBit binary projections (all 7 per layer) | 769.845 | 59.5% |
| lm_head (full vocab dot product) | 517.737 | 40.0% |
| Top-p sampling (sort 151936 logits) | 2.601 | 0.2% |
| SwiGLU activation | 2.259 | 0.2% |
| Self-attention core (scores + softmax + V) | 1.103 | 0.1% |
| Input + post + final RMSNorm | 0.433 | 0.0% |
| QK RMSNorm | 0.231 | 0.0% |
| Residual adds | 0.160 | 0.0% |
| Rotary embedding | 0.090 | 0.0% |
| KV cache write | 0.081 | 0.0% |
| Greedy sampling | 0.024 | 0.0% |
| Embedding lookup | 0.002 | 0.0% |

## Per-layer cost

- Mean per layer per forward: `21.526 ms`
- Slowest 5 layers (total over 14 replayed forwards): L9=379.7 ms, L13=365.5 ms, L11=363.9 ms, L4=360.5 ms, L12=358.1 ms
- Total transformer time by forward index (prefill then decode): `673.7, 677.8, 742.5, 1825.2, 685.8, 685.7, 711.0, 704.3, 707.2, 702.2, 676.7, 684.1, 675.7, 697.4` ms

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

- Projection effective bandwidth: `0.637 GB/s`
- `lm_head` bytes per call: `1.159 GiB`
- `lm_head` effective bandwidth (in-forward): `2.404 GB/s`
- `lm_head` isolated bandwidth (3 repeats): `2.408 GB/s`
- KV data read by attention core: `29.5 MiB` over 14 forwards

## Interpretation

The numbers above are the baseline before any optimization. Top-3 proposals will be written in `docs/phase9-profiling-baseline.md` after review; this generated file contains raw measurements only.
