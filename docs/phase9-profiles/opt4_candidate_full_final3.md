# Phase 9 Profiling — opt4_candidate_full_final3

- Commit: `9605266`
- Generated (UTC): `2026-11-23T11:56:13Z`
- Command: `/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt4/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt4_candidate_full_final3.json`
- Environment: `AMD EPYC 9354P 32-Core Processor`; nproc=4; flags=avx2 avx512f bmi2 popcnt sse4_2; governor=unknown
- Engine threads: `1`; CPU utilization: `99.2%` of one core (~24.8% of the 4-core machine)
- Real model: `/home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts`; layers=36; hidden=4096; vocab=151936

## Reproducibility

```bash
# build
cmake -S . -B /tmp/opencode/lbit-p9 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/lbit-p9 -j4 --target perf_benchmark
# profile
/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt4/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt4_candidate_full_final3.json
```

The benchmark replays `Qwen3Model::forward_token` with timers and checks its logits against the engine on every replayed call: max abs diff = `0.0`. No engine source was changed for the baseline.

## End-to-end timing

| Metric | Value |
|---|---:|
| Model load | 952.240 ms |
| Prompt tokenization | 0.134 ms |
| Prefill (11 tokens) | 12538.047 ms |
| TTFT (tokenize + prefill) | 12538.180 ms |
| Decode (3 tokens) | 3271.677 ms |
| Decode per token | 1090.559 ms |
| RSS after profiling | 2328 MB |

## Top 3 bottlenecks (measured, not assumed)

| Rank | Component | Total replay ms | ms / forward | Share |
|---:|---|---:|---:|---:|
| 1 | LittleBit binary projections (all 7 per layer) | 10351.7 | 739.410 | 64.9% |
| 2 | lm_head (full vocab dot product) | 5503.8 | 393.130 | 34.5% |
| 3 | Top-p sampling (sort 151936 logits) | 36.3 | 2.595 | 0.2% |

## Full component breakdown

| Component | ms / forward | Share |
|---|---:|---:|
| LittleBit binary projections (all 7 per layer) | 739.410 | 64.9% |
| lm_head (full vocab dot product) | 393.130 | 34.5% |
| Top-p sampling (sort 151936 logits) | 2.595 | 0.2% |
| SwiGLU activation | 2.035 | 0.2% |
| Self-attention core (scores + softmax + V) | 1.172 | 0.1% |
| Input + post + final RMSNorm | 0.458 | 0.0% |
| QK RMSNorm | 0.368 | 0.0% |
| Residual adds | 0.166 | 0.0% |
| Rotary embedding | 0.088 | 0.0% |
| KV cache write | 0.085 | 0.0% |
| Greedy sampling | 0.026 | 0.0% |
| Embedding lookup | 0.002 | 0.0% |

## Per-layer cost

- Mean per layer per forward: `20.674 ms`
- Slowest 5 layers (total over 14 replayed forwards): L10=347.3 ms, L24=323.2 ms, L1=320.4 ms, L0=318.6 ms, L5=304.9 ms
- Total transformer time by forward index (prefill then decode): `681.5, 684.7, 696.2, 672.7, 736.7, 767.6, 750.5, 698.2, 706.2, 781.4, 1067.7, 775.0, 721.2, 680.3` ms

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

- Projection effective bandwidth: `0.663 GB/s`
- `lm_head` bytes per call: `1.159 GiB`
- `lm_head` effective bandwidth (in-forward): `3.166 GB/s`
- `lm_head` isolated bandwidth (3 repeats): `3.480 GB/s`
- KV data read by attention core: `29.5 MiB` over 14 forwards

## Interpretation

The numbers above are the baseline before any optimization. Top-3 proposals will be written in `docs/phase9-profiling-baseline.md` after review; this generated file contains raw measurements only.
