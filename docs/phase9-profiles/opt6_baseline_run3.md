# Phase 9 Profiling — opt6_baseline_run3

- Commit: `1f3ad54`
- Generated (UTC): `2026-11-24T04:11:26Z`
- Command: `/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt6/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt6_baseline_run3.json`
- Environment: `AMD EPYC 9354P 32-Core Processor`; nproc=4; flags=avx2 avx512f bmi2 popcnt sse4_2; governor=unknown
- Engine threads: `1`; CPU utilization: `100.0%` of one core (~25.0% of the 4-core machine)
- Real model: `/home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts`; layers=36; hidden=4096; vocab=151936

## Reproducibility

```bash
# build
cmake -S . -B /tmp/opencode/lbit-p9 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/lbit-p9 -j4 --target perf_benchmark
# profile
/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt6/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt6_baseline_run3.json
```

The benchmark replays `Qwen3Model::forward_token` with timers and checks its logits against the engine on every replayed call: max abs diff = `0.0`. No engine source was changed for the baseline.

## End-to-end timing

| Metric | Value |
|---|---:|
| Model load | 881.175 ms |
| Prompt tokenization | 0.132 ms |
| Prefill (11 tokens) | 7845.563 ms |
| TTFT (tokenize + prefill) | 7845.695 ms |
| Decode (3 tokens) | 2219.848 ms |
| Decode per token | 739.949 ms |
| RSS after profiling | 2326 MB |

## Top 3 bottlenecks (measured, not assumed)

| Rank | Component | Total replay ms | ms / forward | Share |
|---:|---|---:|---:|---:|
| 1 | LittleBit binary projections (all 7 per layer) | 5506.9 | 393.348 | 54.5% |
| 2 | lm_head (full vocab dot product) | 4514.7 | 322.480 | 44.7% |
| 3 | Top-p sampling (sort 151936 logits) | 35.2 | 2.512 | 0.3% |

## Full component breakdown

| Component | ms / forward | Share |
|---|---:|---:|
| LittleBit binary projections (all 7 per layer) | 393.348 | 54.5% |
| lm_head (full vocab dot product) | 322.480 | 44.7% |
| Top-p sampling (sort 151936 logits) | 2.512 | 0.3% |
| SwiGLU activation | 1.660 | 0.2% |
| Self-attention core (scores + softmax + V) | 1.024 | 0.1% |
| Input + post + final RMSNorm | 0.409 | 0.1% |
| QK RMSNorm | 0.222 | 0.0% |
| Residual adds | 0.145 | 0.0% |
| KV cache write | 0.077 | 0.0% |
| Rotary embedding | 0.073 | 0.0% |
| Greedy sampling | 0.023 | 0.0% |
| Embedding lookup | 0.002 | 0.0% |

## Per-layer cost

- Mean per layer per forward: `11.035 ms`
- Slowest 5 layers (total over 14 replayed forwards): L31=162.2 ms, L32=160.3 ms, L28=158.3 ms, L30=157.9 ms, L29=156.9 ms
- Total transformer time by forward index (prefill then decode): `414.0, 395.5, 381.3, 385.1, 383.9, 385.7, 382.0, 393.9, 398.7, 387.3, 393.8, 400.3, 406.8, 453.6` ms

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

- Projection effective bandwidth: `1.247 GB/s`
- `lm_head` bytes per call: `1.159 GiB`
- `lm_head` effective bandwidth (in-forward): `3.860 GB/s`
- `lm_head` isolated bandwidth (3 repeats): `3.736 GB/s`
- KV data read by attention core: `29.5 MiB` over 14 forwards

## Interpretation

The numbers above are the baseline before any optimization. Top-3 proposals will be written in `docs/phase9-profiling-baseline.md` after review; this generated file contains raw measurements only.
