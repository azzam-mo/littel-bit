# Phase 9 Profiling — opt6_baseline_run6

- Commit: `1f3ad54`
- Generated (UTC): `2026-11-24T04:12:41Z`
- Command: `/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt6/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt6_baseline_run6.json`
- Environment: `AMD EPYC 9354P 32-Core Processor`; nproc=4; flags=avx2 avx512f bmi2 popcnt sse4_2; governor=unknown
- Engine threads: `1`; CPU utilization: `99.5%` of one core (~24.9% of the 4-core machine)
- Real model: `/home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts`; layers=36; hidden=4096; vocab=151936

## Reproducibility

```bash
# build
cmake -S . -B /tmp/opencode/lbit-p9 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/lbit-p9 -j4 --target perf_benchmark
# profile
/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt6/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt6_baseline_run6.json
```

The benchmark replays `Qwen3Model::forward_token` with timers and checks its logits against the engine on every replayed call: max abs diff = `0.0`. No engine source was changed for the baseline.

## End-to-end timing

| Metric | Value |
|---|---:|
| Model load | 917.371 ms |
| Prompt tokenization | 0.141 ms |
| Prefill (11 tokens) | 8075.814 ms |
| TTFT (tokenize + prefill) | 8075.955 ms |
| Decode (3 tokens) | 2161.139 ms |
| Decode per token | 720.380 ms |
| RSS after profiling | 2326 MB |

## Top 3 bottlenecks (measured, not assumed)

| Rank | Component | Total replay ms | ms / forward | Share |
|---:|---|---:|---:|---:|
| 1 | LittleBit binary projections (all 7 per layer) | 5629.6 | 402.117 | 55.2% |
| 2 | lm_head (full vocab dot product) | 4484.3 | 320.309 | 44.0% |
| 3 | Top-p sampling (sort 151936 logits) | 35.2 | 2.516 | 0.3% |

## Full component breakdown

| Component | ms / forward | Share |
|---|---:|---:|
| LittleBit binary projections (all 7 per layer) | 402.117 | 55.2% |
| lm_head (full vocab dot product) | 320.309 | 44.0% |
| Top-p sampling (sort 151936 logits) | 2.516 | 0.3% |
| SwiGLU activation | 1.723 | 0.2% |
| Self-attention core (scores + softmax + V) | 1.026 | 0.1% |
| Input + post + final RMSNorm | 0.445 | 0.1% |
| QK RMSNorm | 0.225 | 0.0% |
| Residual adds | 0.143 | 0.0% |
| KV cache write | 0.072 | 0.0% |
| Rotary embedding | 0.071 | 0.0% |
| Greedy sampling | 0.027 | 0.0% |
| Embedding lookup | 0.003 | 0.0% |

## Per-layer cost

- Mean per layer per forward: `11.282 ms`
- Slowest 5 layers (total over 14 replayed forwards): L17=170.4 ms, L11=166.2 ms, L15=165.5 ms, L1=165.4 ms, L14=164.9 ms
- Total transformer time by forward index (prefill then decode): `467.7, 436.6, 434.6, 388.7, 389.3, 393.5, 389.2, 385.2, 407.7, 413.3, 381.9, 394.8, 397.1, 406.5` ms

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

- Projection effective bandwidth: `1.219 GB/s`
- `lm_head` bytes per call: `1.159 GiB`
- `lm_head` effective bandwidth (in-forward): `3.886 GB/s`
- `lm_head` isolated bandwidth (3 repeats): `3.818 GB/s`
- KV data read by attention core: `29.5 MiB` over 14 forwards

## Interpretation

The numbers above are the baseline before any optimization. Top-3 proposals will be written in `docs/phase9-profiling-baseline.md` after review; this generated file contains raw measurements only.
