# Phase 9 Profiling — opt6_baseline_run5

- Commit: `1f3ad54`
- Generated (UTC): `2026-11-24T04:12:17Z`
- Command: `/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt6/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt6_baseline_run5.json`
- Environment: `AMD EPYC 9354P 32-Core Processor`; nproc=4; flags=avx2 avx512f bmi2 popcnt sse4_2; governor=unknown
- Engine threads: `1`; CPU utilization: `99.1%` of one core (~24.8% of the 4-core machine)
- Real model: `/home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts`; layers=36; hidden=4096; vocab=151936

## Reproducibility

```bash
# build
cmake -S . -B /tmp/opencode/lbit-p9 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/lbit-p9 -j4 --target perf_benchmark
# profile
/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt6/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt6_baseline_run5.json
```

The benchmark replays `Qwen3Model::forward_token` with timers and checks its logits against the engine on every replayed call: max abs diff = `0.0`. No engine source was changed for the baseline.

## End-to-end timing

| Metric | Value |
|---|---:|
| Model load | 967.335 ms |
| Prompt tokenization | 0.143 ms |
| Prefill (11 tokens) | 9437.868 ms |
| TTFT (tokenize + prefill) | 9438.012 ms |
| Decode (3 tokens) | 2285.130 ms |
| Decode per token | 761.710 ms |
| RSS after profiling | 2326 MB |

## Top 3 bottlenecks (measured, not assumed)

| Rank | Component | Total replay ms | ms / forward | Share |
|---:|---|---:|---:|---:|
| 1 | LittleBit binary projections (all 7 per layer) | 6167.1 | 440.507 | 53.1% |
| 2 | lm_head (full vocab dot product) | 5352.3 | 382.309 | 46.1% |
| 3 | Top-p sampling (sort 151936 logits) | 36.1 | 2.580 | 0.3% |

## Full component breakdown

| Component | ms / forward | Share |
|---|---:|---:|
| LittleBit binary projections (all 7 per layer) | 440.507 | 53.1% |
| lm_head (full vocab dot product) | 382.309 | 46.1% |
| Top-p sampling (sort 151936 logits) | 2.580 | 0.3% |
| SwiGLU activation | 1.853 | 0.2% |
| Self-attention core (scores + softmax + V) | 1.170 | 0.1% |
| Input + post + final RMSNorm | 0.446 | 0.1% |
| QK RMSNorm | 0.226 | 0.0% |
| Residual adds | 0.173 | 0.0% |
| KV cache write | 0.083 | 0.0% |
| Rotary embedding | 0.082 | 0.0% |
| Greedy sampling | 0.024 | 0.0% |
| Embedding lookup | 0.003 | 0.0% |

## Per-layer cost

- Mean per layer per forward: `12.360 ms`
- Slowest 5 layers (total over 14 replayed forwards): L6=202.7 ms, L8=202.7 ms, L5=200.0 ms, L18=188.3 ms, L29=182.5 ms
- Total transformer time by forward index (prefill then decode): `403.7, 462.5, 413.1, 428.6, 479.4, 622.8, 416.2, 431.9, 480.1, 422.0, 410.4, 421.1, 435.6, 402.0` ms

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

- Projection effective bandwidth: `1.113 GB/s`
- `lm_head` bytes per call: `1.159 GiB`
- `lm_head` effective bandwidth (in-forward): `3.256 GB/s`
- `lm_head` isolated bandwidth (3 repeats): `3.665 GB/s`
- KV data read by attention core: `29.5 MiB` over 14 forwards

## Interpretation

The numbers above are the baseline before any optimization. Top-3 proposals will be written in `docs/phase9-profiling-baseline.md` after review; this generated file contains raw measurements only.
