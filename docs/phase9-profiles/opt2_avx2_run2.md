# Phase 9 Profiling — opt2_avx2_run2

- Commit: `87c2048`
- Generated (UTC): `2026-11-23T10:17:44Z`
- Command: `/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt2/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt2_avx2_run2.json`
- Environment: `AMD EPYC 9354P 32-Core Processor`; nproc=4; flags=avx2 avx512f bmi2 popcnt sse4_2; governor=unknown
- Engine threads: `1`; CPU utilization: `99.8%` of one core (~25.0% of the 4-core machine)
- Real model: `/home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts`; layers=36; hidden=4096; vocab=151936

## Reproducibility

```bash
# build
cmake -S . -B /tmp/opencode/lbit-p9 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/lbit-p9 -j4 --target perf_benchmark
# profile
/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt2/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt2_avx2_run2.json
```

The benchmark replays `Qwen3Model::forward_token` with timers and checks its logits against the engine on every replayed call: max abs diff = `0.0`. No engine source was changed for the baseline.

## End-to-end timing

| Metric | Value |
|---|---:|
| Model load | 971.000 ms |
| Prompt tokenization | 0.139 ms |
| Prefill (11 tokens) | 13209.997 ms |
| TTFT (tokenize + prefill) | 13210.136 ms |
| Decode (3 tokens) | 3566.099 ms |
| Decode per token | 1188.700 ms |
| RSS after profiling | 2332 MB |

## Top 3 bottlenecks (measured, not assumed)

| Rank | Component | Total replay ms | ms / forward | Share |
|---:|---|---:|---:|---:|
| 1 | LittleBit binary projections (all 7 per layer) | 9513.1 | 679.508 | 56.7% |
| 2 | lm_head (full vocab dot product) | 7182.2 | 513.011 | 42.8% |
| 3 | Top-p sampling (sort 151936 logits) | 39.9 | 2.848 | 0.2% |

## Full component breakdown

| Component | ms / forward | Share |
|---|---:|---:|
| LittleBit binary projections (all 7 per layer) | 679.508 | 56.7% |
| lm_head (full vocab dot product) | 513.011 | 42.8% |
| Top-p sampling (sort 151936 logits) | 2.848 | 0.2% |
| SwiGLU activation | 1.791 | 0.1% |
| Self-attention core (scores + softmax + V) | 1.072 | 0.1% |
| Input + post + final RMSNorm | 0.425 | 0.0% |
| QK RMSNorm | 0.234 | 0.0% |
| Residual adds | 0.162 | 0.0% |
| KV cache write | 0.083 | 0.0% |
| Rotary embedding | 0.078 | 0.0% |
| Greedy sampling | 0.024 | 0.0% |
| Embedding lookup | 0.002 | 0.0% |

## Per-layer cost

- Mean per layer per forward: `18.994 ms`
- Slowest 5 layers (total over 14 replayed forwards): L3=272.1 ms, L8=271.8 ms, L33=271.0 ms, L24=269.6 ms, L7=269.3 ms
- Total transformer time by forward index (prefill then decode): `720.2, 670.6, 677.4, 687.8, 697.0, 667.7, 673.5, 677.5, 680.8, 679.9, 679.2, 695.3, 692.4, 673.7` ms

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

- Projection effective bandwidth: `0.722 GB/s`
- `lm_head` bytes per call: `1.159 GiB`
- `lm_head` effective bandwidth (in-forward): `2.426 GB/s`
- `lm_head` isolated bandwidth (3 repeats): `2.434 GB/s`
- KV data read by attention core: `29.5 MiB` over 14 forwards

## Interpretation

The numbers above are the baseline before any optimization. Top-3 proposals will be written in `docs/phase9-profiling-baseline.md` after review; this generated file contains raw measurements only.
