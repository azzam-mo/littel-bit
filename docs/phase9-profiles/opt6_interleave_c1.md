# Phase 9 Profiling — opt6_interleave_c1

- Commit: `1f3ad54`
- Generated (UTC): `2026-11-24T04:18:11Z`
- Command: `/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt6/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt6_interleave_c1.json`
- Environment: `AMD EPYC 9354P 32-Core Processor`; nproc=4; flags=avx2 avx512f bmi2 popcnt sse4_2; governor=unknown
- Engine threads: `1`; CPU utilization: `99.4%` of one core (~24.9% of the 4-core machine)
- Real model: `/home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts`; layers=36; hidden=4096; vocab=151936

## Reproducibility

```bash
# build
cmake -S . -B /tmp/opencode/lbit-p9 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/lbit-p9 -j4 --target perf_benchmark
# profile
/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt6/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt6_interleave_c1.json
```

The benchmark replays `Qwen3Model::forward_token` with timers and checks its logits against the engine on every replayed call: max abs diff = `0.0`. No engine source was changed for the baseline.

## End-to-end timing

| Metric | Value |
|---|---:|
| Model load | 989.344 ms |
| Prompt tokenization | 0.141 ms |
| Prefill (11 tokens) | 6453.875 ms |
| TTFT (tokenize + prefill) | 6454.016 ms |
| Decode (3 tokens) | 1564.199 ms |
| Decode per token | 521.400 ms |
| RSS after profiling | 2326 MB |

## Top 3 bottlenecks (measured, not assumed)

| Rank | Component | Total replay ms | ms / forward | Share |
|---:|---|---:|---:|---:|
| 1 | LittleBit binary projections (all 7 per layer) | 6409.5 | 457.825 | 79.9% |
| 2 | lm_head (full vocab dot product) | 1516.2 | 108.303 | 18.9% |
| 3 | Top-p sampling (sort 151936 logits) | 35.8 | 2.556 | 0.4% |

## Full component breakdown

| Component | ms / forward | Share |
|---|---:|---:|
| LittleBit binary projections (all 7 per layer) | 457.825 | 79.9% |
| lm_head (full vocab dot product) | 108.303 | 18.9% |
| Top-p sampling (sort 151936 logits) | 2.556 | 0.4% |
| SwiGLU activation | 2.244 | 0.4% |
| Self-attention core (scores + softmax + V) | 1.123 | 0.2% |
| Input + post + final RMSNorm | 0.548 | 0.1% |
| QK RMSNorm | 0.231 | 0.0% |
| Residual adds | 0.167 | 0.0% |
| KV cache write | 0.077 | 0.0% |
| Rotary embedding | 0.077 | 0.0% |
| Greedy sampling | 0.024 | 0.0% |
| Embedding lookup | 0.003 | 0.0% |

## Per-layer cost

- Mean per layer per forward: `12.857 ms`
- Slowest 5 layers (total over 14 replayed forwards): L35=217.5 ms, L5=214.7 ms, L1=205.9 ms, L32=202.4 ms, L6=195.8 ms
- Total transformer time by forward index (prefill then decode): `422.6, 401.9, 413.2, 391.7, 462.1, 732.2, 520.8, 477.4, 419.9, 417.1, 544.3, 421.2, 428.5, 427.0` ms

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

- Projection effective bandwidth: `1.071 GB/s`
- `lm_head` bytes per call: `1.159 GiB`
- `lm_head` effective bandwidth (in-forward): `11.492 GB/s`
- `lm_head` isolated bandwidth (3 repeats): `12.722 GB/s`
- KV data read by attention core: `29.5 MiB` over 14 forwards

## Interpretation

The numbers above are the baseline before any optimization. Top-3 proposals will be written in `docs/phase9-profiling-baseline.md` after review; this generated file contains raw measurements only.
