# Phase 9 Profiling — opt6_interleave_b1

- Commit: `1f3ad54`
- Generated (UTC): `2026-11-24T04:17:53Z`
- Command: `/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt4/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt6_interleave_b1.json`
- Environment: `AMD EPYC 9354P 32-Core Processor`; nproc=4; flags=avx2 avx512f bmi2 popcnt sse4_2; governor=unknown
- Engine threads: `1`; CPU utilization: `99.3%` of one core (~24.8% of the 4-core machine)
- Real model: `/home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts`; layers=36; hidden=4096; vocab=151936

## Reproducibility

```bash
# build
cmake -S . -B /tmp/opencode/lbit-p9 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/lbit-p9 -j4 --target perf_benchmark
# profile
/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt4/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt6_interleave_b1.json
```

The benchmark replays `Qwen3Model::forward_token` with timers and checks its logits against the engine on every replayed call: max abs diff = `0.0`. No engine source was changed for the baseline.

## End-to-end timing

| Metric | Value |
|---|---:|
| Model load | 923.983 ms |
| Prompt tokenization | 0.147 ms |
| Prefill (11 tokens) | 8599.003 ms |
| TTFT (tokenize + prefill) | 8599.150 ms |
| Decode (3 tokens) | 2231.680 ms |
| Decode per token | 743.893 ms |
| RSS after profiling | 2326 MB |

## Top 3 bottlenecks (measured, not assumed)

| Rank | Component | Total replay ms | ms / forward | Share |
|---:|---|---:|---:|---:|
| 1 | LittleBit binary projections (all 7 per layer) | 5782.9 | 413.066 | 53.7% |
| 2 | lm_head (full vocab dot product) | 4901.2 | 350.087 | 45.5% |
| 3 | Top-p sampling (sort 151936 logits) | 35.4 | 2.527 | 0.3% |

## Full component breakdown

| Component | ms / forward | Share |
|---|---:|---:|
| LittleBit binary projections (all 7 per layer) | 413.066 | 53.7% |
| lm_head (full vocab dot product) | 350.087 | 45.5% |
| Top-p sampling (sort 151936 logits) | 2.527 | 0.3% |
| SwiGLU activation | 1.776 | 0.2% |
| Self-attention core (scores + softmax + V) | 1.078 | 0.1% |
| Input + post + final RMSNorm | 0.419 | 0.1% |
| QK RMSNorm | 0.244 | 0.0% |
| Residual adds | 0.154 | 0.0% |
| KV cache write | 0.080 | 0.0% |
| Rotary embedding | 0.076 | 0.0% |
| Greedy sampling | 0.025 | 0.0% |
| Embedding lookup | 0.003 | 0.0% |

## Per-layer cost

- Mean per layer per forward: `11.592 ms`
- Slowest 5 layers (total over 14 replayed forwards): L3=172.3 ms, L6=171.5 ms, L17=167.5 ms, L2=166.8 ms, L22=165.1 ms
- Total transformer time by forward index (prefill then decode): `401.1, 437.1, 427.1, 430.2, 396.3, 414.2, 406.1, 409.2, 417.9, 434.7, 450.3, 400.5, 396.3, 421.3` ms

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

- Projection effective bandwidth: `1.187 GB/s`
- `lm_head` bytes per call: `1.159 GiB`
- `lm_head` effective bandwidth (in-forward): `3.555 GB/s`
- `lm_head` isolated bandwidth (3 repeats): `3.419 GB/s`
- KV data read by attention core: `29.5 MiB` over 14 forwards

## Interpretation

The numbers above are the baseline before any optimization. Top-3 proposals will be written in `docs/phase9-profiling-baseline.md` after review; this generated file contains raw measurements only.
