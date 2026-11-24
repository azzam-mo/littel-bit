# Phase 9 Profiling — opt6_interleave_c2

- Commit: `1f3ad54`
- Generated (UTC): `2026-11-24T04:18:53Z`
- Command: `/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt6/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt6_interleave_c2.json`
- Environment: `AMD EPYC 9354P 32-Core Processor`; nproc=4; flags=avx2 avx512f bmi2 popcnt sse4_2; governor=unknown
- Engine threads: `1`; CPU utilization: `99.9%` of one core (~25.0% of the 4-core machine)
- Real model: `/home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts`; layers=36; hidden=4096; vocab=151936

## Reproducibility

```bash
# build
cmake -S . -B /tmp/opencode/lbit-p9 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/lbit-p9 -j4 --target perf_benchmark
# profile
/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt6/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt6_interleave_c2.json
```

The benchmark replays `Qwen3Model::forward_token` with timers and checks its logits against the engine on every replayed call: max abs diff = `0.0`. No engine source was changed for the baseline.

## End-to-end timing

| Metric | Value |
|---|---:|
| Model load | 980.967 ms |
| Prompt tokenization | 0.143 ms |
| Prefill (11 tokens) | 5749.888 ms |
| TTFT (tokenize + prefill) | 5750.031 ms |
| Decode (3 tokens) | 1475.303 ms |
| Decode per token | 491.768 ms |
| RSS after profiling | 2326 MB |

## Top 3 bottlenecks (measured, not assumed)

| Rank | Component | Total replay ms | ms / forward | Share |
|---:|---|---:|---:|---:|
| 1 | LittleBit binary projections (all 7 per layer) | 5665.1 | 404.653 | 80.0% |
| 2 | lm_head (full vocab dot product) | 1332.4 | 95.169 | 18.8% |
| 3 | Top-p sampling (sort 151936 logits) | 35.3 | 2.519 | 0.5% |

## Full component breakdown

| Component | ms / forward | Share |
|---|---:|---:|
| LittleBit binary projections (all 7 per layer) | 404.653 | 80.0% |
| lm_head (full vocab dot product) | 95.169 | 18.8% |
| Top-p sampling (sort 151936 logits) | 2.519 | 0.5% |
| SwiGLU activation | 1.722 | 0.3% |
| Self-attention core (scores + softmax + V) | 1.074 | 0.2% |
| Input + post + final RMSNorm | 0.400 | 0.1% |
| QK RMSNorm | 0.219 | 0.0% |
| Residual adds | 0.156 | 0.0% |
| KV cache write | 0.076 | 0.0% |
| Rotary embedding | 0.068 | 0.0% |
| Greedy sampling | 0.025 | 0.0% |
| Embedding lookup | 0.003 | 0.0% |

## Per-layer cost

- Mean per layer per forward: `11.353 ms`
- Slowest 5 layers (total over 14 replayed forwards): L13=168.1 ms, L1=167.8 ms, L5=164.0 ms, L0=163.8 ms, L3=162.8 ms
- Total transformer time by forward index (prefill then decode): `410.7, 409.3, 424.2, 432.0, 395.7, 385.4, 416.6, 414.1, 414.3, 398.7, 404.9, 394.1, 403.3, 418.8` ms

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

- Projection effective bandwidth: `1.212 GB/s`
- `lm_head` bytes per call: `1.159 GiB`
- `lm_head` effective bandwidth (in-forward): `13.078 GB/s`
- `lm_head` isolated bandwidth (3 repeats): `13.536 GB/s`
- KV data read by attention core: `29.5 MiB` over 14 forwards

## Interpretation

The numbers above are the baseline before any optimization. Top-3 proposals will be written in `docs/phase9-profiling-baseline.md` after review; this generated file contains raw measurements only.
