# Phase 9 Profiling — opt5_candidate_g4_5

- Commit: `a056a5d`
- Generated (UTC): `2026-11-24T03:11:32Z`
- Command: `/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt4/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt5_candidate_g4_5.json`
- Environment: `AMD EPYC 9354P 32-Core Processor`; nproc=4; flags=avx2 avx512f bmi2 popcnt sse4_2; governor=unknown
- Engine threads: `1`; CPU utilization: `99.9%` of one core (~25.0% of the 4-core machine)
- Real model: `/home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts`; layers=36; hidden=4096; vocab=151936

## Reproducibility

```bash
# build
cmake -S . -B /tmp/opencode/lbit-p9 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/lbit-p9 -j4 --target perf_benchmark
# profile
/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt4/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt5_candidate_g4_5.json
```

The benchmark replays `Qwen3Model::forward_token` with timers and checks its logits against the engine on every replayed call: max abs diff = `0.0`. No engine source was changed for the baseline.

## End-to-end timing

| Metric | Value |
|---|---:|
| Model load | 940.958 ms |
| Prompt tokenization | 0.145 ms |
| Prefill (11 tokens) | 8055.244 ms |
| TTFT (tokenize + prefill) | 8055.389 ms |
| Decode (3 tokens) | 2227.066 ms |
| Decode per token | 742.355 ms |
| RSS after profiling | 2326 MB |

## Top 3 bottlenecks (measured, not assumed)

| Rank | Component | Total replay ms | ms / forward | Share |
|---:|---|---:|---:|---:|
| 1 | LittleBit binary projections (all 7 per layer) | 5576.5 | 398.321 | 53.9% |
| 2 | lm_head (full vocab dot product) | 4680.9 | 334.352 | 45.2% |
| 3 | Top-p sampling (sort 151936 logits) | 35.4 | 2.530 | 0.3% |

## Full component breakdown

| Component | ms / forward | Share |
|---|---:|---:|
| LittleBit binary projections (all 7 per layer) | 398.321 | 53.9% |
| lm_head (full vocab dot product) | 334.352 | 45.2% |
| Top-p sampling (sort 151936 logits) | 2.530 | 0.3% |
| SwiGLU activation | 1.706 | 0.2% |
| Self-attention core (scores + softmax + V) | 1.039 | 0.1% |
| Input + post + final RMSNorm | 0.419 | 0.1% |
| QK RMSNorm | 0.221 | 0.0% |
| Residual adds | 0.149 | 0.0% |
| KV cache write | 0.082 | 0.0% |
| Rotary embedding | 0.073 | 0.0% |
| Greedy sampling | 0.026 | 0.0% |
| Embedding lookup | 0.003 | 0.0% |

## Per-layer cost

- Mean per layer per forward: `11.176 ms`
- Slowest 5 layers (total over 14 replayed forwards): L30=160.2 ms, L21=160.1 ms, L19=159.3 ms, L33=158.7 ms, L18=158.6 ms
- Total transformer time by forward index (prefill then decode): `445.6, 392.6, 396.4, 395.4, 408.8, 395.2, 382.4, 416.1, 380.5, 390.0, 396.6, 387.2, 391.5, 454.5` ms

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

- Projection effective bandwidth: `1.231 GB/s`
- `lm_head` bytes per call: `1.159 GiB`
- `lm_head` effective bandwidth (in-forward): `3.723 GB/s`
- `lm_head` isolated bandwidth (3 repeats): `3.388 GB/s`
- KV data read by attention core: `29.5 MiB` over 14 forwards

## Interpretation

The numbers above are the baseline before any optimization. Top-3 proposals will be written in `docs/phase9-profiling-baseline.md` after review; this generated file contains raw measurements only.
