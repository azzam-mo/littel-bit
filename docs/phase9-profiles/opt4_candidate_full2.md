# Phase 9 Profiling — opt4_candidate_full2

- Commit: `9605266`
- Generated (UTC): `2026-11-23T11:45:18Z`
- Command: `/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt4/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt4_candidate_full2.json`
- Environment: `AMD EPYC 9354P 32-Core Processor`; nproc=4; flags=avx2 avx512f bmi2 popcnt sse4_2; governor=unknown
- Engine threads: `1`; CPU utilization: `98.9%` of one core (~24.7% of the 4-core machine)
- Real model: `/home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts`; layers=36; hidden=4096; vocab=151936

## Reproducibility

```bash
# build
cmake -S . -B /tmp/opencode/lbit-p9 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/lbit-p9 -j4 --target perf_benchmark
# profile
/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt4/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt4_candidate_full2.json
```

The benchmark replays `Qwen3Model::forward_token` with timers and checks its logits against the engine on every replayed call: max abs diff = `0.0`. No engine source was changed for the baseline.

## End-to-end timing

| Metric | Value |
|---|---:|
| Model load | 888.277 ms |
| Prompt tokenization | 0.142 ms |
| Prefill (11 tokens) | 12051.571 ms |
| TTFT (tokenize + prefill) | 12051.713 ms |
| Decode (3 tokens) | 3190.262 ms |
| Decode per token | 1063.421 ms |
| RSS after profiling | 2328 MB |

## Top 3 bottlenecks (measured, not assumed)

| Rank | Component | Total replay ms | ms / forward | Share |
|---:|---|---:|---:|---:|
| 1 | LittleBit binary projections (all 7 per layer) | 10839.7 | 774.263 | 65.3% |
| 2 | lm_head (full vocab dot product) | 5672.2 | 405.157 | 34.1% |
| 3 | Top-p sampling (sort 151936 logits) | 38.2 | 2.726 | 0.2% |

## Full component breakdown

| Component | ms / forward | Share |
|---|---:|---:|
| LittleBit binary projections (all 7 per layer) | 774.263 | 65.3% |
| lm_head (full vocab dot product) | 405.157 | 34.1% |
| Top-p sampling (sort 151936 logits) | 2.726 | 0.2% |
| SwiGLU activation | 1.833 | 0.2% |
| Self-attention core (scores + softmax + V) | 1.262 | 0.1% |
| Input + post + final RMSNorm | 0.428 | 0.0% |
| QK RMSNorm | 0.374 | 0.0% |
| Residual adds | 0.169 | 0.0% |
| Rotary embedding | 0.097 | 0.0% |
| KV cache write | 0.090 | 0.0% |
| Greedy sampling | 0.024 | 0.0% |
| Embedding lookup | 0.002 | 0.0% |

## Per-layer cost

- Mean per layer per forward: `21.648 ms`
- Slowest 5 layers (total over 14 replayed forwards): L20=493.9 ms, L21=431.6 ms, L16=415.7 ms, L26=366.7 ms, L28=362.6 ms
- Total transformer time by forward index (prefill then decode): `683.5, 689.0, 670.8, 677.3, 674.0, 676.8, 692.0, 1604.0, 1057.2, 725.1, 689.3, 695.1, 691.1, 685.3` ms

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

- Projection effective bandwidth: `0.633 GB/s`
- `lm_head` bytes per call: `1.159 GiB`
- `lm_head` effective bandwidth (in-forward): `3.072 GB/s`
- `lm_head` isolated bandwidth (3 repeats): `3.835 GB/s`
- KV data read by attention core: `29.5 MiB` over 14 forwards

## Interpretation

The numbers above are the baseline before any optimization. Top-3 proposals will be written in `docs/phase9-profiling-baseline.md` after review; this generated file contains raw measurements only.
