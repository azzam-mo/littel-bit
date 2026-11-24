# Phase 9 Profiling — opt4_base_full1

- Commit: `9605266`
- Generated (UTC): `2026-11-23T11:42:56Z`
- Command: `/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt4-base/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt4_base_full1.json`
- Environment: `AMD EPYC 9354P 32-Core Processor`; nproc=4; flags=avx2 avx512f bmi2 popcnt sse4_2; governor=unknown
- Engine threads: `1`; CPU utilization: `99.7%` of one core (~24.9% of the 4-core machine)
- Real model: `/home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts`; layers=36; hidden=4096; vocab=151936

## Reproducibility

```bash
# build
cmake -S . -B /tmp/opencode/lbit-p9 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/lbit-p9 -j4 --target perf_benchmark
# profile
/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt4-base/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt4_base_full1.json
```

The benchmark replays `Qwen3Model::forward_token` with timers and checks its logits against the engine on every replayed call: max abs diff = `0.0`. No engine source was changed for the baseline.

## End-to-end timing

| Metric | Value |
|---|---:|
| Model load | 1186.650 ms |
| Prompt tokenization | 0.179 ms |
| Prefill (11 tokens) | 13083.953 ms |
| TTFT (tokenize + prefill) | 13084.132 ms |
| Decode (3 tokens) | 3543.343 ms |
| Decode per token | 1181.114 ms |
| RSS after profiling | 2328 MB |

## Top 3 bottlenecks (measured, not assumed)

| Rank | Component | Total replay ms | ms / forward | Share |
|---:|---|---:|---:|---:|
| 1 | LittleBit binary projections (all 7 per layer) | 9378.8 | 669.911 | 56.4% |
| 2 | lm_head (full vocab dot product) | 7164.7 | 511.761 | 43.1% |
| 3 | Top-p sampling (sort 151936 logits) | 35.3 | 2.521 | 0.2% |

## Full component breakdown

| Component | ms / forward | Share |
|---|---:|---:|
| LittleBit binary projections (all 7 per layer) | 669.911 | 56.4% |
| lm_head (full vocab dot product) | 511.761 | 43.1% |
| Top-p sampling (sort 151936 logits) | 2.521 | 0.2% |
| SwiGLU activation | 1.721 | 0.1% |
| Self-attention core (scores + softmax + V) | 1.044 | 0.1% |
| Input + post + final RMSNorm | 0.419 | 0.0% |
| QK RMSNorm | 0.232 | 0.0% |
| Residual adds | 0.155 | 0.0% |
| KV cache write | 0.084 | 0.0% |
| Rotary embedding | 0.077 | 0.0% |
| Greedy sampling | 0.025 | 0.0% |
| Embedding lookup | 0.002 | 0.0% |

## Per-layer cost

- Mean per layer per forward: `18.724 ms`
- Slowest 5 layers (total over 14 replayed forwards): L21=272.7 ms, L1=270.3 ms, L3=269.2 ms, L0=268.8 ms, L2=265.5 ms
- Total transformer time by forward index (prefill then decode): `683.6, 664.9, 680.7, 673.8, 684.3, 689.2, 676.2, 670.4, 663.9, 676.1, 660.4, 674.0, 677.1, 662.3` ms

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

- Projection effective bandwidth: `0.732 GB/s`
- `lm_head` bytes per call: `1.159 GiB`
- `lm_head` effective bandwidth (in-forward): `2.432 GB/s`
- `lm_head` isolated bandwidth (3 repeats): `2.438 GB/s`
- KV data read by attention core: `29.5 MiB` over 14 forwards

## Interpretation

The numbers above are the baseline before any optimization. Top-3 proposals will be written in `docs/phase9-profiling-baseline.md` after review; this generated file contains raw measurements only.
