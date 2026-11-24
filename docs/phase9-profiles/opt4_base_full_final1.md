# Phase 9 Profiling — opt4_base_full_final1

- Commit: `9605266`
- Generated (UTC): `2026-11-23T11:53:13Z`
- Command: `/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt4-base/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt4_base_full_final1.json`
- Environment: `AMD EPYC 9354P 32-Core Processor`; nproc=4; flags=avx2 avx512f bmi2 popcnt sse4_2; governor=unknown
- Engine threads: `1`; CPU utilization: `99.3%` of one core (~24.8% of the 4-core machine)
- Real model: `/home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts`; layers=36; hidden=4096; vocab=151936

## Reproducibility

```bash
# build
cmake -S . -B /tmp/opencode/lbit-p9 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/lbit-p9 -j4 --target perf_benchmark
# profile
/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt4-base/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt4_base_full_final1.json
```

The benchmark replays `Qwen3Model::forward_token` with timers and checks its logits against the engine on every replayed call: max abs diff = `0.0`. No engine source was changed for the baseline.

## End-to-end timing

| Metric | Value |
|---|---:|
| Model load | 953.433 ms |
| Prompt tokenization | 0.163 ms |
| Prefill (11 tokens) | 14097.644 ms |
| TTFT (tokenize + prefill) | 14097.807 ms |
| Decode (3 tokens) | 3640.864 ms |
| Decode per token | 1213.621 ms |
| RSS after profiling | 2328 MB |

## Top 3 bottlenecks (measured, not assumed)

| Rank | Component | Total replay ms | ms / forward | Share |
|---:|---|---:|---:|---:|
| 1 | LittleBit binary projections (all 7 per layer) | 9726.7 | 694.767 | 56.5% |
| 2 | lm_head (full vocab dot product) | 7396.7 | 528.335 | 43.0% |
| 3 | Top-p sampling (sort 151936 logits) | 38.0 | 2.712 | 0.2% |

## Full component breakdown

| Component | ms / forward | Share |
|---|---:|---:|
| LittleBit binary projections (all 7 per layer) | 694.767 | 56.5% |
| lm_head (full vocab dot product) | 528.335 | 43.0% |
| Top-p sampling (sort 151936 logits) | 2.712 | 0.2% |
| SwiGLU activation | 1.977 | 0.2% |
| Self-attention core (scores + softmax + V) | 1.115 | 0.1% |
| Input + post + final RMSNorm | 0.444 | 0.0% |
| QK RMSNorm | 0.235 | 0.0% |
| Residual adds | 0.163 | 0.0% |
| KV cache write | 0.083 | 0.0% |
| Rotary embedding | 0.078 | 0.0% |
| Greedy sampling | 0.028 | 0.0% |
| Embedding lookup | 0.003 | 0.0% |

## Per-layer cost

- Mean per layer per forward: `19.426 ms`
- Slowest 5 layers (total over 14 replayed forwards): L3=306.8 ms, L2=300.4 ms, L5=295.0 ms, L0=288.8 ms, L6=277.5 ms
- Total transformer time by forward index (prefill then decode): `680.4, 682.6, 665.6, 687.0, 672.5, 688.4, 709.9, 687.9, 674.7, 693.3, 876.7, 691.4, 686.2, 694.3` ms

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

- Projection effective bandwidth: `0.706 GB/s`
- `lm_head` bytes per call: `1.159 GiB`
- `lm_head` effective bandwidth (in-forward): `2.356 GB/s`
- `lm_head` isolated bandwidth (3 repeats): `2.420 GB/s`
- KV data read by attention core: `29.5 MiB` over 14 forwards

## Interpretation

The numbers above are the baseline before any optimization. Top-3 proposals will be written in `docs/phase9-profiling-baseline.md` after review; this generated file contains raw measurements only.
