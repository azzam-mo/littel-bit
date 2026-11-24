# Phase 9 Profiling — opt6_candidate_run1

- Commit: `1f3ad54`
- Generated (UTC): `2026-11-24T04:15:13Z`
- Command: `/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt6/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt6_candidate_run1.json`
- Environment: `AMD EPYC 9354P 32-Core Processor`; nproc=4; flags=avx2 avx512f bmi2 popcnt sse4_2; governor=unknown
- Engine threads: `1`; CPU utilization: `99.1%` of one core (~24.8% of the 4-core machine)
- Real model: `/home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts`; layers=36; hidden=4096; vocab=151936

## Reproducibility

```bash
# build
cmake -S . -B /tmp/opencode/lbit-p9 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/lbit-p9 -j4 --target perf_benchmark
# profile
/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt6/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt6_candidate_run1.json
```

The benchmark replays `Qwen3Model::forward_token` with timers and checks its logits against the engine on every replayed call: max abs diff = `0.0`. No engine source was changed for the baseline.

## End-to-end timing

| Metric | Value |
|---|---:|
| Model load | 868.688 ms |
| Prompt tokenization | 0.132 ms |
| Prefill (11 tokens) | 7002.382 ms |
| TTFT (tokenize + prefill) | 7002.514 ms |
| Decode (3 tokens) | 1568.273 ms |
| Decode per token | 522.758 ms |
| RSS after profiling | 2326 MB |

## Top 3 bottlenecks (measured, not assumed)

| Rank | Component | Total replay ms | ms / forward | Share |
|---:|---|---:|---:|---:|
| 1 | LittleBit binary projections (all 7 per layer) | 7041.7 | 502.981 | 80.2% |
| 2 | lm_head (full vocab dot product) | 1631.5 | 116.535 | 18.6% |
| 3 | Top-p sampling (sort 151936 logits) | 39.9 | 2.847 | 0.5% |

## Full component breakdown

| Component | ms / forward | Share |
|---|---:|---:|
| LittleBit binary projections (all 7 per layer) | 502.981 | 80.2% |
| lm_head (full vocab dot product) | 116.535 | 18.6% |
| Top-p sampling (sort 151936 logits) | 2.847 | 0.5% |
| SwiGLU activation | 2.171 | 0.3% |
| Self-attention core (scores + softmax + V) | 1.268 | 0.2% |
| Input + post + final RMSNorm | 0.459 | 0.1% |
| QK RMSNorm | 0.229 | 0.0% |
| Residual adds | 0.165 | 0.0% |
| Rotary embedding | 0.103 | 0.0% |
| KV cache write | 0.082 | 0.0% |
| Greedy sampling | 0.024 | 0.0% |
| Embedding lookup | 0.003 | 0.0% |

## Per-layer cost

- Mean per layer per forward: `14.108 ms`
- Slowest 5 layers (total over 14 replayed forwards): L18=281.0 ms, L20=258.0 ms, L17=245.8 ms, L16=242.2 ms, L34=234.0 ms
- Total transformer time by forward index (prefill then decode): `390.5, 387.8, 382.6, 395.6, 520.5, 797.7, 975.8, 454.4, 419.2, 565.5, 426.5, 482.0, 454.7, 457.5` ms

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

- Projection effective bandwidth: `0.975 GB/s`
- `lm_head` bytes per call: `1.159 GiB`
- `lm_head` effective bandwidth (in-forward): `10.681 GB/s`
- `lm_head` isolated bandwidth (3 repeats): `7.649 GB/s`
- KV data read by attention core: `29.5 MiB` over 14 forwards

## Interpretation

The numbers above are the baseline before any optimization. Top-3 proposals will be written in `docs/phase9-profiling-baseline.md` after review; this generated file contains raw measurements only.
