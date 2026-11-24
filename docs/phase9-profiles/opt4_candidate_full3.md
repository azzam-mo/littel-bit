# Phase 9 Profiling — opt4_candidate_full3

- Commit: `9605266`
- Generated (UTC): `2026-11-23T11:45:50Z`
- Command: `/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt4/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt4_candidate_full3.json`
- Environment: `AMD EPYC 9354P 32-Core Processor`; nproc=4; flags=avx2 avx512f bmi2 popcnt sse4_2; governor=unknown
- Engine threads: `1`; CPU utilization: `99.9%` of one core (~25.0% of the 4-core machine)
- Real model: `/home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts`; layers=36; hidden=4096; vocab=151936

## Reproducibility

```bash
# build
cmake -S . -B /tmp/opencode/lbit-p9 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/lbit-p9 -j4 --target perf_benchmark
# profile
/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt4/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt4_candidate_full3.json
```

The benchmark replays `Qwen3Model::forward_token` with timers and checks its logits against the engine on every replayed call: max abs diff = `0.0`. No engine source was changed for the baseline.

## End-to-end timing

| Metric | Value |
|---|---:|
| Model load | 1008.859 ms |
| Prompt tokenization | 0.168 ms |
| Prefill (11 tokens) | 11302.912 ms |
| TTFT (tokenize + prefill) | 11303.080 ms |
| Decode (3 tokens) | 2925.105 ms |
| Decode per token | 975.035 ms |
| RSS after profiling | 2328 MB |

## Top 3 bottlenecks (measured, not assumed)

| Rank | Component | Total replay ms | ms / forward | Share |
|---:|---|---:|---:|---:|
| 1 | LittleBit binary projections (all 7 per layer) | 9369.1 | 669.218 | 65.9% |
| 2 | lm_head (full vocab dot product) | 4754.2 | 339.587 | 33.5% |
| 3 | Top-p sampling (sort 151936 logits) | 35.0 | 2.504 | 0.2% |

## Full component breakdown

| Component | ms / forward | Share |
|---|---:|---:|
| LittleBit binary projections (all 7 per layer) | 669.218 | 65.9% |
| lm_head (full vocab dot product) | 339.587 | 33.5% |
| Top-p sampling (sort 151936 logits) | 2.504 | 0.2% |
| SwiGLU activation | 1.731 | 0.2% |
| Self-attention core (scores + softmax + V) | 1.037 | 0.1% |
| Input + post + final RMSNorm | 0.423 | 0.0% |
| QK RMSNorm | 0.231 | 0.0% |
| Residual adds | 0.156 | 0.0% |
| KV cache write | 0.082 | 0.0% |
| Rotary embedding | 0.078 | 0.0% |
| Greedy sampling | 0.024 | 0.0% |
| Embedding lookup | 0.002 | 0.0% |

## Per-layer cost

- Mean per layer per forward: `18.704 ms`
- Slowest 5 layers (total over 14 replayed forwards): L1=266.1 ms, L10=265.1 ms, L2=264.2 ms, L5=264.2 ms, L29=264.0 ms
- Total transformer time by forward index (prefill then decode): `680.3, 665.9, 668.5, 665.1, 702.0, 668.5, 676.0, 661.0, 717.6, 656.1, 670.1, 666.2, 665.7, 664.0` ms

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

- Projection effective bandwidth: `0.733 GB/s`
- `lm_head` bytes per call: `1.159 GiB`
- `lm_head` effective bandwidth (in-forward): `3.665 GB/s`
- `lm_head` isolated bandwidth (3 repeats): `4.165 GB/s`
- KV data read by attention core: `29.5 MiB` over 14 forwards

## Interpretation

The numbers above are the baseline before any optimization. Top-3 proposals will be written in `docs/phase9-profiling-baseline.md` after review; this generated file contains raw measurements only.
