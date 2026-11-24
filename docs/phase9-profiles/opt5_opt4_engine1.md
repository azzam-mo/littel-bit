# Phase 9 Profiling — opt5_opt4_engine1

- Commit: `a056a5d`
- Generated (UTC): `2026-11-24T02:58:30Z`
- Command: `/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt4/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt5_opt4_engine1.json`
- Environment: `AMD EPYC 9354P 32-Core Processor`; nproc=4; flags=avx2 avx512f bmi2 popcnt sse4_2; governor=unknown
- Engine threads: `1`; CPU utilization: `99.2%` of one core (~24.8% of the 4-core machine)
- Real model: `/home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts`; layers=36; hidden=4096; vocab=151936

## Reproducibility

```bash
# build
cmake -S . -B /tmp/opencode/lbit-p9 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/lbit-p9 -j4 --target perf_benchmark
# profile
/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt4/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt5_opt4_engine1.json
```

The benchmark replays `Qwen3Model::forward_token` with timers and checks its logits against the engine on every replayed call: max abs diff = `0.0`. No engine source was changed for the baseline.

## End-to-end timing

| Metric | Value |
|---|---:|
| Model load | 1692.147 ms |
| Prompt tokenization | 0.172 ms |
| Prefill (11 tokens) | 12680.308 ms |
| TTFT (tokenize + prefill) | 12680.479 ms |
| Decode (3 tokens) | 3089.932 ms |
| Decode per token | 1029.977 ms |
| RSS after profiling | 2326 MB |

## Top 3 bottlenecks (measured, not assumed)

| Rank | Component | Total replay ms | ms / forward | Share |
|---:|---|---:|---:|---:|
| 1 | LittleBit binary projections (all 7 per layer) | 10216.8 | 729.774 | 67.3% |
| 2 | lm_head (full vocab dot product) | 4860.1 | 347.153 | 32.0% |
| 3 | Top-p sampling (sort 151936 logits) | 35.4 | 2.530 | 0.2% |

## Full component breakdown

| Component | ms / forward | Share |
|---|---:|---:|
| LittleBit binary projections (all 7 per layer) | 729.774 | 67.3% |
| lm_head (full vocab dot product) | 347.153 | 32.0% |
| Top-p sampling (sort 151936 logits) | 2.530 | 0.2% |
| SwiGLU activation | 1.905 | 0.2% |
| Self-attention core (scores + softmax + V) | 1.164 | 0.1% |
| Input + post + final RMSNorm | 0.454 | 0.0% |
| QK RMSNorm | 0.282 | 0.0% |
| Residual adds | 0.172 | 0.0% |
| KV cache write | 0.088 | 0.0% |
| Rotary embedding | 0.083 | 0.0% |
| Greedy sampling | 0.026 | 0.0% |
| Embedding lookup | 0.004 | 0.0% |

## Per-layer cost

- Mean per layer per forward: `20.400 ms`
- Slowest 5 layers (total over 14 replayed forwards): L8=303.9 ms, L10=303.1 ms, L22=297.8 ms, L5=297.4 ms, L6=294.7 ms
- Total transformer time by forward index (prefill then decode): `961.4, 715.9, 686.4, 679.9, 682.7, 714.3, 665.9, 698.2, 674.6, 699.0, 705.5, 677.7, 692.0, 1028.0` ms

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

- Projection effective bandwidth: `0.672 GB/s`
- `lm_head` bytes per call: `1.159 GiB`
- `lm_head` effective bandwidth (in-forward): `3.585 GB/s`
- `lm_head` isolated bandwidth (3 repeats): `3.842 GB/s`
- KV data read by attention core: `29.5 MiB` over 14 forwards

## Interpretation

The numbers above are the baseline before any optimization. Top-3 proposals will be written in `docs/phase9-profiling-baseline.md` after review; this generated file contains raw measurements only.
