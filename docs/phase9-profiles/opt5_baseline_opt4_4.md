# Phase 9 Profiling — opt5_baseline_opt4_4

- Commit: `a056a5d`
- Generated (UTC): `2026-11-24T03:09:35Z`
- Command: `/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt4-baseline/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt5_baseline_opt4_4.json`
- Environment: `AMD EPYC 9354P 32-Core Processor`; nproc=4; flags=avx2 avx512f bmi2 popcnt sse4_2; governor=unknown
- Engine threads: `1`; CPU utilization: `99.8%` of one core (~24.9% of the 4-core machine)
- Real model: `/home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts`; layers=36; hidden=4096; vocab=151936

## Reproducibility

```bash
# build
cmake -S . -B /tmp/opencode/lbit-p9 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/lbit-p9 -j4 --target perf_benchmark
# profile
/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt4-baseline/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt5_baseline_opt4_4.json
```

The benchmark replays `Qwen3Model::forward_token` with timers and checks its logits against the engine on every replayed call: max abs diff = `0.0`. No engine source was changed for the baseline.

## End-to-end timing

| Metric | Value |
|---|---:|
| Model load | 1832.957 ms |
| Prompt tokenization | 0.144 ms |
| Prefill (11 tokens) | 11281.004 ms |
| TTFT (tokenize + prefill) | 11281.147 ms |
| Decode (3 tokens) | 3104.179 ms |
| Decode per token | 1034.726 ms |
| RSS after profiling | 2326 MB |

## Top 3 bottlenecks (measured, not assumed)

| Rank | Component | Total replay ms | ms / forward | Share |
|---:|---|---:|---:|---:|
| 1 | LittleBit binary projections (all 7 per layer) | 9540.3 | 681.453 | 65.9% |
| 2 | lm_head (full vocab dot product) | 4849.9 | 346.425 | 33.5% |
| 3 | Top-p sampling (sort 151936 logits) | 35.5 | 2.535 | 0.2% |

## Full component breakdown

| Component | ms / forward | Share |
|---|---:|---:|
| LittleBit binary projections (all 7 per layer) | 681.453 | 65.9% |
| lm_head (full vocab dot product) | 346.425 | 33.5% |
| Top-p sampling (sort 151936 logits) | 2.535 | 0.2% |
| SwiGLU activation | 1.783 | 0.2% |
| Self-attention core (scores + softmax + V) | 1.087 | 0.1% |
| Input + post + final RMSNorm | 0.427 | 0.0% |
| QK RMSNorm | 0.232 | 0.0% |
| Residual adds | 0.163 | 0.0% |
| KV cache write | 0.085 | 0.0% |
| Rotary embedding | 0.074 | 0.0% |
| Greedy sampling | 0.025 | 0.0% |
| Embedding lookup | 0.003 | 0.0% |

## Per-layer cost

- Mean per layer per forward: `19.048 ms`
- Slowest 5 layers (total over 14 replayed forwards): L20=275.4 ms, L1=275.1 ms, L18=274.5 ms, L0=272.6 ms, L25=271.0 ms
- Total transformer time by forward index (prefill then decode): `689.7, 669.3, 680.1, 692.5, 688.9, 686.7, 695.8, 683.0, 690.9, 667.8, 697.1, 687.1, 693.1, 678.0` ms

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

- Projection effective bandwidth: `0.720 GB/s`
- `lm_head` bytes per call: `1.159 GiB`
- `lm_head` effective bandwidth (in-forward): `3.593 GB/s`
- `lm_head` isolated bandwidth (3 repeats): `3.697 GB/s`
- KV data read by attention core: `29.5 MiB` over 14 forwards

## Interpretation

The numbers above are the baseline before any optimization. Top-3 proposals will be written in `docs/phase9-profiling-baseline.md` after review; this generated file contains raw measurements only.
