# Phase 9 Profiling — opt4_base_full2

- Commit: `9605266`
- Generated (UTC): `2026-11-23T11:43:34Z`
- Command: `/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt4-base/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt4_base_full2.json`
- Environment: `AMD EPYC 9354P 32-Core Processor`; nproc=4; flags=avx2 avx512f bmi2 popcnt sse4_2; governor=unknown
- Engine threads: `1`; CPU utilization: `99.8%` of one core (~24.9% of the 4-core machine)
- Real model: `/home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts`; layers=36; hidden=4096; vocab=151936

## Reproducibility

```bash
# build
cmake -S . -B /tmp/opencode/lbit-p9 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/lbit-p9 -j4 --target perf_benchmark
# profile
/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt4-base/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt4_base_full2.json
```

The benchmark replays `Qwen3Model::forward_token` with timers and checks its logits against the engine on every replayed call: max abs diff = `0.0`. No engine source was changed for the baseline.

## End-to-end timing

| Metric | Value |
|---|---:|
| Model load | 909.525 ms |
| Prompt tokenization | 0.153 ms |
| Prefill (11 tokens) | 13727.976 ms |
| TTFT (tokenize + prefill) | 13728.129 ms |
| Decode (3 tokens) | 3561.875 ms |
| Decode per token | 1187.292 ms |
| RSS after profiling | 2328 MB |

## Top 3 bottlenecks (measured, not assumed)

| Rank | Component | Total replay ms | ms / forward | Share |
|---:|---|---:|---:|---:|
| 1 | LittleBit binary projections (all 7 per layer) | 9564.7 | 683.190 | 56.6% |
| 2 | lm_head (full vocab dot product) | 7252.0 | 517.998 | 42.9% |
| 3 | Top-p sampling (sort 151936 logits) | 35.1 | 2.504 | 0.2% |

## Full component breakdown

| Component | ms / forward | Share |
|---|---:|---:|
| LittleBit binary projections (all 7 per layer) | 683.190 | 56.6% |
| lm_head (full vocab dot product) | 517.998 | 42.9% |
| Top-p sampling (sort 151936 logits) | 2.504 | 0.2% |
| SwiGLU activation | 2.004 | 0.2% |
| Self-attention core (scores + softmax + V) | 1.074 | 0.1% |
| Input + post + final RMSNorm | 0.425 | 0.0% |
| QK RMSNorm | 0.228 | 0.0% |
| Residual adds | 0.161 | 0.0% |
| KV cache write | 0.095 | 0.0% |
| Rotary embedding | 0.081 | 0.0% |
| Greedy sampling | 0.025 | 0.0% |
| Embedding lookup | 0.003 | 0.0% |

## Per-layer cost

- Mean per layer per forward: `19.103 ms`
- Slowest 5 layers (total over 14 replayed forwards): L1=283.3 ms, L31=281.4 ms, L0=277.9 ms, L2=276.9 ms, L30=274.9 ms
- Total transformer time by forward index (prefill then decode): `742.1, 736.4, 668.1, 713.0, 678.5, 681.3, 668.2, 671.3, 692.9, 677.0, 672.6, 673.3, 674.0, 679.4` ms

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

- Projection effective bandwidth: `0.718 GB/s`
- `lm_head` bytes per call: `1.159 GiB`
- `lm_head` effective bandwidth (in-forward): `2.403 GB/s`
- `lm_head` isolated bandwidth (3 repeats): `2.445 GB/s`
- KV data read by attention core: `29.5 MiB` over 14 forwards

## Interpretation

The numbers above are the baseline before any optimization. Top-3 proposals will be written in `docs/phase9-profiling-baseline.md` after review; this generated file contains raw measurements only.
