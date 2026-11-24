# Phase 9 Profiling — opt6_candidate_run6

- Commit: `1f3ad54`
- Generated (UTC): `2026-11-24T04:16:47Z`
- Command: `/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt6/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt6_candidate_run6.json`
- Environment: `AMD EPYC 9354P 32-Core Processor`; nproc=4; flags=avx2 avx512f bmi2 popcnt sse4_2; governor=unknown
- Engine threads: `1`; CPU utilization: `99.6%` of one core (~24.9% of the 4-core machine)
- Real model: `/home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts`; layers=36; hidden=4096; vocab=151936

## Reproducibility

```bash
# build
cmake -S . -B /tmp/opencode/lbit-p9 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/lbit-p9 -j4 --target perf_benchmark
# profile
/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt6/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt6_candidate_run6.json
```

The benchmark replays `Qwen3Model::forward_token` with timers and checks its logits against the engine on every replayed call: max abs diff = `0.0`. No engine source was changed for the baseline.

## End-to-end timing

| Metric | Value |
|---|---:|
| Model load | 933.737 ms |
| Prompt tokenization | 0.138 ms |
| Prefill (11 tokens) | 5831.334 ms |
| TTFT (tokenize + prefill) | 5831.472 ms |
| Decode (3 tokens) | 1681.867 ms |
| Decode per token | 560.622 ms |
| RSS after profiling | 2326 MB |

## Top 3 bottlenecks (measured, not assumed)

| Rank | Component | Total replay ms | ms / forward | Share |
|---:|---|---:|---:|---:|
| 1 | LittleBit binary projections (all 7 per layer) | 5897.1 | 421.224 | 79.8% |
| 2 | lm_head (full vocab dot product) | 1404.9 | 100.349 | 19.0% |
| 3 | Top-p sampling (sort 151936 logits) | 36.7 | 2.623 | 0.5% |

## Full component breakdown

| Component | ms / forward | Share |
|---|---:|---:|
| LittleBit binary projections (all 7 per layer) | 421.224 | 79.8% |
| lm_head (full vocab dot product) | 100.349 | 19.0% |
| Top-p sampling (sort 151936 logits) | 2.623 | 0.5% |
| SwiGLU activation | 1.813 | 0.3% |
| Self-attention core (scores + softmax + V) | 1.114 | 0.2% |
| Input + post + final RMSNorm | 0.425 | 0.1% |
| QK RMSNorm | 0.252 | 0.0% |
| Residual adds | 0.154 | 0.0% |
| KV cache write | 0.082 | 0.0% |
| Rotary embedding | 0.075 | 0.0% |
| Greedy sampling | 0.024 | 0.0% |
| Embedding lookup | 0.003 | 0.0% |

## Per-layer cost

- Mean per layer per forward: `11.825 ms`
- Slowest 5 layers (total over 14 replayed forwards): L1=186.2 ms, L0=177.3 ms, L5=176.1 ms, L4=175.0 ms, L2=173.3 ms
- Total transformer time by forward index (prefill then decode): `408.4, 431.5, 422.5, 426.4, 414.7, 422.7, 400.7, 412.0, 405.2, 404.0, 428.9, 422.6, 437.4, 523.1` ms

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

- Projection effective bandwidth: `1.164 GB/s`
- `lm_head` bytes per call: `1.159 GiB`
- `lm_head` effective bandwidth (in-forward): `12.403 GB/s`
- `lm_head` isolated bandwidth (3 repeats): `12.191 GB/s`
- KV data read by attention core: `29.5 MiB` over 14 forwards

## Interpretation

The numbers above are the baseline before any optimization. Top-3 proposals will be written in `docs/phase9-profiling-baseline.md` after review; this generated file contains raw measurements only.
