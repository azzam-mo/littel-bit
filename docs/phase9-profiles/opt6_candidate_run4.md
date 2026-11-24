# Phase 9 Profiling — opt6_candidate_run4

- Commit: `1f3ad54`
- Generated (UTC): `2026-11-24T04:16:12Z`
- Command: `/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt6/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt6_candidate_run4.json`
- Environment: `AMD EPYC 9354P 32-Core Processor`; nproc=4; flags=avx2 avx512f bmi2 popcnt sse4_2; governor=unknown
- Engine threads: `1`; CPU utilization: `99.5%` of one core (~24.9% of the 4-core machine)
- Real model: `/home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts`; layers=36; hidden=4096; vocab=151936

## Reproducibility

```bash
# build
cmake -S . -B /tmp/opencode/lbit-p9 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/lbit-p9 -j4 --target perf_benchmark
# profile
/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt6/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt6_candidate_run4.json
```

The benchmark replays `Qwen3Model::forward_token` with timers and checks its logits against the engine on every replayed call: max abs diff = `0.0`. No engine source was changed for the baseline.

## End-to-end timing

| Metric | Value |
|---|---:|
| Model load | 974.559 ms |
| Prompt tokenization | 0.142 ms |
| Prefill (11 tokens) | 7026.054 ms |
| TTFT (tokenize + prefill) | 7026.196 ms |
| Decode (3 tokens) | 1566.017 ms |
| Decode per token | 522.006 ms |
| RSS after profiling | 2326 MB |

## Top 3 bottlenecks (measured, not assumed)

| Rank | Component | Total replay ms | ms / forward | Share |
|---:|---|---:|---:|---:|
| 1 | LittleBit binary projections (all 7 per layer) | 7172.4 | 512.311 | 80.7% |
| 2 | lm_head (full vocab dot product) | 1610.3 | 115.023 | 18.1% |
| 3 | Top-p sampling (sort 151936 logits) | 37.1 | 2.650 | 0.4% |

## Full component breakdown

| Component | ms / forward | Share |
|---|---:|---:|
| LittleBit binary projections (all 7 per layer) | 512.311 | 80.7% |
| lm_head (full vocab dot product) | 115.023 | 18.1% |
| Top-p sampling (sort 151936 logits) | 2.650 | 0.4% |
| SwiGLU activation | 2.129 | 0.3% |
| Self-attention core (scores + softmax + V) | 1.522 | 0.2% |
| Input + post + final RMSNorm | 0.429 | 0.1% |
| QK RMSNorm | 0.373 | 0.1% |
| Residual adds | 0.194 | 0.0% |
| Rotary embedding | 0.081 | 0.0% |
| KV cache write | 0.080 | 0.0% |
| Greedy sampling | 0.025 | 0.0% |
| Embedding lookup | 0.003 | 0.0% |

## Per-layer cost

- Mean per layer per forward: `14.376 ms`
- Slowest 5 layers (total over 14 replayed forwards): L17=283.0 ms, L9=246.1 ms, L3=232.7 ms, L6=232.1 ms, L14=231.5 ms
- Total transformer time by forward index (prefill then decode): `474.4, 455.3, 418.0, 429.4, 538.8, 1030.5, 670.1, 466.1, 465.1, 615.6, 431.3, 414.1, 414.6, 422.2` ms

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

- Projection effective bandwidth: `0.957 GB/s`
- `lm_head` bytes per call: `1.159 GiB`
- `lm_head` effective bandwidth (in-forward): `10.821 GB/s`
- `lm_head` isolated bandwidth (3 repeats): `11.572 GB/s`
- KV data read by attention core: `29.5 MiB` over 14 forwards

## Interpretation

The numbers above are the baseline before any optimization. Top-3 proposals will be written in `docs/phase9-profiling-baseline.md` after review; this generated file contains raw measurements only.
