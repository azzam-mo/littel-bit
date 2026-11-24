# Phase 9 Profiling — opt6_candidate_run2

- Commit: `1f3ad54`
- Generated (UTC): `2026-11-24T04:15:33Z`
- Command: `/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt6/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt6_candidate_run2.json`
- Environment: `AMD EPYC 9354P 32-Core Processor`; nproc=4; flags=avx2 avx512f bmi2 popcnt sse4_2; governor=unknown
- Engine threads: `1`; CPU utilization: `99.9%` of one core (~25.0% of the 4-core machine)
- Real model: `/home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts`; layers=36; hidden=4096; vocab=151936

## Reproducibility

```bash
# build
cmake -S . -B /tmp/opencode/lbit-p9 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/lbit-p9 -j4 --target perf_benchmark
# profile
/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt6/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt6_candidate_run2.json
```

The benchmark replays `Qwen3Model::forward_token` with timers and checks its logits against the engine on every replayed call: max abs diff = `0.0`. No engine source was changed for the baseline.

## End-to-end timing

| Metric | Value |
|---|---:|
| Model load | 1014.745 ms |
| Prompt tokenization | 0.151 ms |
| Prefill (11 tokens) | 6483.209 ms |
| TTFT (tokenize + prefill) | 6483.361 ms |
| Decode (3 tokens) | 1610.633 ms |
| Decode per token | 536.878 ms |
| RSS after profiling | 2326 MB |

## Top 3 bottlenecks (measured, not assumed)

| Rank | Component | Total replay ms | ms / forward | Share |
|---:|---|---:|---:|---:|
| 1 | LittleBit binary projections (all 7 per layer) | 6460.4 | 461.460 | 79.5% |
| 2 | lm_head (full vocab dot product) | 1570.5 | 112.181 | 19.3% |
| 3 | Top-p sampling (sort 151936 logits) | 36.4 | 2.600 | 0.4% |

## Full component breakdown

| Component | ms / forward | Share |
|---|---:|---:|
| LittleBit binary projections (all 7 per layer) | 461.460 | 79.5% |
| lm_head (full vocab dot product) | 112.181 | 19.3% |
| Top-p sampling (sort 151936 logits) | 2.600 | 0.4% |
| SwiGLU activation | 2.054 | 0.4% |
| Self-attention core (scores + softmax + V) | 1.137 | 0.2% |
| Input + post + final RMSNorm | 0.469 | 0.1% |
| Residual adds | 0.307 | 0.1% |
| QK RMSNorm | 0.275 | 0.0% |
| Rotary embedding | 0.086 | 0.0% |
| KV cache write | 0.081 | 0.0% |
| Greedy sampling | 0.023 | 0.0% |
| Embedding lookup | 0.003 | 0.0% |

## Per-layer cost

- Mean per layer per forward: `12.953 ms`
- Slowest 5 layers (total over 14 replayed forwards): L8=206.2 ms, L4=200.2 ms, L25=196.6 ms, L9=195.9 ms, L10=195.8 ms
- Total transformer time by forward index (prefill then decode): `469.3, 440.3, 478.4, 497.8, 458.6, 444.0, 448.3, 430.9, 451.1, 491.7, 526.5, 444.8, 533.5, 412.9` ms

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

- Projection effective bandwidth: `1.063 GB/s`
- `lm_head` bytes per call: `1.159 GiB`
- `lm_head` effective bandwidth (in-forward): `11.095 GB/s`
- `lm_head` isolated bandwidth (3 repeats): `12.690 GB/s`
- KV data read by attention core: `29.5 MiB` over 14 forwards

## Interpretation

The numbers above are the baseline before any optimization. Top-3 proposals will be written in `docs/phase9-profiling-baseline.md` after review; this generated file contains raw measurements only.
