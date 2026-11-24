# Phase 9 Profiling — opt5_opt4_engine2

- Commit: `a056a5d`
- Generated (UTC): `2026-11-24T02:59:05Z`
- Command: `/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt4/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt5_opt4_engine2.json`
- Environment: `AMD EPYC 9354P 32-Core Processor`; nproc=4; flags=avx2 avx512f bmi2 popcnt sse4_2; governor=unknown
- Engine threads: `1`; CPU utilization: `99.5%` of one core (~24.9% of the 4-core machine)
- Real model: `/home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts`; layers=36; hidden=4096; vocab=151936

## Reproducibility

```bash
# build
cmake -S . -B /tmp/opencode/lbit-p9 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/lbit-p9 -j4 --target perf_benchmark
# profile
/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt4/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt5_opt4_engine2.json
```

The benchmark replays `Qwen3Model::forward_token` with timers and checks its logits against the engine on every replayed call: max abs diff = `0.0`. No engine source was changed for the baseline.

## End-to-end timing

| Metric | Value |
|---|---:|
| Model load | 963.156 ms |
| Prompt tokenization | 0.160 ms |
| Prefill (11 tokens) | 11766.626 ms |
| TTFT (tokenize + prefill) | 11766.786 ms |
| Decode (3 tokens) | 4363.341 ms |
| Decode per token | 1454.447 ms |
| RSS after profiling | 2326 MB |

## Top 3 bottlenecks (measured, not assumed)

| Rank | Component | Total replay ms | ms / forward | Share |
|---:|---|---:|---:|---:|
| 1 | LittleBit binary projections (all 7 per layer) | 9762.7 | 697.336 | 66.0% |
| 2 | lm_head (full vocab dot product) | 4933.9 | 352.420 | 33.4% |
| 3 | Top-p sampling (sort 151936 logits) | 35.8 | 2.559 | 0.2% |

## Full component breakdown

| Component | ms / forward | Share |
|---|---:|---:|
| LittleBit binary projections (all 7 per layer) | 697.336 | 66.0% |
| lm_head (full vocab dot product) | 352.420 | 33.4% |
| Top-p sampling (sort 151936 logits) | 2.559 | 0.2% |
| SwiGLU activation | 1.878 | 0.2% |
| Self-attention core (scores + softmax + V) | 1.147 | 0.1% |
| Input + post + final RMSNorm | 0.434 | 0.0% |
| QK RMSNorm | 0.232 | 0.0% |
| Residual adds | 0.168 | 0.0% |
| KV cache write | 0.086 | 0.0% |
| Rotary embedding | 0.079 | 0.0% |
| Greedy sampling | 0.026 | 0.0% |
| Embedding lookup | 0.004 | 0.0% |

## Per-layer cost

- Mean per layer per forward: `19.496 ms`
- Slowest 5 layers (total over 14 replayed forwards): L0=294.1 ms, L7=286.7 ms, L18=285.5 ms, L27=283.6 ms, L2=282.9 ms
- Total transformer time by forward index (prefill then decode): `722.4, 701.5, 696.1, 673.1, 678.9, 685.6, 666.7, 678.3, 688.7, 664.2, 670.5, 656.3, 761.5, 882.1` ms

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

- Projection effective bandwidth: `0.703 GB/s`
- `lm_head` bytes per call: `1.159 GiB`
- `lm_head` effective bandwidth (in-forward): `3.532 GB/s`
- `lm_head` isolated bandwidth (3 repeats): `3.752 GB/s`
- KV data read by attention core: `29.5 MiB` over 14 forwards

## Interpretation

The numbers above are the baseline before any optimization. Top-3 proposals will be written in `docs/phase9-profiling-baseline.md` after review; this generated file contains raw measurements only.
