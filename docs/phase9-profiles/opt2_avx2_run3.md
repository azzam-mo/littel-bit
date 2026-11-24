# Phase 9 Profiling — opt2_avx2_run3

- Commit: `87c2048`
- Generated (UTC): `2026-11-23T10:18:23Z`
- Command: `/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt2/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt2_avx2_run3.json`
- Environment: `AMD EPYC 9354P 32-Core Processor`; nproc=4; flags=avx2 avx512f bmi2 popcnt sse4_2; governor=unknown
- Engine threads: `1`; CPU utilization: `99.6%` of one core (~24.9% of the 4-core machine)
- Real model: `/home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts`; layers=36; hidden=4096; vocab=151936

## Reproducibility

```bash
# build
cmake -S . -B /tmp/opencode/lbit-p9 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/lbit-p9 -j4 --target perf_benchmark
# profile
/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt2/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt2_avx2_run3.json
```

The benchmark replays `Qwen3Model::forward_token` with timers and checks its logits against the engine on every replayed call: max abs diff = `0.0`. No engine source was changed for the baseline.

## End-to-end timing

| Metric | Value |
|---|---:|
| Model load | 945.458 ms |
| Prompt tokenization | 0.140 ms |
| Prefill (11 tokens) | 14112.690 ms |
| TTFT (tokenize + prefill) | 14112.830 ms |
| Decode (3 tokens) | 3713.321 ms |
| Decode per token | 1237.774 ms |
| RSS after profiling | 2332 MB |

## Top 3 bottlenecks (measured, not assumed)

| Rank | Component | Total replay ms | ms / forward | Share |
|---:|---|---:|---:|---:|
| 1 | LittleBit binary projections (all 7 per layer) | 10009.1 | 714.935 | 57.3% |
| 2 | lm_head (full vocab dot product) | 7371.2 | 526.512 | 42.2% |
| 3 | Top-p sampling (sort 151936 logits) | 37.1 | 2.649 | 0.2% |

## Full component breakdown

| Component | ms / forward | Share |
|---|---:|---:|
| LittleBit binary projections (all 7 per layer) | 714.935 | 57.3% |
| lm_head (full vocab dot product) | 526.512 | 42.2% |
| Top-p sampling (sort 151936 logits) | 2.649 | 0.2% |
| SwiGLU activation | 1.932 | 0.2% |
| Self-attention core (scores + softmax + V) | 1.162 | 0.1% |
| Input + post + final RMSNorm | 0.434 | 0.0% |
| QK RMSNorm | 0.256 | 0.0% |
| Residual adds | 0.166 | 0.0% |
| KV cache write | 0.082 | 0.0% |
| Rotary embedding | 0.082 | 0.0% |
| Greedy sampling | 0.029 | 0.0% |
| Embedding lookup | 0.010 | 0.0% |

## Per-layer cost

- Mean per layer per forward: `19.987 ms`
- Slowest 5 layers (total over 14 replayed forwards): L3=299.6 ms, L25=296.5 ms, L6=292.3 ms, L0=291.7 ms, L23=290.5 ms
- Total transformer time by forward index (prefill then decode): `661.9, 679.8, 665.4, 675.7, 698.8, 842.6, 795.7, 704.6, 672.9, 710.1, 786.1, 803.9, 682.8, 693.0` ms

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

- Projection effective bandwidth: `0.686 GB/s`
- `lm_head` bytes per call: `1.159 GiB`
- `lm_head` effective bandwidth (in-forward): `2.364 GB/s`
- `lm_head` isolated bandwidth (3 repeats): `2.405 GB/s`
- KV data read by attention core: `29.5 MiB` over 14 forwards

## Interpretation

The numbers above are the baseline before any optimization. Top-3 proposals will be written in `docs/phase9-profiling-baseline.md` after review; this generated file contains raw measurements only.
