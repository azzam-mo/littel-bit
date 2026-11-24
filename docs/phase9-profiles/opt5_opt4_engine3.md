# Phase 9 Profiling — opt5_opt4_engine3

- Commit: `a056a5d`
- Generated (UTC): `2026-11-24T02:59:37Z`
- Command: `/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt4/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt5_opt4_engine3.json`
- Environment: `AMD EPYC 9354P 32-Core Processor`; nproc=4; flags=avx2 avx512f bmi2 popcnt sse4_2; governor=unknown
- Engine threads: `1`; CPU utilization: `99.9%` of one core (~25.0% of the 4-core machine)
- Real model: `/home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts`; layers=36; hidden=4096; vocab=151936

## Reproducibility

```bash
# build
cmake -S . -B /tmp/opencode/lbit-p9 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/lbit-p9 -j4 --target perf_benchmark
# profile
/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt4/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt5_opt4_engine3.json
```

The benchmark replays `Qwen3Model::forward_token` with timers and checks its logits against the engine on every replayed call: max abs diff = `0.0`. No engine source was changed for the baseline.

## End-to-end timing

| Metric | Value |
|---|---:|
| Model load | 881.814 ms |
| Prompt tokenization | 0.145 ms |
| Prefill (11 tokens) | 11336.058 ms |
| TTFT (tokenize + prefill) | 11336.203 ms |
| Decode (3 tokens) | 3007.407 ms |
| Decode per token | 1002.469 ms |
| RSS after profiling | 2326 MB |

## Top 3 bottlenecks (measured, not assumed)

| Rank | Component | Total replay ms | ms / forward | Share |
|---:|---|---:|---:|---:|
| 1 | LittleBit binary projections (all 7 per layer) | 9529.1 | 680.647 | 66.6% |
| 2 | lm_head (full vocab dot product) | 4683.1 | 334.505 | 32.7% |
| 3 | Top-p sampling (sort 151936 logits) | 37.7 | 2.691 | 0.3% |

## Full component breakdown

| Component | ms / forward | Share |
|---|---:|---:|
| LittleBit binary projections (all 7 per layer) | 680.647 | 66.6% |
| lm_head (full vocab dot product) | 334.505 | 32.7% |
| Top-p sampling (sort 151936 logits) | 2.691 | 0.3% |
| SwiGLU activation | 1.791 | 0.2% |
| Self-attention core (scores + softmax + V) | 1.086 | 0.1% |
| Input + post + final RMSNorm | 0.438 | 0.0% |
| QK RMSNorm | 0.234 | 0.0% |
| Residual adds | 0.169 | 0.0% |
| KV cache write | 0.087 | 0.0% |
| Rotary embedding | 0.087 | 0.0% |
| Greedy sampling | 0.028 | 0.0% |
| Embedding lookup | 0.003 | 0.0% |

## Per-layer cost

- Mean per layer per forward: `19.028 ms`
- Slowest 5 layers (total over 14 replayed forwards): L0=273.9 ms, L2=273.2 ms, L1=272.8 ms, L3=270.4 ms, L13=270.0 ms
- Total transformer time by forward index (prefill then decode): `680.5, 678.5, 675.4, 716.2, 693.9, 673.3, 735.7, 687.6, 691.2, 666.8, 663.0, 677.1, 676.0, 674.8` ms

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
- `lm_head` effective bandwidth (in-forward): `3.721 GB/s`
- `lm_head` isolated bandwidth (3 repeats): `3.665 GB/s`
- KV data read by attention core: `29.5 MiB` over 14 forwards

## Interpretation

The numbers above are the baseline before any optimization. Top-3 proposals will be written in `docs/phase9-profiling-baseline.md` after review; this generated file contains raw measurements only.
