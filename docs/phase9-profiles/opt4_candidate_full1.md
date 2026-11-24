# Phase 9 Profiling — opt4_candidate_full1

- Commit: `9605266`
- Generated (UTC): `2026-11-23T11:44:43Z`
- Command: `/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt4/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt4_candidate_full1.json`
- Environment: `AMD EPYC 9354P 32-Core Processor`; nproc=4; flags=avx2 avx512f bmi2 popcnt sse4_2; governor=unknown
- Engine threads: `1`; CPU utilization: `99.9%` of one core (~25.0% of the 4-core machine)
- Real model: `/home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts`; layers=36; hidden=4096; vocab=151936

## Reproducibility

```bash
# build
cmake -S . -B /tmp/opencode/lbit-p9 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/lbit-p9 -j4 --target perf_benchmark
# profile
/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt4/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt4_candidate_full1.json
```

The benchmark replays `Qwen3Model::forward_token` with timers and checks its logits against the engine on every replayed call: max abs diff = `0.0`. No engine source was changed for the baseline.

## End-to-end timing

| Metric | Value |
|---|---:|
| Model load | 932.230 ms |
| Prompt tokenization | 0.168 ms |
| Prefill (11 tokens) | 10936.887 ms |
| TTFT (tokenize + prefill) | 10937.055 ms |
| Decode (3 tokens) | 2994.060 ms |
| Decode per token | 998.020 ms |
| RSS after profiling | 2328 MB |

## Top 3 bottlenecks (measured, not assumed)

| Rank | Component | Total replay ms | ms / forward | Share |
|---:|---|---:|---:|---:|
| 1 | LittleBit binary projections (all 7 per layer) | 9352.8 | 668.060 | 66.6% |
| 2 | lm_head (full vocab dot product) | 4606.2 | 329.013 | 32.8% |
| 3 | Top-p sampling (sort 151936 logits) | 35.1 | 2.507 | 0.2% |

## Full component breakdown

| Component | ms / forward | Share |
|---|---:|---:|
| LittleBit binary projections (all 7 per layer) | 668.060 | 66.6% |
| lm_head (full vocab dot product) | 329.013 | 32.8% |
| Top-p sampling (sort 151936 logits) | 2.507 | 0.2% |
| SwiGLU activation | 1.834 | 0.2% |
| Self-attention core (scores + softmax + V) | 1.072 | 0.1% |
| Input + post + final RMSNorm | 0.422 | 0.0% |
| QK RMSNorm | 0.230 | 0.0% |
| Residual adds | 0.154 | 0.0% |
| KV cache write | 0.081 | 0.0% |
| Rotary embedding | 0.076 | 0.0% |
| Greedy sampling | 0.025 | 0.0% |
| Embedding lookup | 0.002 | 0.0% |

## Per-layer cost

- Mean per layer per forward: `18.676 ms`
- Slowest 5 layers (total over 14 replayed forwards): L16=267.0 ms, L1=266.3 ms, L2=264.8 ms, L7=264.7 ms, L4=264.6 ms
- Total transformer time by forward index (prefill then decode): `672.6, 686.2, 662.8, 673.3, 672.5, 671.5, 680.6, 677.3, 663.0, 685.7, 668.4, 675.4, 670.2, 653.0` ms

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

- Projection effective bandwidth: `0.734 GB/s`
- `lm_head` bytes per call: `1.159 GiB`
- `lm_head` effective bandwidth (in-forward): `3.783 GB/s`
- `lm_head` isolated bandwidth (3 repeats): `3.847 GB/s`
- KV data read by attention core: `29.5 MiB` over 14 forwards

## Interpretation

The numbers above are the baseline before any optimization. Top-3 proposals will be written in `docs/phase9-profiling-baseline.md` after review; this generated file contains raw measurements only.
