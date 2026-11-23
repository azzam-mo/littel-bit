# Phase 9 Profiling — opt2_scalar_run2

- Commit: `87c2048`
- Generated (UTC): `2026-11-23T10:09:47Z`
- Command: `/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt2/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt2_scalar_run2.json`
- Environment: `AMD EPYC 9354P 32-Core Processor`; nproc=4; flags=avx2 avx512f bmi2 popcnt sse4_2; governor=unknown
- Engine threads: `1`; CPU utilization: `99.6%` of one core (~24.9% of the 4-core machine)
- Real model: `/home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts`; layers=36; hidden=4096; vocab=151936

## Reproducibility

```bash
# build
cmake -S . -B /tmp/opencode/lbit-p9 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/opencode/lbit-p9 -j4 --target perf_benchmark
# profile
/usr/bin/taskset -c 0-3 /tmp/opencode/lbit-p9-opt2/perf_benchmark /home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts مرحبا 262,220,16 512 1 3 /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader/docs/phase9-profiles/opt2_scalar_run2.json
```

The benchmark replays `Qwen3Model::forward_token` with timers and checks its logits against the engine on every replayed call: max abs diff = `0.0`. No engine source was changed for the baseline.

## End-to-end timing

| Metric | Value |
|---|---:|
| Model load | 1049.734 ms |
| Prompt tokenization | 0.242 ms |
| Prefill (11 tokens) | 157387.416 ms |
| TTFT (tokenize + prefill) | 157387.658 ms |
| Decode (3 tokens) | 43192.974 ms |
| Decode per token | 14397.658 ms |
| RSS after profiling | 2332 MB |

## Top 3 bottlenecks (measured, not assumed)

| Rank | Component | Total replay ms | ms / forward | Share |
|---:|---|---:|---:|---:|
| 1 | LittleBit binary projections (all 7 per layer) | 189098.3 | 13507.019 | 95.5% |
| 2 | lm_head (full vocab dot product) | 8773.1 | 626.649 | 4.4% |
| 3 | Top-p sampling (sort 151936 logits) | 36.2 | 2.587 | 0.0% |

## Full component breakdown

| Component | ms / forward | Share |
|---|---:|---:|
| LittleBit binary projections (all 7 per layer) | 13507.019 | 95.5% |
| lm_head (full vocab dot product) | 626.649 | 4.4% |
| Top-p sampling (sort 151936 logits) | 2.587 | 0.0% |
| SwiGLU activation | 1.885 | 0.0% |
| Self-attention core (scores + softmax + V) | 1.146 | 0.0% |
| Input + post + final RMSNorm | 0.461 | 0.0% |
| QK RMSNorm | 0.268 | 0.0% |
| Residual adds | 0.196 | 0.0% |
| KV cache write | 0.103 | 0.0% |
| Rotary embedding | 0.094 | 0.0% |
| Greedy sampling | 0.025 | 0.0% |
| Embedding lookup | 0.002 | 0.0% |

## Per-layer cost

- Mean per layer per forward: `375.337 ms`
- Slowest 5 layers (total over 14 replayed forwards): L34=6602.2 ms, L35=6020.9 ms, L31=5625.1 ms, L32=5546.6 ms, L33=5528.0 ms
- Total transformer time by forward index (prefill then decode): `13596.7, 15093.0, 13706.8, 15281.9, 13429.2, 13632.0, 13013.7, 13285.9, 13084.2, 12742.7, 12609.0, 12933.0, 13529.4, 13232.5` ms

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

- Projection effective bandwidth: `0.036 GB/s`
- `lm_head` bytes per call: `1.159 GiB`
- `lm_head` effective bandwidth (in-forward): `1.986 GB/s`
- `lm_head` isolated bandwidth (3 repeats): `2.355 GB/s`
- KV data read by attention core: `29.5 MiB` over 14 forwards

## Interpretation

The numbers above are the baseline before any optimization. Top-3 proposals will be written in `docs/phase9-profiling-baseline.md` after review; this generated file contains raw measurements only.
