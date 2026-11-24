# Phase 9 — Optimization 4: AVX2 Dense BF16 LM Head

## Decision

**ACCEPTED.** The fixed AVX2 LM-head kernel is bit-exact against the original scalar path in production builds and gives a clear, stable full-model improvement. The integration is left **uncommitted** because the task requires HEAD to remain `9605266`; no reset, checkout, force push, or push was performed.

Older candidate artifacts `opt4_lm_head_candidate{1,2,3}.json` and `opt4_lm_head_candidate_probe_fixed.json` are superseded and retained only as history. They predate or include the failed first attempt at the final-group gather fix.

## Design

- New internal kernel `src/lm_head_avx2.cpp` / `src/lm_head_avx2.h`; no public header or model-provider API changed.
- Eight vocabulary rows are processed in eight AVX2 lanes; each lane walks the hidden dimension in the original `i` order.
- BF16 weights are gathered as 32-bit words and shifted into the high half of an FP32 word. The final 8-element hidden group uses in-bounds 16-bit loads so the gather never reads two bytes past the LM-head table.
- Per-8-element reductions reproduce GCC's original `-O3` auto-vectorized order, including alternating `vaddps` operand order and `vmulps` weight/hidden operand order, using inline asm. This preserves NaN payload/sign selection.
- Runtime dispatch uses `avx2_available()`; `LITTLEBIT_DISABLE_AVX2=1` keeps the original scalar path. Non-multiple-of-8 `H`, `V < 8`, and vocabulary tails use the exact scalar loop.
- Build: `-O3 -Wall -Wextra -std=c++17`; no `-march=native`, AVX-512, or multithreading. Benchmarks used 1 thread with `taskset -c 0-3`.

## Standalone real-weight probe

Artifacts: `docs/phase9-profiles/opt4_lm_head_baseline{1,2,3}.json` and `opt4_lm_head_candidate_fixed{1,2,3}.json`.

Baseline was recorded before any engine change:

| Stage | Median ms/call | Throughput |
|---|---:|---:|
| Original scalar reference | 512.950 | 2.426 GB/s |
| Memory only | 82.992 | 14.997 GB/s |
| BF16→FP32 convert only | 96.071 | 12.956 GB/s |
| Multiply only | 147.486 | 8.439 GB/s |

Fixed candidate, median of 3:

| Kernel | Median ms/call | Throughput |
|---|---:|---:|
| Same-run scalar reference | 524.710 | 2.372 GB/s |
| **AVX2 row-lane kernel** | **332.168** | **3.747 GB/s** |

Speedup: **1.579×** against the same-run reference and **1.544×** against the frozen pre-change baseline. Every fixed run reported `special_bit_exact=true`, `replay_max_abs=0`, and `candidate_real_mismatches=0`.

## Bit-exactness and validation

- `opt4_lm_head_exact.cpp` compares AVX2, scalar fallback, and a no-inline scalar oracle over `V = 1,7,8,9,16,17`, `H = 1,7,8,9,15,16,33,64`, NaN signs/payloads, ±Inf, signed zero, subnormals, min/max finite, tails, plus four random `257×4096` cases: **52/52 cases PASS**.
- Real weights: all **151,936** logits match bit-for-bit with zero mismatches; replay over all 36 layers gives `replay_max_abs=0`.
- `ctest`: **8/8 PASS** in Release, with `LITTLEBIT_DISABLE_AVX2=1`, under ASan/UBSan, and under ASan/UBSan plus scalar fallback.
- ASan/UBSan finite exact variant: **4/4 PASS**. A NaN-payload-only exact comparison built with sanitizer flags changes GCC's scalar multiply operand order in the oracle; production `-O3` remains exact. This is the same compiler-instrumentation artifact documented for Optimization 2, not an AVX2 numeric defect.
- `scripts/validate_real_model.py`: **PASS** (4431 tensors, no missing tensors; layer max_abs `4.29153e-06`, rel `1.10562e-06`).
- `scripts/validate_tokenizer.py`: **PASS** (11/11 checks, including Arabic and mixed text).
- `scripts/compare_logits.py`, 36 layers:
  - `262,220,16`: **PASS**, `max_abs=7.72476e-05`, `max_rel=3.79305e-06`, top-1 `15=15`, top-5 `5/5`.
  - `151644,9707`: **PASS**, `max_abs=6.10352e-05`, `max_rel=3.56484e-06`, top-1 `11=11`, top-5 `5/5`.
- All six full-model runs (baseline and candidate) reported `replay_logit_max_abs=0.0`, `replay_calls=14`, `sampled_id=15`, and `threads=1`.

## Full-model benchmark

Prompt `مرحبا`, decode tokens `262,220,16`, `max_seq_len=512`, detail `1`, `LM_REPEATS=3`, 1 thread, `taskset -c 0-3`, fixed real weights. Medians are over six runs per mode (`opt4_*_full{1,2,3}.json` plus `opt4_*_full_final{1,2,3}.json`); raw per-run values are in `opt4_summary.json`.

| Metric | Baseline median | Candidate median | Change / speedup |
|---|---:|---:|---:|
| LM-head isolated (ms/call) | 509.777 | 327.220 | **1.558×** |
| LM-head total (ms) | 7250.142 | 4951.880 | **1.464×** |
| Prefill (ms) | 13499.606 | 11459.363 | **1.178×** |
| TTFT (ms) | 13499.753 | 11459.508 | **1.178×** |
| Decode (ms/token) | 1193.268 | 1050.704 | **1.136×** |
| Wall (ms) | 35920.607 | 30811.429 | **1.166×** |
| Peak RSS (MB) | 2328 | 2328 | unchanged |
| CPU utilization (% of one core) | 99.741 | 99.534 | unchanged within noise |

One candidate isolated-LM-head run had a 587 ms scheduler outlier; the median remains 327 ms, and every candidate prefill, decode, and wall run was faster than every corresponding baseline run. The end-to-end result is therefore stable and not dependent on the outlier.

## Files

Modified: `CMakeLists.txt`, `src/model.cpp`, `tests/perf_benchmark.cpp`.
Added: `src/lm_head_avx2.cpp`, `src/lm_head_avx2.h`.
`include/littlebit/model.h` and all model-provider interfaces are unchanged.

Rollback: set `LITTLEBIT_DISABLE_AVX2=1` for scalar execution, or remove `src/lm_head_avx2.cpp` from `CMakeLists.txt` and restore the three original call sites from `9605266`.
