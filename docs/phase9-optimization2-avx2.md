# Phase 9 — Optimization 2: AVX2 Binary MatVec

## Decision

**AVX2 was adopted** after the full bit-exactness, sanitizer, real-model, logits, and model-level benchmark gates passed. AVX-512 and multithreading were not used, and `-march=native` was removed from all new builds.

## Design

- Original scalar `binary_matvec_mult` body from `87c2048` is preserved verbatim as the rollback path.
- New kernel `src/binary_packer_avx2.cpp` processes **8 output rows in 8 AVX2 lanes**. Each lane walks `k` in exactly the same order as the scalar kernel, so accumulation order per output is unchanged.
- Packed words are gathered with `_mm256_i32gather_epi32`; padding bits beyond `K` are never read.
- Both `sum + x` and `sum - x` are computed, then selected with a mask. Inline asm forces `src1=sum, src2=x` for `vaddps`/`vsubps`, because GCC may otherwise swap commutative add operands and change which NaN payload/sign is propagated.
- Rows that do not fill a full 8-lane vector use the exact scalar order.
- Runtime dispatch uses `__builtin_cpu_supports("avx2")`; `LITTLEBIT_DISABLE_AVX2=1` forces the scalar fallback.

## Bit-exactness

- Frozen oracle: branchy scalar kernel from `87c2048`.
- Standalone special-value sweep: **8,100 outputs, 0 mismatches** (NaN signs/payloads, ±Inf, ±0, subnormals, min/max finite, tails, dirty padding, M=1..33, K=1..4096).
- CTest Test 6: **1,360 shape/pattern cases**, including NaN payloads `0x7fa12345` / `0xffa12345`, dirty padding, and forced scalar fallback — all bit-exact.
- ASan + UBSan at `-O3` (production optimization level): all tests pass. At `-O1` GCC swaps commutative add operands in the scalar oracle itself, changing NaN selection; this is a compiler artifact, not an AVX2 defect.

## Matvec kernel benchmark (real weights, median of 3)

Same model, `taskset -c 0-3`, 5 repeats per call.

| Implementation | Sum over 12 real projection paths (ms/call) | Speedup vs scalar |
|---|---:|---:|
| Frozen scalar reference | 213.360 | 1.000× |
| Scalar engine (no `-march=native`) | 209.776 | 1.000× |
| **AVX2 kernel** | **10.624** | **19.745×** |

All 3 runs reported `all_bit_exact=true` and `all_avx2_bit_exact=true`.

## Model-level benchmark (median of 3 per mode)

Same binary, prompt `مرحبا`, decode tokens `262,220,16`, `max_seq_len=512`, 1 thread, `taskset -c 0-3`.

| Metric | Scalar-forced | AVX2 enabled | Speedup |
|---|---:|---:|---:|
| Prefill (ms) | 155,522.800 | 13,258.091 | **11.730×** |
| TTFT (ms) | 155,522.965 | 13,258.238 | **11.730×** |
| Decode (ms/token) | 14,397.658 | 1,237.774 | **11.632×** |
| q_proj (ms) | 16,550.754 | 847.882 | 19.520× |
| k_plus_v_proj (ms) | 7,802.162 | 422.064 | 18.486× |
| o_proj (ms) | 16,427.339 | 847.342 | 19.387× |
| gate_proj (ms) | 48,553.754 | 2,601.099 | 18.667× |
| up_proj (ms) | 50,690.232 | 2,619.164 | 19.354× |
| down_proj (ms) | 49,770.311 | 2,671.533 | 18.630× |
| MLP total (ms) | 148,973.622 | 7,891.796 | 18.876× |
| lm_head (ms) | 7,984.998 | 7,327.252 | 1.090× |
| Wall (ms) | 400,151.075 | 36,865.677 | 10.854× |

`replay_logit_max_abs = 0.0` in all six runs. RSS unchanged at 2,332 MB.

For reference, the saved Optimization 1 baseline (`-march=native`) was Prefill 146,246.154 ms and Decode 13,441.365 ms/token. Those files were not modified or rerun.

## Validation after merge

- `ctest --test-dir /tmp/opencode/lbit-p9-opt2 --output-on-failure`: **8/8 PASS**
- `scripts/validate_real_model.py`: **PASS** (`max_abs=4.29153e-06`, `rel=1.10562e-06`)
- `scripts/validate_tokenizer.py`: **11/11 PASS**
- `scripts/compare_logits.py`, 36 layers:
  - `262,220,16`: **PASS**, `max_rel=3.79305e-06`, top-1/top-5 match
  - `151644,9707`: **PASS**, `max_rel=3.56484e-06`, top-1/top-5 match
- ASan/UBSan: **PASS**

## Rollback

`LITTLEBIT_DISABLE_AVX2=1` selects the original scalar path at runtime. Removing the 5-line dispatcher in `src/binary_packer.cpp` restores the exact pre-optimization default.
