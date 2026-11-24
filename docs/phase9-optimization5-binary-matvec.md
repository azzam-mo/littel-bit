# Phase 9 — Optimization 5: Wider AVX2 Binary-MatVec Groups

## Decision

**ACCEPTED.** The 32-row AVX2 binary-matvec kernel is bit-exact against the frozen scalar/reference path and produces a clear full-model improvement. It is committed independently after Optimization 4. No AVX-512, multithreading, `-march=native`, public-API change, or model-provider change was introduced.

## Profiling after Optimization 4

Baseline commit: `a056a5d` (Optimization 4, AVX2 binary matvec + AVX2 LM head).
Real model, 1 thread, `taskset -c 0-3`, prompt `مرحبا`, decode `262,220,16`, `max_seq_len=512`, `LM_REPEATS=3`.

Six baseline runs (`opt5_opt4_engine{1,2,3}` plus paired `opt5_baseline_opt4_{4,5,6}`), medians:

| Component | ms over 14 forwards | Share of measured forward time |
|---|---:|---:|
| LittleBit binary projections (7/layer) | 9764.16 | **66.2%** |
| LM head | 4897.01 | **33.2%** |
| Everything else | 93.25 | 0.6% |

Within the projections:

| Projection | Initial median ms |
|---|---:|
| gate_proj | 2542.34 |
| up_proj | 2552.16 |
| down_proj | 2599.98 |
| q_proj | 826.31 |
| o_proj | 827.84 |
| k+v_proj | 415.53 |

Measurement interpretation:

- Projection weights are only **467.7 MiB/token**, but the old kernel achieved just **0.703 GB/s** effective bandwidth. The kernel is therefore compute/latency-bound, not memory-bound.
- LM head is already the second-largest component at **3.56 GB/s** and is not the first target for this phase.
- Attention core was `15.795 ms` over all 14 forwards (0.1%), top-p sampling `36.750 ms` (0.3%), tokenizer `0.17 ms`, and RMSNorm/SwiGLU/KV/embedding costs were all below 1%. The data did **not** identify attention or tokenizer as the next bottleneck.
- Peak RSS was stable at `2326 MB`; CPU utilization was `99.64%` of one core.

## Why Optimization 5

The binary projection path dominates the measured forward time. The old AVX2 kernel processed eight output rows per vector and recomputed the gather offsets inside every word iteration. That leaves only one dependent accumulation chain per eight rows, while the CPU can sustain more independent work.

I measured three exact variants from the same real weights:

| Variant | Rows per group | Relative path sum | Bit-exact |
|---|---:|---:|---|
| Existing 8-row kernel | 8 | 13.60 ms | yes |
| Two vectors | 16 | 7.79 ms | yes |
| **Four vectors** | **32** | **7.51 ms** | **yes** |
| Eight vectors | 64 | 7.66 ms | yes |

The 32-row variant was the best trade-off. It keeps four independent accumulation chains, hoists the gather offsets, avoids register spills seen with wider groups, and needs no extra weight memory. The change is isolated to one source file and is trivially reversible.

## Implementation

`src/binary_packer_avx2.cpp` now has a templated `avx2_rows_group<G>` kernel:

- `G=4` processes 32 rows in four independent AVX2 lanes.
- The packed word for each eight-row group is gathered once per 32-bit word.
- The gather offset vector is computed once per call instead of inside the `w` loop.
- For each bit, the kernel still computes both `sum + x` and `sum - x` and selects with the sign mask. The inline-asm operand order remains exactly `src1=sum, src2=x`, preserving NaN payload/sign selection.
- Incomplete 32-row groups fall back to the 8-row AVX2 path, then the final 0..7 rows use the exact scalar path.
- `LITTLEBIT_DISABLE_AVX2=1` still selects the original scalar fallback.

No public headers or model-provider interfaces changed. No AVX-512 or threading was added.

## Correctness gates

- `test_binary_packer` Test 6: **1360 shape/pattern cases** bit-exact, including NaN payloads and dirty padding.
- Standalone special sweep: **72 shape combinations × G=2/G=4/G=8 = 216 checks**, all pass, including NaN, ±Inf, signed zero, subnormals, min/max finite, and tails.
- `binary_matvec_probe` on real weights: all paths `exact=yes`, `avx2_exact=yes`.
- Release CTest: **8/8 PASS**.
- Scalar-forced CTest (`LITTLEBIT_DISABLE_AVX2=1`): **8/8 PASS**.
- ASan/UBSan CTest: **8/8 PASS**.
- ASan/UBSan + scalar fallback: **8/8 PASS**.
- `validate_real_model.py`: **PASS** (`max_abs=4.29153e-06`, `rel=1.10562e-06`).
- `validate_tokenizer.py`: **PASS** (11/11).
- 36-layer `compare_logits.py`:
  - `262,220,16`: **PASS**, `max_abs=7.72476e-05`, `max_rel=3.79305e-06`, top-1 `15=15`, top-5 `5/5`.
  - `151644,9707`: **PASS**, `max_abs=6.10352e-05`, `max_rel=3.56484e-06`, top-1 `11=11`, top-5 `5/5`.
- Every full-model run (six baseline + six candidate) reported `replay_logit_max_abs=0.0`, `replay_calls=14`, and `sampled_id=15`.

## Performance after the change

Six runs per mode, medians:

| Metric | Before (Opt4) | After (Opt5) | Speedup |
|---|---:|---:|---:|
| Wall | 31354.46 ms | 22262.36 ms | **1.408×** |
| Prefill | 11471.88 ms | 8445.53 ms | **1.358×** |
| TTFT | 11472.03 ms | 8445.67 ms | **1.358×** |
| Decode/token | 1038.83 ms | 748.61 ms | **1.388×** |
| Binary projections total | 9764.16 ms | 5676.47 ms | **1.720×** |
| q_proj | 826.31 ms | 463.30 ms | 1.784× |
| k+v_proj | 415.53 ms | 238.31 ms | 1.744× |
| o_proj | 827.84 ms | 463.07 ms | 1.788× |
| gate_proj | 2542.34 ms | 1525.98 ms | 1.666× |
| up_proj | 2552.16 ms | 1534.83 ms | 1.663× |
| down_proj | 2599.98 ms | 1454.65 ms | 1.787× |
| LM head | 4897.01 ms | 4728.39 ms | unchanged code; noise |
| LM head isolated | 336.76 ms/call | 345.97 ms/call | unchanged code; noise |
| Peak RSS | 2326 MB | 2326 MB | unchanged |
| CPU utilization | 99.64% | 99.91% | unchanged within noise |
| Projection bandwidth | 0.703 GB/s | 1.21 GB/s | 1.72× |

The candidate’s wall time was lower than every baseline wall run; decode/token was lower than every baseline run. One candidate prefill run was a 11260 ms outlier, but the candidate median is 8446 ms and the projection speedup is consistent across every run.

After the change, the remaining measured split is approximately **54.1% binary projections / 45.1% LM head / 0.8% other**. The LM head is now the largest single component; further work on it would require a separate investigation and is not part of this optimization.

## Artifacts

Implementation: `src/binary_packer_avx2.cpp`.
Profiles: `docs/phase9-profiles/opt5_*`.
Machine summary: `docs/phase9-profiles/opt5_summary.json`.
Standalone probe source: `docs/phase9-profiles/opt5_matvec_probe.cpp`.

## Rollback

Restore `src/binary_packer_avx2.cpp` from commit `a056a5d` (Optimization 4), or force the scalar path with `LITTLEBIT_DISABLE_AVX2=1`. No other files need to change.
