# Phase 9 — Optimization 6: Bit-Exact Transpose AVX2 LM Head

## Decision

**ACCEPTED.** Replacing the LM head's per-element 32-bit gathers with contiguous 16-bit row loads plus an 8×8 register transpose is bit-exact against the frozen scalar/reference path and produces a large, stable full-model improvement.

- Base commit: `1f3ad54` (`perf(matvec): widen AVX2 binary matvec to 32-row groups`).
- Changed engine file: `src/lm_head_avx2.cpp` only.
- No AVX-512, no `-march=native`, no multithreading, no public-header or model-provider change.
- `LITTLEBIT_DISABLE_AVX2=1` still selects the original scalar path.
- No push was performed; HEAD was not reset, rebased, or amended.

## Starting point and fresh baseline

Optimization 5 left the measured split at roughly **54.1% binary projections / 45.1% LM head / 0.8% other**. The LM head is a dense BF16 table of `151936 × 4096`, read row-major from mmap: **1,244,659,712 bytes per call**.

Methodology was unchanged from Optimizations 4–5:

- Real Qwen3/LittleBit checkpoint at `/home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts`.
- Release build (`-O3 -Wall -Wextra -std=c++17`), AVX2 enabled, one thread, `taskset -c 0-3`.
- Six runs per configuration, report medians.
- Prompt `مرحبا`, decode `262,220,16`, `max_seq_len=512`, `LM_REPEATS=3`.

Fresh six-run baseline at `1f3ad54`:

| Metric | Median |
|---|---:|
| Wall | 21722.58 ms |
| Prefill / TTFT | 8063.83 / 8063.96 ms |
| Decode | 744.89 ms/token |
| LM head total (14 forwards) | 4695.11 ms |
| LM head isolated | 334.28 ms/call |
| Binary projections total | 5621.48 ms |
| Peak RSS | 2326 MB |
| CPU utilization | 99.94% of one core |

Every baseline run reported `replay_logit_max_abs=0.0`, `replay_calls=14`, `threads=1`, and `sampled_id=15`.

## Kernel investigation

### The old kernel is gather-bound, not memory-bound

The previous kernel kept eight vocabulary rows in eight AVX2 lanes and gathered eight BF16 weights per hidden group. For the real table that is:

- `151936 / 8 = 18992` row groups,
- `4096 / 8 = 512` hidden groups per row group,
- `8` `vpgatherdd` instructions per hidden group,
- therefore **77.8 million gathers per call**.

Each `vpgatherdd` addresses eight different vocabulary rows, so one instruction touches eight distinct cache lines (one per lane). The generated hot loop also rebuilt the gather offsets per group and needed a separate scalar-load path for the final hidden group to avoid an out-of-bounds 32-bit read.

The isolated probe measured the old AVX2 kernel at **328.93 ms/call (3.8 GB/s)** while a pure streaming read of the same bytes ran at **42.14 ms (29.5 GB/s)**. The kernel was therefore latency/instruction-bound by the gathers, not limited by memory bandwidth.

### Candidate kernels

All candidates were compared bit-for-bit against the production scalar path on the real table and against the special-value oracle. The probe source is `docs/phase9-profiles/opt6_lm_head_probe.cpp`.

| Variant | Rows / tile | Bit-exact | Isolated median | Outcome |
|---|---|---:|---:|---|
| Old AVX2 gather kernel | 8 rows, gather | yes | 328.93 ms | baseline |
| **8×8 register transpose** | **8 rows, 8 columns** | **yes** | **91.8–103.1 ms** | **chosen** |
| 16-column-wide transpose | 8 rows, 16 columns | yes | 94.7–96.7 ms | rejected: no stable gain, extra lane extraction |
| Two 8-row groups | 16 rows, 8 columns | yes | 145.6–149.7 ms | rejected: vector spills |
| Two groups, interleaved adds | 16 rows, 8 columns | yes | 147.9–163.2 ms | rejected: spills, no gain |
| Independent row chains | 8 rows, scalar chains | **no** | 208.9–216.6 ms | rejected: 24/80 NaN cases differ |

The independent-row candidate failed 24 of 80 special-shape cases despite matching on the real weights. The cause is the legacy `addss`/`mulps` NaN operand selection used by the compiled scalar oracle versus the VEX three-operand form used by the candidate. It was not pursued.

The two-group variants were slower because GCC spilled live vector state to the stack (66 `%rsp` references in `transpose16` versus 12 pointer-only references in `transpose8`). The 32-byte-wide transpose produced no consistent in-process gain over the 128-bit version; since it also needs two ordering phases per tile, the simpler variant was kept.

## Implementation

`src/lm_head_avx2.cpp` now uses the following hot loop for each eight-row group:

1. Load `8 × 16 B` of contiguous BF16 weights, one 8-element hidden slice per vocabulary row.
2. Transpose the 8×8 array of `uint16` values with 24 `vpunpck*` instructions, so each of the eight output registers holds one hidden element from all eight vocabulary rows.
3. Convert each column with `vpmovzxwd` + `vpslld 16`, multiply by the broadcast hidden value, and accumulate with the exact compiled-oracle order.
4. Store the eight logits once per row group.

The first two products use `vaddps` with the product as `src1` and the accumulator as `src2`; the remaining six use the accumulator as `src1`. This preserves the scalar oracle's alternating operand order and therefore its NaN payload/sign selection. The multiply remains `src1=weight, src2=hidden`.

Generated hot-loop instruction mix for a 64-element tile (verified in the Release assembly):

| Operation | Count |
|---|---:|
| `vmovdqu` 16-byte row loads | 8 |
| `vpunpck*` transpose | 24 |
| `vpmovzxwd` BF16 widening | 8 |
| `vpslld` bit placement | 8 |
| `vbroadcastss` hidden scalars | 8 |
| `vmulps` | 8 |
| ordered `vaddps` | 8 |
| `vpgatherdd` | **0** |
| vector stack spills in the hot loop | **0** |

All loads are in-bounds 16-byte row loads, so the Optimization 4 special-case scalar load for the final hidden group is no longer needed. Shapes with `H % 8 != 0` or `V < 8`, and the final `V % 8` rows, still use the unchanged scalar path.

## Correctness gates

- `opt4_lm_head_exact.cpp` against the new engine: **52/52 PASS**. Covers `V = 1,7,8,9,16,17`, `H = 1,7,8,9,15,16,33,64`, NaN payloads/signs, ±Inf, signed zero, subnormals, min/max finite, tails, and four random `257 × 4096` cases.
- Probe special sweep: **80/80 shapes PASS** for the transpose kernel, including all special patterns and tails.
- Real weights: the integrated AVX2 path matches the engine scalar path on all **151,936** logits; an independent no-inline oracle also matches.
- Release CTest: **8/8 PASS**.
- Scalar-forced CTest (`LITTLEBIT_DISABLE_AVX2=1`): **8/8 PASS**.
- ASan/UBSan CTest: **8/8 PASS**.
- ASan/UBSan scalar CTest: **8/8 PASS**.
- ASan/UBSan finite exactness harness: **4/4 PASS**. The previously documented sanitizer-only NaN-payload oracle caveat is unchanged; production `-O3` is bit-exact.
- `validate_real_model.py`: **PASS** (4431 tensors, missing 0; layer `max_abs=4.29153e-06`, `rel=1.10562e-06`).
- `validate_tokenizer.py`: **PASS** (11/11). The probe loads the model's `tokenizer.json`; placeholder `vocab.json`/`merges.txt` were required by the validator wrapper only, not by the tokenizer path under test.
- 36-layer `compare_logits.py`:
  - `262,220,16`: **PASS**, `max_abs=7.72476e-05`, `max_rel=3.79305e-06`, top-1 `15=15`, top-5 `5/5`.
  - `151644,9707`: **PASS**, `max_abs=6.10352e-05`, `max_rel=3.56484e-06`, top-1 `11=11`, top-5 `5/5`.
- Every full-model run (baseline, candidate, and interleaved) reported `replay_logit_max_abs=0.0`, `replay_calls=14`, `sampled_id=15`, and `threads=1`.

## Performance

Six runs per mode, medians. Raw per-run values are in `docs/phase9-profiles/opt6_summary.json`.

| Metric | Baseline (Opt5) | Candidate (Opt6) | Change |
|---|---:|---:|---:|
| Wall | 21722.58 ms | 16369.45 ms | **1.327×** |
| Prefill | 8063.83 ms | 6346.02 ms | **1.271×** |
| TTFT | 8063.96 ms | 6346.16 ms | **1.271×** |
| Decode/token | 744.89 ms | 529.82 ms | **1.406×** |
| LM head total (14 forwards) | 4695.11 ms | 1534.87 ms | **3.059×** |
| LM head isolated | 334.28 ms/call | 104.83 ms/call | **3.189×** |
| Peak RSS | 2326 MB | 2326 MB | unchanged |
| CPU utilization | 99.94% | 99.62% | unchanged within noise |

The LM head's share of measured projection-plus-LM-head time falls from **45.5% to 19.5%**. After this change the binary projections dominate again.

Isolated probe medians over three independent six-round runs:

| Measurement | Before | After |
|---|---:|---:|
| Engine AVX2 | 328.93 ms (3.8 GB/s) | **97.89 ms (12.7 GB/s)** |
| Scalar reference | 508.80 ms | 513.44 ms (unchanged code; noise) |
| Streaming read only | 42.14 ms (29.5 GB/s) | 44.09 ms (28.2 GB/s) |

The new kernel is **3.36× faster than the old kernel** and **2.22× above the pure streaming floor** on this machine.

Every candidate wall-time run was below every baseline wall-time run: candidate walls were `15148–17849 ms`, baseline walls were `21181–24365 ms`. Decode improved in every paired run.

### Projection-timing cross-check

In the primary six-run candidate batch the unchanged binary projections measured about 12% higher than in the baseline batch. That was run-order drift, not an engine regression:

- Candidate projection code was untouched, and a same-process A/B confirmed the LM head kernel alone was identical.
- An interleaved baseline/candidate check (`opt6_interleave_b1/c1/b2/c2`) reproduced the candidate gain and showed the effect is not systematic: the second candidate run's projection sum was **5665 ms**, essentially equal to the second baseline run's **5716 ms**.
- Wall time and LM-head time improved in both interleaved candidate runs.

## Artifacts

- Implementation: `src/lm_head_avx2.cpp`.
- Report: `docs/phase9-optimization6-lm-head.md`.
- Standalone probe source: `docs/phase9-profiles/opt6_lm_head_probe.cpp`.
- Machine summary: `docs/phase9-profiles/opt6_summary.json`.
- Full-model runs: `docs/phase9-profiles/opt6_baseline_run{1..6}.{json,md,stdout.log}`, `opt6_candidate_run{1..6}.{json,md,stdout.log}`.
- Interleaved check: `opt6_interleave_{b1,c1,b2,c2}.{json,md,stdout.log}`.
- Probe runs: `opt6_probe_baseline{1..3}.json`, `opt6_probe_candidate{1..3}.json`, plus candidate `2`/`3` stdout logs.
- Validation logs: `opt6_ctest_{release,scalar}.log`, `opt6_san_ctest{,_scalar}.log`, `opt6_lm_head_exact_{release,san_finite}.log`, `opt6_validate_real_model.log`, `opt6_validate_tokenizer.log`, `opt6_logits_candidate_{26222016,1516449707}.log`.

## Rollback

Restore the Optimization 5 file and rebuild:

```
git checkout 1f3ad54 -- src/lm_head_avx2.cpp
```

Or force the unchanged scalar path at runtime with `LITTLEBIT_DISABLE_AVX2=1`. No other engine file, public header, test, or model-provider interface needs to change.
