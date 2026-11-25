# Phase 12 follow-up — NEON4 root cause, packaging split, and validation evidence

> **Autonomous-completion update (2026-11-25):** The branch was pushed at
> `3ba0b91`, and an AArch64 Linux ELF was executed under QEMU user-mode with
> Test 7 = 15,960, Test 7c = 308/7 kernels, and CTest = 9/9. This verifies the
> fix on real AArch64 instructions but is not native ARM64 hardware or an
> iPhone. The current decision and raw evidence are in
> `docs/phase12/FINAL_COMPLETION_REPORT.md` and `docs/phase12/evidence/`.

## Status summary

| Check | Result |
|---|---|
| Linux Release build + CTest | PASS — 8/8 tests |
| Binary packer exactness (x86 scalar/AVX2 paths compiled here) | PASS |
| ARM64 NEON execution | BLOCKED — host is `x86_64`; no `qemu-aarch64`, `aarch64-linux-gnu-g++`, or `clang++` |
| macOS Xcode / XCTest | BLOCKED — no macOS runner, `xcodebuild`, `codesign`, `plutil`, `lipo`, `security`, or device |
| IPA packaging / AltServer re-signing | BLOCKED — scripts are static-checked only; packaged-IPA extraction/readback is now part of both packaging paths but has not run on macOS |
| Entitlement validation | Static/workflow only: original privileged plist preserved; AltStore plist empty; final-IPA blob check added; no installed-app blob inspected |
| Cross-platform CI runs for `9451eb9` | BLOCKED — latest commit is not on `origin` and `gh` is unauthenticated; see `docs/phase12/CROSS_PLATFORM_VERIFICATION.md` |
| Workflow YAML | PASS — 4/4 parsed with `yaml.safe_load` |
| Workflow `run:` shell syntax | PASS — 42/42 blocks parsed with `bash -n` |
| pbxproj validators | PASS — `validate_pbxproj.py` and `check_xcode_project.py` |
| C++ warning cleanup | PASS — clean Release rebuild with `-Wall -Wextra` (0 warnings); Swift changes are code-reviewed only |
| Swift warning cleanup | CODE ADDED — no local `swiftc`/Xcode; `@unknown default` and unused locals addressed |
| `git diff --check` | PASS — clean |

No push was performed. Phase 13 was not started.

---

## 1. NEON4 mismatch: root cause and fix

### Reported symptom

`NEON VARIANT MISMATCH [NEON4-engine] M=4 K=7 pattern=2 dirty=0`, followed by `Abort trap: 6` / exit 134.

### First bad intermediate

For `M=4, K=7, pattern=2, dirty=0`, `pack_binary_matrix` gives row words `0x0000002a` and `0x00000055` (rows 0/2 and 1/3 respectively). `x[0..6]` cycles through the fixed specials array, so `x[6] = +qNaN (0x7fc00000)`.

Both row types have a subtract at `k=6`:

- rows 0/2: accumulator is `+Inf`, then `sum -= +qNaN`;
- rows 1/3: accumulator is `-Inf`, then `sum -= +qNaN`.

This is the first NaN operand in the case. All earlier intermediate operations agree. Under the suspected `fneg`+`fadd` lowering, the oracle produces `0xffc00000` (-qNaN) while the NEON `fsub` produces `0x7fc00000` (+qNaN). The divergence therefore occurs at the final 7-bit tail operation, not in packing, stride, or dirty-padding handling. Local ARM64 execution is unavailable, so the exact readback bits must be confirmed by the ARM64 diagnostic trace.

### Why only NEON4 failed

`M=4` is a complete 4-row group only for the production `neon_rows_group<1>` path (`binary_matvec_mult_neon` / `neon_variant_4`). `NEON8`, `NEON16`, `NEON32`, `vtst4`, `prefetch32`, and the default `binary_matvec_mult` dispatch (`neon_variant_32`) all route `M=4` through the scalar tail. They therefore never execute the divergent vector lane for this shape.

### Cause

The NEON kernel emits an explicit AArch64 ordered vector subtraction:

```asm
fsub vDst.4s, vSum.4s, vX.4s
```

AArch64 `FPProcessNaNs` chooses operand A for two-input `fadd`/`fsub`, so a NaN from `vX` keeps its sign and payload.

The scalar oracle source says `sum -= val`, but Apple clang can legally lower that to `fneg val` followed by `fadd sum, -val`. That changes the propagated NaN sign because the NaN is now operand B after negation. The result is a scalar “expected” NaN with the opposite sign from the NEON `fsub` result. This is a compiler-lowering artifact in the oracle/scalar tail, not a packing or mask defect.

### Fix

- Added `littlebit::detail::ordered_add` / `ordered_sub` in `include/littlebit/scalar_kernels.h`.
  - On ARM64 they use explicit scalar `fadd %s0, %s1, %s2` and `fsub %s0, %s1, %s2` with early-clobber outputs.
  - On other architectures they delegate to the original `a + b` / `a - b` expressions.
- `src/scalar_kernels.cpp` and the frozen test oracle now use those helpers, so the oracle, scalar tails, and NEON lanes all use the same ordered AArch64 semantics.
- The NEON kernel still uses ordered vector `fadd`/`fsub`; it was not disabled, routed to scalar, or weakened.
- Added `neon_variant_4_trace` (test-only) to expose all four accumulator lanes after each packed bit.

### Permanent diagnostics and regressions

`tests/test_binary_packer.cpp` now reports, on any mismatch:

- kernel label, `M`, logical `K`, padded `K`, pattern, dirty flag, and row;
- expected and actual output bit patterns and decimal values;
- packed row words as hex plus little-endian bytes;
- stride in words and bytes, total packed bytes;
- ordered reference tail accumulators with `k`, operation, and `x` bits.

Test 7c adds:

- the exact `M=4 K=7 pattern=2 dirty=0` case with lane-level first-divergence tracing;
- adjacent tail boundaries `K ∈ {5,6,7,8,9,31,32,33,63,64,65}`;
- `pattern ∈ {2,3}` and both dirty states;
- all six NEON variants plus the public engine entry point;
- 308 additional regression checks.

The existing Test 7 matrix is unchanged at 15,960 cases, so `.github/workflows/arm64-neon.yml` needs no count change. All added ARM64 code is **not executed on this x86_64 host**; it must be validated on the ARM64 runner.

---

## 2. AltStore / privileged entitlement split

### Original failure

`ARCHIVE SUCCEEDED` was followed by:

```text
ERROR: com.apple.developer.kernel.increased-memory-limit is missing from the ad-hoc signature
```

The workflow archives with `CODE_SIGNING_ALLOWED=NO`, `CODE_SIGNING_REQUIRED=NO`, and an empty `CODE_SIGN_IDENTITY`. Such an archive has no valid signed entitlement blob. Re-signing it ad hoc with the privileged repository plist did not create a trustworthy entitlement carrier; the script then asserted a key that could not be present in the expected form and reported it as if it had been granted.

### Separate paths now

| Path | Entitlements | Artifact | Status |
|---|---|---|---|
| AltStore-compatible source IPA | `ios/LittleBitLoader/LittleBitLoader.AltStore.entitlements` (empty dict) | `LittleBitLoader-altstore-compatible-source.ipa` | Prepared; requires macOS/AltServer to become installable |
| Privileged IPA | original `ios/LittleBitLoader/LittleBitLoader.entitlements` (unchanged, contains `com.apple.developer.kernel.increased-memory-limit`) | `LittleBitLoader-privileged.ipa` | BLOCKED without explicit authorization, identity, and matching profile |

- `scripts/package_altstore_ipa.sh` now rejects any AltStore entitlement file or readback blob containing `com.apple.developer.kernel.increased-memory-limit`.
- Both packaging scripts parse plists as JSON before lookup, because the dot-containing entitlement key is not a dotted `plutil` key path.
- It reports `Verified packaged IPA entitlement blob: no privileged entitlements` instead of claiming the privileged key; the check is performed on the app extracted from the final IPA.
- `scripts/package_privileged_ipa.sh` exits 2 unless `LITTLEBIT_PRIVILEGED_SIGNING_AUTHORIZED=1`, `LITTLEBIT_SIGNING_IDENTITY`, and `LITTLEBIT_PROVISIONING_PROFILE` are supplied. It decodes the profile with `security cms -D`, requires the restricted key to be `true`, requires the profile `application-identifier` to match `com.littlebit.loader`, verifies the identity is in the keychain, and re-reads the final signed IPA blob. It was not executed.
- `.github/workflows/ios-ci.yml` uses the AltStore script in the `altstore-ipa` job (now automatic on push/PR/dispatch), uploads the preserved artifact name, and invokes the privileged script to assert its `BLOCKED` exit-2 guard.
- `docs/phase12/AltStore-source.example.json` now declares `appPermissions.entitlements: []` to match the source IPA.

### Entitlement claims

An unsigned archive, an ad-hoc signature, and an Apple-issued signature are distinct. The AltStore artifact's empty ad-hoc blob proves only that the archive was packaged without privileged entitlements. No claim is made about the profile AltStore generates after re-signing. Any `increased-memory-limit` result requires an installed-app entitlement dump, profile inspection, and device memory evidence.

---

## 3. Memory review without `increased-memory-limit`

Code review only; no device or Instruments data exists locally.

### Verified from code

- `SafeTensorsFile::close()` unmaps and closes the fd; destructor and move operations null out moved-from state.
- The iOS bridge defers teardown while busy and drains pending unloads before clearing `_busy`; Swift holds the security scope until the bridge confirms load.
- `clear_cache()` resets generation state; actual mmap release occurs when `unique_ptr<Qwen3Model>` is replaced. The misleading bridge comment was corrected.
- Full `max_seq_len` KV buffers are allocated and zeroed at load. `clear_cache()` intentionally reuses them; every position is written before being read.

### Safe changes made

- Reject hostile safetensors header lengths before integer arithmetic can wrap.
- `SafeTensorInfo::numel()` now rejects negative dimensions and multiplication overflow.
- Tensor loads validate that dtype/shape bytes fit inside the declared `data_offsets` range before resizing/copying.
- KV cache reload swaps away old buffers before assigning a different capacity, avoiding simultaneous old+new peak allocation.
- Tokenizer maps reserve capacities from parsed container sizes.

These changes do not alter valid checkpoint bytes, numerics, or token ranks. They reduce avoidable allocations and make malformed Files.app imports fail earlier.

### Still unresolved / requires device

- Real Jetsam threshold and the hardcoded ~3000 MB iPhone 13 assumption in `PerformanceMonitorView.swift`.
- The ~1.19 GiB BF16 LM-head scan and resulting RSS jump during generation; any `madvise(MADV_DONTNEED)` A/B test needs device instrumentation.
- Actual tokenizer DOM peak for a real 151,936-entry vocabulary.
- Repeated load/generate/cancel/unload cycles, security-scope behaviour after backgrounding, and installed-app entitlement/profile state.

---

## 4. Compiler warning cleanup

A clean Release rebuild with the project's `-Wall -Wextra` flags produced **0 warnings**. No warning flags were suppressed, and no assertion or expected value was weakened.

- `tests/test_binary_packer.cpp`: the genuinely unused `candidate_xor_sign` and `candidate_mul_sign` helpers were removed from this test; they remain exercised in `tests/binary_matvec_probe.cpp`. The reported `expected`, `expected_words_per_row`, and both `expected_sum` values are marked `[[maybe_unused]]` because release builds strip their `assert` consumers.
- `tests/test_ops.cpp`: assert-only `expected`, `silu_val`, `sum`, and `norm_before` values were marked `[[maybe_unused]]` for the same release-only warning class.
- `tests/test_safetensors.cpp` and `tests/test_tokenizer.cpp`: assert-only `ok` values were marked `[[maybe_unused]]`.
- `ios/LittleBitLoader/Bridge/LittleBitEngine.swift`: added `@unknown default: break` to the switch over imported, non-frozen `EngineState`.
- `ios/LittleBitLoader/Views/ModelManagerView.swift`: deleted unused `hasTokenizerJSON` / `hasVocabAndMerges` locals after confirming compatibility validation delegates to `verifyModelDirectory` and only uses `hasShardedIndex` in the displayed message.
- `ios/LittleBitLoader/Views/HomeView.swift`: left unchanged because its switch is over a local frozen Swift enum.

Swift files could not be compiled locally (no `swiftc`/Xcode); those changes are evidence-based source edits, not a locally verified Swift build.

---

## 5. Remaining validation sequence

On a real ARM64 macOS runner:

1. Build Linux-equivalent ARM64 target and run `test_binary_packer`; confirm Test 7 = 15,960 cases and Test 7c = 308 cases, both PASS.
2. Run the NEON disassembly checker; confirm ordered `fadd`/`fsub` and no FMA-family instructions.
3. Run `xcodebuild test` on the iOS Simulator; record real XCTest counts.
4. Run the AltStore-compatible source IPA job on macOS (push/PR/dispatch); record SHA-256, `lipo -archs`, ad-hoc verification, and the absence of the restricted key in the final packaged IPA blob.
5. With an authorized identity/profile, run the privileged script separately; do not infer its result from step 4.
6. Install with AltStore or the authorized path, inspect the installed app/profile, then execute P12-T01–T11 in `docs/phase12/DEVICE_TEST_LOG.md`.

Until those steps produce evidence, Phase 12 remains `BLOCKED` for real-device validation.

---

## 6. Cross-platform CI preparation

- `arm64-neon.yml` now fails unless the runner is `aarch64`/`arm64`, the test binary is an AArch64 ELF, Test 7 reports exactly 15,960 cases, and Test 7c reports exactly 308 cases across 7 kernels with no first divergent intermediate.
- `ios-ci.yml` now performs a full unsigned Release device build before the arm64 simulator `build-for-testing`/`test-without-building` XCTest run. The AltStore IPA job is no longer dispatch-only.
- Both AltStore packaging paths now extract the **final IPA** and read the entitlement blob from the packaged app; the restricted key must be absent. The privileged script additionally checks profile App ID matching, keychain identity availability, and the final signed IPA blob.
- The stale macOS workflow assertion that expected `increased-memory-limit=true` in the AltStore IPA was corrected to assert exact-key absence.
- No external run was available for commit `9451eb9`; the complete blocker/run-URL matrix is in `docs/phase12/CROSS_PLATFORM_VERIFICATION.md`.
