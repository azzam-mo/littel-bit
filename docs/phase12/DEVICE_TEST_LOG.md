# Phase 12 — Real iPhone Test Log

> **Autonomous-completion update (2026-11-25):** P12-R07 was later executed
> against an AArch64 Linux ELF under QEMU user-mode with Test 7c = 308 cases
> across 7 kernels and `first divergent intermediate: none`; CTest reached 9/9.
> This is not native ARM64 hardware, macOS, or iPhone evidence. Every T01–T11
> record below remains `BLOCKED`. See
> `docs/phase12/FINAL_COMPLETION_REPORT.md` for the current decision and
> platform-labeled evidence.

هذا الملف هو سجل نتائج الجهاز الحقيقي. لم تُنفذ أي اختبارات على iPhone في بيئة العمل الحالية، لذلك كل السجلات أدناه `BLOCKED`. لا تُغيّر الحالة إلى `PASS` إلا بعد تنفيذ الاختبار فعليًا على جهاز موصول وإرفاق الدليل المطلوب.

## Environment prerequisites

| Item | Required | Local status | Evidence needed on device run |
|---|---|---|---|
| macOS + Xcode 16.x | Yes | NOT AVAILABLE locally | `xcodebuild -version` output |
| iPhone (real device) | Yes | NOT AVAILABLE locally | Device model + iOS version |
| Signing path A: Apple Developer Team ID + certificate + profile | One of A or B | NOT AVAILABLE locally | Team ID + profile name + entitlement dump |
| Signing path B: AltStore Classic + AltServer + free Apple ID | One of A or B | NOT AVAILABLE locally | AltStore version + account type screen; credentials stay on user's Mac only |
| Bundle ID `com.littlebit.loader` in built app | Yes | VERIFIED statically in `Info.plist`/generator | `plutil` on installed app |
| arm64 device binary / no x86_64 | Yes | VERIFIED statically in generator; script check added, not executed on macOS | `lipo -archs` output from build |
| `com.apple.developer.kernel.increased-memory-limit` in generated profile | Yes for increased-memory validation | UNKNOWN | Entitlement dump from installed app + profile |
| AltStore-compatible source IPA with empty ad-hoc entitlement blob | For path B | CODE ADDED — NOT EXECUTED (no macOS) | Script output + SHA-256 + readback plist showing no privileged entitlement |
| Converted LittleBit model directory | Yes | NOT COMMITTED / NOT AVAILABLE locally | Directory listing + checksums |
| IPA installed and launched on iPhone | Yes | NOT AVAILABLE locally | Installed app version screen + launch screenshot |

## Test records

### P12-T01 — Load real model
- Status: `BLOCKED`
- Command/action: import converted model folder, select it, tap load.
- Expected: state becomes `.ready`; `currentModelPath` set; no memory warning.
- Evidence required: timestamped screen recording/screenshot + `xcodebuild`/device console log.
- Actual result: not executed.

### P12-T02 — Generation
- Status: `BLOCKED`
- Command/action: send an Arabic and an English prompt; wait for completion.
- Expected: valid UTF-8 stream, finite metrics, no crash.
- Evidence required: transcript text + metrics (`TTFT`, `tokensPerSecond`, `tokensGenerated`).
- Actual result: not executed.

### P12-T03 — Cancel
- Status: `BLOCKED`
- Command/action: start a long generation, tap stop mid-stream.
- Expected: generation stops promptly, UI returns to `.ready`, no orphan background work.
- Evidence required: timestamped log of cancel request and completion.
- Actual result: not executed.

### P12-T04 — Unload during generation
- Status: `BLOCKED`
- Command/action: start generation, request unload before it finishes.
- Expected: deferred teardown; unload completion after mmap release; state `.unloaded`.
- Evidence required: console log showing unload request, teardown, and completion order.
- Actual result: not executed.

### P12-T05 — Reload after unload
- Status: `BLOCKED`
- Command/action: load the same model again after T04.
- Expected: successful load; no stale KV/prompt state; state `.ready`.
- Evidence required: load log + screenshot.
- Actual result: not executed.

### P12-T06 — Memory / Jetsam
- Status: `BLOCKED`
- Command/action: run at `maxSeqLen` 512 and 1024; monitor memory warnings and termination.
- Expected: no Jetsam at target context; record peak RSS from bridge metrics.
- Evidence required: Instruments memory graph or OS log plus `residentMemoryMB`, `kvCacheMB`.
- Actual result: not executed.

### P12-T07 — Prefill / Decode / TTFT
- Status: `BLOCKED`
- Command/action: measure prefill and decode for short and long prompts.
- Expected: finite `ttftMs` and `tokensPerSecond`; note temperature/top-p.
- Evidence required: table of prompt length, `ttftMs`, `tokensGenerated`, `tokensPerSecond`.
- Actual result: not executed.

### P12-T08 — Files import / security-scoped resource
- Status: `BLOCKED`
- Command/action: import model folder from Files, load, unload, delete/re-import.
- Expected: mmap remains valid while loaded; no stale scope; replacement waits for teardown.
- Evidence required: console log + import/delete/reload sequence.
- Actual result: not executed.

### P12-T09 — Long session stability
- Status: `BLOCKED`
- Command/action: repeated load/generate/cancel/unload/reload cycles for 30+ minutes.
- Expected: no crash, no leak growth, no state deadlock.
- Evidence required: long-run log + memory samples.
- Actual result: not executed.

### P12-T10 — Signed IPA install (Xcode/Developer path)
- Status: `BLOCKED`
- Command/action: archive, export with `docs/phase12/ExportOptions-device.plist.example`, install via Xcode/Configurator.
- Expected: app launches on iPhone with correct bundle ID and entitlement.
- Evidence required: exported IPA name/checksum + installed app screen + entitlement dump.
- Actual result: not executed.

### P12-T11 — AltStore/free Apple ID install and refresh
- Status: `BLOCKED`
- Command/action: build archive without Apple signing, run `scripts/package_altstore_ipa.sh` (empty AltStore entitlements), sideload the resulting IPA with AltStore Classic, trust the developer profile, enable Developer Mode, then refresh before 7 days expire.
- Expected: AltStore re-signs the app; app launches with `com.littlebit.loader`; installed app/profile contains no unexpected privileged entitlement. This path does not validate `com.apple.developer.kernel.increased-memory-limit`.
- Evidence required: AltStore version + IPA SHA-256 + installed app screen + entitlement dump + refresh timestamp. Apple ID credentials must never be recorded in the repo or this log.
- Actual result: not executed.

## Phase 12 follow-up regressions (code added, not executed here)

These are Xcode/iOS tests added during the second review pass. They are **not** real-device results and must not be marked PASS without an actual `xcodebuild test` run.

| ID | Test | Status |
|---|---|---|
| P12-R01 | `testVerifyModelDirectoryMatchesEngineContract` | CODE ADDED — NOT EXECUTED (no macOS/Xcode) |
| P12-R02 | `testModelManagerCompatibilityMatchesEngineContract` | CODE ADDED — NOT EXECUTED (no macOS/Xcode) |
| P12-R03 | `testLoadRejectedWhileGenerationInFlight` | CODE ADDED — NOT EXECUTED (no macOS/Xcode) |
| P12-R04 | `testLoadFromUnloadCompletionIsAccepted` | CODE ADDED — NOT EXECUTED (no macOS/Xcode) |
| P12-R05 | `testRejectedGenerationReportsAcceptedFalse` | CODE ADDED — NOT EXECUTED (no macOS/Xcode) |
| P12-R06 | `testEngineUnloadCompletionRunsAfterBridgeTeardown` | CODE ADDED — NOT EXECUTED (no macOS/Xcode) |
| P12-R07 | ARM64 Test 7c NEON4 `M=4 K=7 pattern=2 dirty=0` lane trace + tail boundaries | CODE ADDED — NOT EXECUTED (host is x86_64; no ARM64 runner locally) |

Linux regression for the same mmap-lifetime class (`test_model_inference.cpp`: bad `lm_head` shape, failed reload, optional head tensor) **is executed** and passes as part of CTest 8/8. Linux binary-packer exactness is also executed and passes, but the ARM64-only NEON section is compiled out on this host. A clean Linux Release rebuild with `-Wall -Wextra` reports 0 warnings; workflow YAML (4/4) and all `run:` blocks (42/42) pass static checks. Swift warning fixes are source-only here because no `swiftc`/Xcode is available.

AltStore packaging script, privileged packaging script, and source JSON example are **static-check only** here: `bash -n` and JSON parse pass on Linux, but `codesign`, `plutil`, `lipo`, IPA packaging, authorized provisioning, AltServer re-signing, and device install are all `NOT EXECUTED` without macOS, an authorized identity/profile, and an iPhone. The AltStore path intentionally has an empty entitlement blob; the privileged path is `BLOCKED` by default.

## Cross-platform validation classification

| Category | Meaning | Status |
|---|---|---|
| Native ARM64 hardware execution | `ubuntu-24.04-arm` Linux runner, native `aarch64`, Test 7 = 15,960 cases and Test 7c = 308 cases | WORKFLOW PREPARED — NOT EXECUTED |
| macOS simulator execution | `macos-15` + `iphonesimulator` + arm64 XCTest; full unsigned Release device build is a separate compile check | WORKFLOW PREPARED — NOT EXECUTED |
| Signed IPA validation | Ad-hoc carrier signature and entitlement blob read back from the extracted final AltStore source IPA; not Apple signing and not installability proof | WORKFLOW PREPARED — NOT EXECUTED |
| Physical iPhone validation | Installed-app entitlements/profile, Jetsam/memory, Files security scope, T01–T11 | BLOCKED |

No actual GitHub Actions run URL exists for `9451eb9`: the local commit is not on `origin` (remote branch is `161b78f`), and `gh` is not authenticated. See `docs/phase12/CROSS_PLATFORM_VERIFICATION.md` for the full blocker matrix.
