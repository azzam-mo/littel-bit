# Phase 12 — Real iPhone Test Log

هذا الملف هو سجل نتائج الجهاز الحقيقي. لم تُنفذ أي اختبارات على iPhone في بيئة العمل الحالية، لذلك كل السجلات أدناه `BLOCKED`. لا تُغيّر الحالة إلى `PASS` إلا بعد تنفيذ الاختبار فعليًا على جهاز موصول وإرفاق الدليل المطلوب.

## Environment prerequisites

| Item | Required | Local status | Evidence needed on device run |
|---|---|---|---|
| macOS + Xcode 16.x | Yes | NOT AVAILABLE locally | `xcodebuild -version` output |
| Apple Developer account / Team ID | Yes | NOT AVAILABLE locally | Team ID used for signing |
| Registered iPhone UDID | Yes | NOT AVAILABLE locally | Device list / identifier |
| Development certificate | Yes | NOT AVAILABLE locally | Certificate common name |
| Development profile for `com.littlebit.loader` | Yes | NOT AVAILABLE locally | Profile name + entitlements dump |
| `com.apple.developer.kernel.increased-memory-limit` in profile | Yes for real-device validation | UNKNOWN | Provisioning profile entitlement dump |
| Converted LittleBit model directory | Yes | NOT COMMITTED / NOT AVAILABLE locally | Directory listing + checksums |
| IPA installed on iPhone | Yes | NOT AVAILABLE locally | Installed app version screen |

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

### P12-T10 — Signed IPA install
- Status: `BLOCKED`
- Command/action: archive, export with `docs/phase12/ExportOptions-device.plist.example`, install via Xcode/Configurator.
- Expected: app launches on iPhone with correct bundle ID and entitlement.
- Evidence required: exported IPA name/checksum + installed app screen + entitlement dump.
- Actual result: not executed.
