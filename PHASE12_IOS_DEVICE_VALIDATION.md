# Phase 12 — Real iOS Device Validation

## 0. القواعد الالتزامية

- تم فحص حالة Git والفرع **قبل أي تعديل**.
- تم التحقق من أن Phase 11 مدمجة فعلاً في `dev`؛ لم يُفترض ذلك.
- لم يُحذف أو يُستبدل أي ملف أو نتيجة سابقة.
- لم يُستخدم `git reset` أو `git clean` أوforce push.
- لم تُغيّر إعدادات API أو مزودي النماذج أو مفاتيحهم.
- لم يُدعَ نجاح أي اختبار على iPhone؛ لا يوجد iPhone ولا Xcode في بيئة العمل.
- فُصلت النتائج بوضوح بين: Linux محلي، macOS CI، واختبارات الجهاز الحقيقي.
- لم تبدأ Phase 13.

## 1. الخلاصة التنفيذية

| البند | الحالة |
|---|---|
| Phase 11 مدمجة في `dev` | CONFIRMED — `fa9b319` يحتوي `28348b1` |
| جاهزية المشروع هيكليًا للتوقيع/Build | PASS جزئيًا — المشروع سليم، الـ entitlement موجود وموصول، لكن Team ID/profile غير متاحين محليًا |
| macOS CI بعد Phase 11 | PASS حسب تقرير المستخدم: build rc=0، XCTest 8/8، NEON مثبت |
| macOS CI بعد تعديلات Phase 12 | UNVERIFIED — التعديلات لم تُرسل بعد إلى CI |
| IPA قابل للتثبيت من GitHub Actions | BLOCKED — الوظيفة الحالية تنتج IPA **غير موقّع** وغير قابل للتثبيت |
| iPhone حقيقي / Apple account / provisioning profile | BLOCKED — غير متاح محليًا |
| اختبارات الجهاز الحقيقي T01–T10 | BLOCKED — لم تُنفذ، ولا توجد نتائج مختلقة |
| Phase 13 | لم تبدأ |

---

## 2. أساس Git

تم الفحص قبل التعديل:

```text
Branch before Phase 12: dev
dev / origin/dev:        fa9b31998c5d9e322a175f7cd0b3076995c6f898
phase11-ios-lifecycle:   28348b1922d61318d74354f774965e87bbda4730
```

فحص الدمج:

```text
git merge-base --is-ancestor phase11-ios-lifecycle dev  => YES
phase11-ios-pbxproj-fix2 -> phase11-ios-paths-fix       => OK
phase11-ios-paths-fix    -> phase11-ios-swift-fix       => OK
phase11-ios-swift-fix    -> phase11-ios-lifecycle       => OK
phase11-ios-lifecycle    -> dev                         => OK
```

تم إنشاء فرع Phase 12 من `dev`:

```text
phase12-ios-device-validation @ fa9b31998c5d9e322a175f7cd0b3076995c6f898
```

لا يوجد push لهذا الفرع.

---

## 3. بيئة العمل المتاحة فعليًا

```text
OS: Linux
xcodebuild:      NOT FOUND
xcrun:           NOT FOUND
ideviceinstaller: NOT FOUND
ios-deploy:      NOT FOUND
security:        NOT FOUND
system_profiler: NOT FOUND
```

لذلك:

- كل ما يخص iOS محليًا هو **تحليل ساكن/ structural checks فقط**.
- أي نجاح iOS سابق هو من GitHub Actions وليس من هذا الجهاز.
- أي اختبار على iPhone يبقى `BLOCKED` إلى أن يتوفر جهاز وحساب Apple وتوقيع.

---

## 4. تدقيق جاهزية البناء والتوقيع

### 4.1 ما هو جاهز في المستودع

| العنصر | الحالة | الدليل |
|---|---|---|
| Bundle ID | `com.littlebit.loader` | `ios/LittleBitLoader/Info.plist:11-12` |
| Deployment target | iOS 16.0 | `scripts/generate_xcode_project.py` |
| Device family | iPhone + iPad (`1,2`) | `scripts/generate_xcode_project.py` |
| arm64 device gate | `"ARCHS[sdk=iphoneos*]" = arm64;` | `LittleBitLoader.xcodeproj/project.pbxproj` |
| C++17 / libc++ | YES | generator settings |
| No `-march=native` | CONFIRMED | `check_xcode_project.py` |
| Bridging header | موجود ومُتحقق منه | `check_xcode_project.py` |
| File sharing / in-place documents | YES | `Info.plist:25-28` |
| arm64 required capability | YES | `Info.plist:31-34` |
| Entitlements file | موجود ومت track | `ios/LittleBitLoader/LittleBitLoader.entitlements` |
| `CODE_SIGN_ENTITLEMENTS` | موصول في Debug وRelease للتطبيق | `project.pbxproj:265,290` |

### 4.2 ما لا يمكن إثباته محليًا

- صلاحية شهادة التطوير.
- وجود App ID صريح يحوي `com.apple.developer.kernel.increased-memory-limit`.
- إمكانية إضافة هذه الـ capability إلى provisioning profile الخاص بحساب Apple مجاني؛ هذا غير معروف هنا.
- أن الـ entitlement مفعّل فعليًا على جهاز حقيقي.
- تثبيت IPA على iPhone.

### 4.3 ملاحظة مهمة

`LittleBitEngineBridge verifyModelDirectory:` يتحقق فقط من وجود `model.safetensors` و`config.json`. المحرك نفسه يحتاج أكثر من ذلك؛ لذلك تم تصحيح فحص الواجهة في Phase 12 ليطابق عقد التحميل الحقيقي. انظر القسم 8.

---

## 5. مراجعة GitHub Actions وإنتاج IPA

### 5.1 الوظيفة الحالية

`.github/workflows/ios-ci.yml` تحتوي وظيفتين:

1. `ios-simulator-tests`
   - تبني وتختبر على iOS Simulator.
   - تنتج summary وتحقق من NEON.
   - لا توقع تطبيقًا.

2. `unsigned-ipa-optional`
   - تعمل فقط عبر `workflow_dispatch`.
   - تستخدم `CODE_SIGNING_ALLOWED=NO`, `CODE_SIGNING_REQUIRED=NO`, `CODE_SIGN_IDENTITY=""`.
   - تنتج أرشيفًا ثم تضع `Payload/LittleBitLoader.app` داخل IPA.
   - السطر الصريح في الـ workflow يقول:
     ```text
     NOTE: this artifact is unsigned and is not installable on a device.
     ```

### 5.2 الحكم

- هذه الـ IPA **غير قابلة للتثبيت على iPhone**.
- لا يمكن اعتبارها دليلًا على جاهزية النشر أو التوقيع.
- لم تُضف وظيفة توقيع في CI لأن ذلك يتطلب حساب Apple/sharing secrets/profile، وهذا غير متاح، ولا يجوز افتراضه.

### 5.3 المسار الصحيح لإنتاج IPA قابل للتثبيت

موضح بالكامل في القسم 6، ويستخدم `docs/phase12/ExportOptions-device.plist.example`.

---

## 6. متطلبات Apple Developer وProvisioning

يلزم الآتي قبل أي اختبار جهاز حقيقي:

1. حساب Apple Developer مع Team ID معروف.
2. iPhone مسجّل بـ UDID داخل حساب المطور.
3. Development certificate صالح.
4. App ID صريح: `com.littlebit.loader`.
5. Development provisioning profile يربط:
   - الـ App ID،
   - الشهادة،
   - الـ device،
   - الـ capability `com.apple.developer.kernel.increased-memory-limit` إن كانت مدعومة.
6. ملف `LittleBitLoader.entitlements` موجود وموصول عبر `CODE_SIGN_ENTITLEMENTS`.
7. Xcode 16.x على macOS.
8. نموذج LittleBit محوّل بالكامل، غير مخزن في المستودع.

غير معروف محليًا:

- هل يقبل provisioning profile مجاني الـ `increased-memory-limit`؟
- هل الـ capability نفسها تحتاج موافقة إضافية من Apple لهذا الحساب؟

لذلك يُسجّل هذا البند كـ `UNKNOWN` وليس كـ `PASS`.

---

## 7. مسار موثق: Build → Sign → Install

### 7.1 توليد المشروع بالفريق

```bash
cd /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader
export TEAM_ID="REPLACE_WITH_TEAM_ID"
DEVELOPMENT_TEAM="$TEAM_ID" python3 scripts/generate_xcode_project.py
python3 scripts/validate_pbxproj.py LittleBitLoader.xcodeproj/project.pbxproj
python3 scripts/check_xcode_project.py
```

### 7.2 Archive موقعّع

```bash
export PROFILE_NAME="REPLACE_WITH_DEVELOPMENT_PROFILE_NAME"
export CERT_NAME="iPhone Developer"

xcodebuild \
  -project LittleBitLoader.xcodeproj \
  -scheme LittleBitLoader \
  -configuration Release \
  -destination 'generic/platform=iOS' \
  -archivePath "$HOME/LittleBitLoader-device.xcarchive" \
  DEVELOPMENT_TEAM="$TEAM_ID" \
  CODE_SIGN_STYLE=Manual \
  PROVISIONING_PROFILE_SPECIFIER="$PROFILE_NAME" \
  CODE_SIGN_IDENTITY="$CERT_NAME" \
  archive
```

### 7.3 Export IPA

```bash
cp docs/phase12/ExportOptions-device.plist.example /tmp/ExportOptions-device.plist
# استبدل REPLACE_WITH_TEAM_ID و REPLACE_WITH_DEVELOPMENT_PROFILE_NAME داخل الملف

xcodebuild -exportArchive \
  -archivePath "$HOME/LittleBitLoader-device.xcarchive" \
  -exportPath "$HOME/LittleBitLoader-device-ipa" \
  -exportOptionsPlist /tmp/ExportOptions-device.plist
```

### 7.4 التثبيت

أي من الطرق التالية بعد توفر الجهاز:

```bash
# عبر Xcode Devices and Simulators
# أو Apple Configurator
# أو TestFlight إذا استُخدم توزيع Ad Hoc/App Store بدل development
```

التحقق المطلوب بعد التثبيت:

- Bundle ID: `com.littlebit.loader`
- App launches
- Entitlement dump يُظهر `com.apple.developer.kernel.increased-memory-limit`

---

## 8. خطة اختبار النموذج الحقيقي على الجهاز

### 8.1 محتويات مجلد النموذج المطلوبة

من `src/model.cpp:457-537`:

| الملف | الحالة |
|---|---|
| `config.json` | مطلوب |
| `littlebit_config.json` | مطلوب فعليًا (`validate_littlebit_checkpoint`) |
| `model.safetensors` | مطلوب، ملف واحد محوّل |
| `tokenizer.json` | مفضّل |
| `vocab.json` + `merges.txt` | بديل مقبول إذا لم يوجد `tokenizer.json` |
| `model.safetensors.index.json` | غير مدعوم؛ وجوده يعني checkpoint مجزأ ويرفضه المحرك |

لا يوجد نموذج حقيقي مضاف إلى المستودع، ولم يُحمّل أو يُنشأ في Phase 12.

### 8.2 تسلسل الاختبار على الجهاز

1. استيراد المجلد من Files.
2. فحص التوافق في `ModelManagerView`.
3. Load.
4. Generation بنص عربي وإنجليزي.
5. Cancel أثناء التوليد.
6. Unload أثناء التوليد.
7. إعادة Load لنفس المجلد.
8. قياس الذاكرة/Jetsam عند `maxSeqLen` 512 ثم 1024.
9. قياس TTFT وPrefill/Decode.
10. حذف النموذج وإعادة استيراده.
11. جلسة طويلة متعددة الدورات.

السجل الكامل موجود في:

```text
docs/phase12/DEVICE_TEST_LOG.md
```

كل اختبار فيه مُسجّل كـ `BLOCKED` مع الدليل المطلوب لتحويله إلى `PASS`.

---

## 9. اختبارات Load / Generation / Cancel / Unload / Reload

| ID | الحالة | المطلوب لإتمامه |
|---|---|---|
| P12-T01 Load real model | BLOCKED | IPA موقعّع + نموذج حقيقي على الجهاز |
| P12-T02 Generation | BLOCKED | نفس المتطلبات + تسجيل النص والمقاييس |
| P12-T03 Cancel | BLOCKED | بدء توليد طويل ثم إيقافه |
| P12-T04 Unload during generation | BLOCKED | مراقبة ترتيب teardown/completion |
| P12-T05 Reload after unload | BLOCKED | إعادة التحميل بعد T04 |

لا توجد نتائج فعلية لهذه الاختبارات؛ لم يُنفذ أي منها.

---

## 10. الذاكرة وJetsam وTTFT

### 10.1 الذاكرة/Jetsam

المطلوب قياسه:

- `residentMemoryMB` من `EnginePerformanceMetrics`.
- `kvCacheMB`.
- Memory warnings.
- أي Jetsam termination.
- المقارنة بين `maxSeqLen=512` و`maxSeqLen=1024`.

الحالة: `BLOCKED`. لا توجد قياسات جهاز.

### 10.2 Prefill / Decode / TTFT

المطلوب تسجيله:

- طولprompt.
- `ttftMs`.
- `tokensGenerated`.
- `tokensPerSecond`.
- زمن التوليد الكلي.

الحالة: `BLOCKED`. لا توجد قياسات جهاز.

---

## 11. Files وSecurity Scoped Resources

المسار الحالي:

- `ModelManagerView` يفتح `.fileImporter` للمجلدات.
- يبدأ security scope ثم ينسخ المجلد إلى Documents.
- `LittleBitEngine.loadModel(from:)` يبدأ scope عند التحميل ويوقفه في unload completion.

المشاكل التي اكتُشفت وأُصلحت في Phase 12:

1. كان استبدال مجلد محمّل يحذف الملفات قبل انتهاء mmap teardown.
2. كان حذف نموذج محمّل ينفذ مباشرة بعد طلب unload دون انتظار completion.
3. كان فحص التوافق يعتبر `littlebit_config.json` اختياريًا رغم أن المحرك يرفض غيابه.
4. كان فحص التوافق يتطلب `vocab.json + merges.txt` حتى عند وجود `tokenizer.json`.

التصحيح:

- نسخ آمن إلى staging أثناء صلاحية security scope.
- تأجيل الاستبدال والحذف حتى اكتمال unload.
- فحص توافق مطابق لعقد `Qwen3Model::load`.

الحالة على الجهاز الحقيقي: `BLOCKED` — لم يُختبر بعد.

---

## 12. الاستقرار والجلسات الطويلة

الخطة:

- 30 دقيقة على الأقل.
- دورات متكررة: load → generate → cancel → unload → reload.
- مراقبة:
  - state deadlock،
  - نمو الذاكرة،
  - crash/Jetsam،
  - تسريب security scope،
  - بقاء `_busy` مفتوحًا.

الحالة: `BLOCKED` — لا يوجد جهاز.

---

## 13. المشكلات التي أُثبتت وأُصلحت

| المشكلة | الدليل | الإصلاح |
|---|---|---|
| CI summary كان يقول إن الـ entitlement غير موجود | سطر hardcoded في workflow رغم وجود الملف و`CODE_SIGN_ENTITLEMENTS` | حساب الحالة فعليًا من checkout عند كتابة الـ summary |
| حذف نموذج محمّل قبل إغلاق mmap | `deleteModel` كان يستدعي `unloadModel()` ثم يحذف فورًا | `LittleBitEngine.unloadModel(completion:)` + حذف داخل الـ completion |
| استبدال نموذج محمّل أثناء mmap | `importModelFolder` كان يحذف الهدف قبل نسخ الجديد | نسخ إلى staging فورًا، ثم استبدال بعد unload completion |
| فحص توافق خاطئ | `littlebit_config.json` كان `false` رغم أنه مطلوب في `src/model.cpp` | جعله مطلوبًا |
|Tokenizer rule غير مطابق للمحرك | المحرك يقبل `tokenizer.json` أو `vocab.json+merges.txt` | الواجهة تطبق نفس القاعدة |

لم تُغيّر هذه الإصلاحات API أو مزودي النماذج أو المفاتيح.

---

## 14. نتائج الفحوصات المحلية

### Linux / repo checks

| الفحص | النتيجة |
|---|---|
| Strict pbxproj validator | OK |
| `check_xcode_project.py` | PASS, including entitlement resolution/capability |
| Validator fixtures | 2/2 OK |
| Xcode path fixtures | 4/4 OK |
| YAML `safe_load` | OK |
| كل `run:` blocks | 11/11 `bash -n` OK |
| x86 Release build | OK |
| x86 CTest | 8/8 |
| `git diff --check` | clean |

### macOS CI

- المرة السابقة بعد Phase 11: green حسب تقرير المستخدم.
- بعد تعديلات Phase 12: **UNVERIFIED** حتى يعمل GitHub Actions من جديد.

### iPhone

- كل الاختبارات: **BLOCKED / NOT EXECUTED**.

---

## 15. المطلوب لفك الحظر

لتشغيل Phase 12 فعليًا على جهاز حقيقي يلزم:

1. macOS مع Xcode 16.x.
2. iPhone مسجّل.
3. Apple Developer Team ID.
4. شهادة تطوير + provisioning profile لـ `com.littlebit.loader`.
5. تأكيد دعم/إمكانية entitlement `increased-memory-limit` في الحساب.
6. IPA موقعّع/installable.
7. مجلد نموذج LittleBit محوّل بالكامل.
8. تنفيذ `docs/phase12/DEVICE_TEST_LOG.md` وتعبئة النتائج الفعلية.

---

## 16. الخاتمة

تم إنهاء التجهيز دون ادعاء أي نجاح على جهاز iPhone:

- Phase 11 مؤكدة داخل `dev`.
- المستودع جاهز هيكليًا للتوقيع، لكن الهوية والـ profile غير متاحين.
- CI الحالي ينتج IPA غير قابل للتثبيت فقط.
- خطة الجهاز واختباراته موثقة ومسجّلة كـ BLOCKED.
- أُصلحت أربع مشكلات قابلة للإثبات من الكود.
- Phase 13 لم تبدأ.
