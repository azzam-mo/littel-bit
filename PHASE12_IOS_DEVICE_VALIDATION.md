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
| macOS CI بعد تعديلات Phase 12 | UNVERIFIED — لا توجد نتيجة فعلية؛ الـ triggers أُصلحت وستُشغّل بعد push |
| Cross-platform CI لـ `9451eb9` | WORKFLOWS PREPARED / RUNS BLOCKED — الـ commit المحلي ليس على `origin` (`161b78f`) ولا يوجد `gh auth`؛ لا توجد run URLs |
| AltStore-compatible source IPA | PREPARED (code/docs only) — يستخدم `LittleBitLoader.AltStore.entitlements` فارغًا، ويستبعد الـ entitlement الممتاز صراحةً؛ لم يُشغّل على macOS ولم يُثبّت |
| IPA قابل للتثبيت من GitHub Actions | BLOCKED — الـ artifact ليس قابلاً للتثبيت؛ AltStore/AltServer يعيد التوقيع بحساب المستخدم |
| NEON4 `M=4 K=7 pattern=2 dirty=0` | FIX PREPARED — توحيد ordered FP بين oracle/scalar tails/NEON؛ التنفيذ على ARM64 BLOCKED محليًا |
| iPhone حقيقي / Apple account / provisioning profile | BLOCKED — غير متاح محليًا |
| اختبارات الجهاز الحقيقي T01–T11 | BLOCKED — لم تُنفذ، ولا توجد نتائج مختلقة |
| تنظيف تحذيرات C++/Swift | C++ PASS محليًا (0 warnings)؛ إصلاحات Swift أُضيفت لكن Swift لم يُترجم محليًا |
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
   - تبني التطبيق كاملاً لجهاز iOS (Release، بدون توقيع Apple) عبر `xcodebuild build -destination 'generic/platform=iOS'`.
   - ثم تبني وتختبر على iOS Simulator arm64 (Debug) عبر `build-for-testing` + `test-without-building`.
   - تنتج summary وتحقق من NEON وعدّ الاختبارات؛ لا توقع تطبيقًا.

2. `altstore-ipa` (كانت `altstore-ipa-optional` وتعمل فقط عبر `workflow_dispatch`)
   - أصبحت تعمل على push/PR/workflow_dispatch لفرع `phase12-**`.
   - تستخدم `CODE_SIGNING_ALLOWED=NO`, `CODE_SIGNING_REQUIRED=NO`, `CODE_SIGN_IDENTITY=""`.
   - تستدعي `scripts/package_altstore_ipa.sh`، الذي يضيف **توقيع ad-hoc** باستخدام `ios/LittleBitLoader/LittleBitLoader.AltStore.entitlements` (فارغ)، ثم يتحقق من:
     - `CFBundleIdentifier == com.littlebit.loader`،
     - `MinimumOSVersion == 16.0`،
     - `UIFileSharingEnabled == true` و`LSSupportsOpeningDocumentsInPlace == true`،
     - وجود شريحة `arm64` وعدم وجود `x86_64`،
     - **غياب** `com.apple.developer.kernel.increased-memory-limit` من ملف الـ entitlements، من blob التطبيق المُعدّ، ومن blob التطبيق المستخرج من الـ IPA النهائي،
     - سلامة التوقيع ad-hoc وSHA-256 للـ IPA الناتج.
   - الاسم المحفوظ للـ artifact: `ios-altstore-compatible-source-ipa-optional` / `LittleBitLoader-altstore-compatible-source.ipa`.
   - خطوة منفصلة تستدعي `scripts/package_privileged_ipa.sh` وتتأكد أنه يفشل عمدًا برمز `2` ورسالة `BLOCKED`؛ لا ينتج IPA ممتازًا.

### 5.2 الحكم

- هذا الـ artifact **ليس قابلاً للتثبيت على iPhone** كما هو.
- الـ archive يُنتج بـ `CODE_SIGNING_ALLOWED=NO`؛ لذلك لا يحتوي blob توقيع صالحًا بالـ entitlements. التوقيع ad-hoc اللاحق هنا يضيف blob فارغًا صالحًا للفحص فقط؛ تتم قراءته الآن من التطبيق المستخرج من الـ IPA النهائي، وهو ليس توقيع Apple ولا صلاحية حقيقية.
- لا يدّعي هذا المسار وجود `com.apple.developer.kernel.increased-memory-limit`. الملف الممتاز الأصلي محفوظ لمسار منفصل يتطلب profile/identity مصرّحًا بهما.
- AltStore Classic يعيد توقيع الـ IPA ب_free Apple ID_ عبر AltStore/AltServer؛ هذا هو المسار المعدّ في القسم 22.
- لا توجد شهادات أو كلمات سر أو مفاتيح خاصة في المستودع أو في الـ workflow.
- `scripts/package_privileged_ipa.sh` موجود كمسار منفصل لكنه يفشل عمدًا ما لم تتوفر تفويض صريح وidentity/profile؛ لم يُنفَّذ هنا.

### 5.3 المسار البديل بحساب مطور مدفوع

موضح في القسم 7 ويستخدم `docs/phase12/ExportOptions-device.plist.example`. هذا مسار مختلف عن AltStore/free Apple ID ولا يُستنتج منه أي نتيجة على الجهاز.

---

## 6. متطلبات التوقيع وProvisioning

هناك مساران منفصلان؛ لا يُخلط بينهما:

### 6.1 المسار A — Apple Developer / Xcode Personal Team

يلزم الآتي قبل أي اختبار جهاز حقيقي عبر هذا المسار:

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

### 6.2 المسار B — AltStore Classic بحساب Apple مجاني

- AltServer يطلب بيانات Apple ID من المستخدم محليًا على جهازه، وينشئ App ID/certificate/profile الخاصة به؛ لا توجد حاجة إلى `TEAM_ID` أو certificate أو profile في المستودع.
- الـ IPA الناتج من القسم 5 يُعاد توقيعه أثناء التثبيت/التجديد؛ انتهاء التطبيقات بعد 7 days، وحد أقصى 3 تطبيقات نشطة، و10 App IDs لكل 7 days حسب توثيق AltStore.
- AltStore 2.2 أعلن دعم sideload للتطبيقات التي تحمل `increased-memory-limit`؛ AltStore 2.3 رفع الحد الأدنى لنظام iOS إلى 17.4. هذا **دعم مُعلن** وليس نتيجة على حسابنا/جهازنا.
- IPA الخاص بنا في هذا المسار **لا يطلب** الـ entitlement الممتاز؛ لذلك لا يجوز اعتباره اختبارًا لهذه الميزة.
- إذا أردنا اختبار `increased-memory-limit` فعليًا، يجب فحص الـ entitlement داخل الـ provisioning profile الناتج بعد التثبيت، واستخدام مسار توقيع مصرّح به صراحةً. وجود المفتاح في ملف المشروع أو في توقيع ad-hoc لا يكفي.

غير معروف محليًا:

- هل سيمنح حساب Apple المستخدم الـ `increased-memory-limit` فعليًا؟
- هل سيُبقي AltStore/AltServer المفتاح في الـ profile الناتج على هذا الجهاز وهذا الإصدار؟
- هل سيُطبّق iOS الـ memory limit الأعلى على طراز الجهاز المتاح؟

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

### 7.5 مسار AltStore/free Apple ID (بدون أسرار في المستودع)

من macOS مع Xcode وAltServer مثبتين:

```bash
cd /home/mrdp_24193/.gemini/antigravity/scratch/LittleBit-Loader

# 1) فحص المشروع المولّد كما هو؛ لا TEAM_ID ولا كلمة سر هنا.
python3 scripts/validate_pbxproj.py LittleBitLoader.xcodeproj/project.pbxproj
python3 scripts/check_xcode_project.py

# 2) أرشيف بدون توقيع Apple.
xcodebuild archive \
  -project LittleBitLoader.xcodeproj \
  -scheme LittleBitLoader \
  -configuration Release \
  -destination 'generic/platform=iOS' \
  -archivePath "$HOME/LittleBitLoader-altstore.xcarchive" \
  CODE_SIGNING_ALLOWED=NO \
  CODE_SIGNING_REQUIRED=NO \
  CODE_SIGN_IDENTITY="" \
  PROVISIONING_PROFILE_SPECIFIER=""

# 3) حزم AltStore-compatible source IPA بتوقيع ad-hoc و entitlements فارغة.
scripts/package_altstore_ipa.sh \
  "$HOME/LittleBitLoader-altstore.xcarchive/Products/Applications/LittleBitLoader.app" \
  "$HOME/LittleBitLoader-altstore-compatible-source.ipa"
```

النتيجة المتوقعة من Script (لم تُنفذ محليًا لغياب macOS):

```text
AltStore-compatible source IPA: .../LittleBitLoader-altstore-compatible-source.ipa
SHA-256: <printed>
Architectures: arm64
Verified from built app: CFBundleIdentifier=com.littlebit.loader, MinimumOSVersion=16.0
Verified entitlement blob: no privileged entitlements
NOTE: ad-hoc signature is NOT installable by iOS. AltStore/AltServer must re-sign with a free Apple ID.
```

ثم على iPhone:

1. ثبّت AltStore Classic وAltServer حسب دليل AltStore، وسجّل الدخول بحساب Apple مجاني داخل AltServer فقط.
2. افتح AltStore → My Apps → أضف الـ IPA من Files أو من source خاص؛ لا تُخزّن بيانات الحساب في المستودع.
3. وافق على الصلاحيات المعروضة، وانتظر إعادة التوقيع والتثبيت.
4. افتح Settings → General → VPN & Device Management وثقّ بحساب التطوير، وفعّل Developer Mode على iOS 16+.
5. تحقّق من الـ entitlement داخل التطبيق المثبّت؛ لا تفترض وجود `increased-memory-limit` من الـ source IPA، ثم نفّذ `docs/phase12/DEVICE_TEST_LOG.md`.
6. جدّد التطبيق قبل انتهاء 7 days؛ لا تعتبر التثبيت نجاحًا إلا برؤية التطبيق يعمل على الجهاز.

قالب source اختياري موجود في:

```text
docs/phase12/AltStore-source.example.json
```

استبدل `REPLACE_WITH_SHA256_FROM_PACKAGE_SCRIPT` و`size` و`downloadURL` بالقيم الفعلية قبل استخدامه. الـ `appPermissions.entitlements` فارغ الآن ليطابق الـ source IPA الذي لا يطلب صلاحيات ممتازة. إذا استُخدم entitlement إضافي لاحقًا، يجبDeclaration في المصدر والـ IPA معًا بعد إثباته على profile موقعّع.

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
| Ghaffar / مراجعة ثانية: إيقاف security scope القديم قبل قبول التحميل الجديد | `loadModel` كان يوقف الـ scope القديم قبل أن يقرر الـ bridge قبول العملية؛ عند الرفض يبقى النموذج القديم محمّلاً بـ scope متوقف | بدء scope جديد بدون لمس الحالة المخزنة، واعتماد الجديد فقط بعد نجاح التحميل |
| رفض التحميل أثناء التوليد كان يفرض `.error` في الـ UI | الـ bridge يرفض بأمان بينما `state` ما زالت `.generating` | تعيين حالة الـ UI من `bridge.state` الفعلي بدل `.error` الثابت |
| `verifyModelDirectory` أضعف من عقد المحرك | كان يتحقق من ملفين فقط | أُضيف فحص `littlebit_config.json` والـ tokenizer ورفض الـ checkpoint المجزأ |
| CI لم يكن سيعمل على فرع Phase 12 | `ios-ci.yml` كان يعمل فقط على `phase11-**` | أُضيفت `phase12-**` لكل workflows، و`dev` كأساس PR |
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
| كل `run:` blocks | 42/42 `bash -n` OK (بعد إضافة triggers لكل الـ workflows) |
| x86 Release build | OK |
| x86 CTest | 8/8 |
| Linux CI-equivalent IPA mock (`package_ipa.sh` + `unzip -l`) | OK |
| `bash -n scripts/package_altstore_ipa.sh` | OK (Script نفسه لم يُنفَّذ؛ يتطلب macOS) |
| `python3 -m json.tool docs/phase12/AltStore-source.example.json` | OK |
| Workflow trigger audit (كل `.yml`) | push/PR تشمل `phase12-**` |
| `git diff --check` | clean |

### macOS CI

- المرة السابقة بعد Phase 11: green حسب تقرير المستخدم.
- بعد تعديلات Phase 12: **UNVERIFIED** حتى يعمل GitHub Actions من جديد.

### iPhone

- كل الاختبارات: **BLOCKED / NOT EXECUTED**.

---

## 15. المطلوب لفك الحظر

لتشغيل Phase 12 فعليًا على جهاز حقيقي يلزم:

1. macOS مع Xcode 16.x وiPhone حقيقي.
2. أحد مسارَي التوقيع:
   - Apple Developer: Team ID + شهادة + provisioning profile لـ `com.littlebit.loader`، أو
   - AltStore Classic: AltServer + حساب Apple مجاني يُدخل فقط على جهاز المستخدم.
3. IPA أُعيد توقيعه فعليًا وثبّت على الجهاز (توقيع ad-hoc وحده لا يكفي).
4. تأكيد دعم/إمكانية entitlement `increased-memory-limit` في الـ provisioning profile الناتج.
5. مجلد نموذج LittleBit محوّل بالكامل.
6. تنفيذ `docs/phase12/DEVICE_TEST_LOG.md` وتعبئة النتائج الفعلية.

---

## 16. الخاتمة

تم إنهاء التجهيز دون ادعاء أي نجاح على جهاز iPhone:

- Phase 11 مؤكدة داخل `dev`.
- المستودع جاهز هيكليًا للتوقيع، لكن الهوية والـ profile غير متاحين.
- أُضيف مسار AltStore/free Apple ID قابل لإعادة الإنتاج، بتوقيع ad-hoc بـ entitlements فارغة (لا يحمل صلاحية ممتازة)، وبدون أي أسرار في المستودع.
- الـ artifact الحالي **ليس قابلاً للتثبيت**؛ إعادة التوقيع والتثبيت عبر AltStore لم تحدث هنا.
- خطة الجهاز واختباراته موثقة ومسجّلة كـ BLOCKED.
- أُصلحت مشكلات قابلة للإثبات من الكود (انظر القسم 17).
- Phase 13 لم تبدأ.

---

## 17. مراجعة الكوميتين `00875ea` و`4230e7a`

### 17.1 النتيجة

| الكوميت | المراجعة |
|---|---|
| `00875ea` | تغييرات حقيقية في 3 ملفات: CI summary، `LittleBitEngine.swift`، `ModelManagerView.swift`. لا حذف لملفات سابقة، ولا تغيير API/مزودي النماذج/مفاتيح. الإصلاحات الأصلية (تأجيل unload، staging، فحص التوافق) سليمة لكنها غير كافية وحدها. |
| `4230e7a` | توثيق فقط: التقرير + سجل الجهاز + مثال ExportOptions. كل اختبارات الجهاز مسجّلة `BLOCKED`، وهذا صحيح. لا نتائج مختلقة. |

### 17.2 مشكلات اكتُشفت أثناء المراجعة الثانية وأُصلحت

| # | المشكلة | الخطورة | الإصلاح |
|---|---|---|---|
| R1 | import/delete كانا ينتظران `currentModelPath` فقط؛ أثناء `loading` تكون المسار `nil` أو قديماً بينما C++ يقرأ المجلد | P0 | كل عمليات delete/replace تمرّ الآن عبر `engine.unloadModel(completion:)` بشكل غير مشروط |
| R2 | تحميل نموذج آخر في نفس كائن `Qwen3Model` يترك مؤشرات mmap معلّقة في الملف القديم | P0 | تصفير `embed_tokens_mmap`/`lm_head_mmap` في بداية `load()` وعند الفشل |
| R3 | تحقّق غير كافٍ من `embed_tokens.weight`/`lm_head.weight` (dtype/shape/bytes) | P0 | فحص BF16 + أبعاد `vocab×hidden` + حجم بايت كافٍ قبل تعيين المؤشرات، ورفض التحميل عند عدم التطابق |
| R4 | الـ security scope القديم كان يُوقف قبل قبول التحميل الجديد | P1 | بدء scope جديد محلياً واعتماده فقط بعد نجاح التحميل؛ الإخفاق/الرفض يوقف scope الجديد ويحتفظ بالقديم |
| R5 | رفض التحميل أثناء التوليد كان يفرض `.error` بينما المحرك ما زال يعمل | P1 | حالة الـ UI تُشتق من `bridge.state` الفعلي |
| R6 | `verifyModelDirectory:` كان أضعف من عقد المحرك؛ الواجهة لا تستخدمه | P1 | توسيعه ليشمل `littlebit_config.json` والـ tokenizer ورفض الـ checkpoint المجزأ، وجعل الواجهة تستخدمه |
| R7 | استدعاء `loadModel` من داخل unload completion كان مرفوضاً دائماً | P1 | تحرير الـ gate قبل callbacks مع علم يمنع `finishBusyOperation` من إلغاء قفل عملية بدأت داخل callback |
| R8 | `stopGeneration()` كان يعلن `.ready` قبل انتهاء العملية، ورفض التوليد كان يكتب مقاييس صفرية | P1 | إضافة `accepted` إلى `EnginePerformanceMetrics`؛ الحالة تُحدَّث من completion فقط، ورسائل الـ token تُستهدف بالـ id لا بالـ index |
| R9 | CI لا يعمل إطلاقاً على فرع `phase12-**` | P1 | إضافة `phase12-**` لكل الـ workflows و`dev` كأساس PR |

لم تُغيّر هذه الإصلاحات API أو مزودي النماذج أو مفاتيحهم.

---

## 18. اختبارات regression الجديدة

### 18.1 C++/Linux — تعمل فعلياً هنا

`tests/test_model_inference.cpp` يضيف الآن:

- رفض checkpoint معدّل `[1,1]` لـ `lm_head.weight` مع تأكيد أن المؤشرات `nullptr`.
- فشل تحميل بعد تحميل ناجح، مع تأكيد عدم بقاء مؤشرات mmap قديمة.
- نجاح تحميل checkpoint بدون `lm_head.weight` مع `lm_head_mmap == nullptr` (لا مؤشر معلّق).

النتيجة الفعلية: `test_model_inference` PASS داخل CTest 8/8، والسطر المطبوع:

```text
✓ invalid/optional mmap tensors are rejected without dangling pointers
```

### 18.2 XCTest — كود مُضاف، لم يُنفَّذ محلياً

أُضيفت هذه الاختبارات إلى `ios/LittleBitLoaderTests/EngineSmokeTests.swift`؛ كلها تتطلب Xcode واختبارات iOS، لذلك حالتها **CODE ADDED — NOT EXECUTED**:

| الاختبار | ما يغطيه |
|---|---|
| `testVerifyModelDirectoryMatchesEngineContract` | أكواد 101–105 وطرق الـ tokenizer المقبولة ورفض الـ sharded checkpoint |
| `testModelManagerCompatibilityMatchesEngineContract` | واجهة الاستيراد تستخدم نفس عقد المحرك |
| `testLoadRejectedWhileGenerationInFlight` | الرفض لا يكسر الـ gate ولا يفرض `.error` |
| `testLoadFromUnloadCompletionIsAccepted` | إعادة التحميل من unload completion تنجح (R7) |
| `testRejectedGenerationReportsAcceptedFalse` | `metrics.accepted == false` عند الرفض (R8) |
| `testEngineUnloadCompletionRunsAfterBridgeTeardown` | استكمال الـ engine بعد teardown الفعلي |

الاختبارات الأصلية A/B/C ما زالت محفوظة ولم تُحذف.

---

## 19. حالة CI الفعلية

- `git rev-parse HEAD`: `9451eb9ef5e2dba728833a537ce5840a40e7e847`.
- `git ls-remote --heads origin refs/heads/phase12-ios-device-validation`: `161b78fd5bd6b449a7736e6e18e011731e964693`.
  - أي أن أحدث commit محلي **لم يُدفع**؛ لا يمكن أن توجد run مرتبطة بـ `9451eb9`.
- `gh auth status`: **not logged in**؛ لا يوجد `GITHUB_TOKEN`/`GH_TOKEN` في البيئة.
- `GET https://api.github.com/repos/azzam-mo/littel-bit` وصفحة Actions: **404** عند الوصول المجهول، رغم أن `git ls-remote` عبر SSH يعمل.
- لذلك: **لا توجد run URLs فعلية ولا نتيجة GitHub Actions لهذا الفرع/الcommit**. لا يُدعى نجاح CI. عناوين تعريف الـ workflows أدناه هي عناوين ملفات فقط وليست أدلة تشغيل:
  - `https://github.com/azzam-mo/littel-bit/actions/workflows/arm64-neon.yml`
  - `https://github.com/azzam-mo/littel-bit/actions/workflows/ios-ci.yml`
  - `https://github.com/azzam-mo/littel-bit/actions/workflows/phase12-real-model-macos.yml`
- التفاصيل الكاملة ومصفوفة المنصات في `docs/phase12/CROSS_PLATFORM_VERIFICATION.md`.

ما تم إثباته محلياً فقط:

| الفحص | النتيجة |
|---|---|
| YAML `safe_load` لكل workflows | OK |
| `bash -n` لكل `run:` blocks | 42/42 OK |
| pbxproj validator | OK |
| `check_xcode_project.py` | PASS |
| validator fixtures | 2/2 |
| Xcode path fixtures | 4/4 |
| x86 Release build | OK |
| x86 CTest | 8/8 |
| Linux CI-equivalent IPA mock | OK |
| `bash -n scripts/package_altstore_ipa.sh` و`scripts/package_privileged_ipa.sh` | 2/2 OK (لم يُنفَّذا؛ macOS only) |
| AltStore source JSON parse | OK |
| Workflow trigger audit | `phase12-**` موجود في push وPR لكل الـ workflows |
| `git diff --check` | clean |

بعد push فقط سيصبح بإمكان `phase12-**` تشغيل `ios-ci.yml`، `ci.yml`، و`arm64-neon.yml`. لا توجد نتيجة تشغيل فعلية من هذه البيئة.

---

## 20. مخاطر متبقية

| الخطر | الحالة |
|---|---|
| iPhone حقيقي / توقيع / provisioning | BLOCKED — لم يتغير |
| نتائج XCTest بعد تعديلات Phase 12 | UNKNOWN — لم تُشغّل على macOS |
| فحص توافق المجلد يفحص وجود الملفات فقط، لا صحة المحتوى الداخلي لـ `littlebit_config.json` | مقبول؛ المحرك نفسه هو الحكم النهائي عند Load |
| استيراد مجلدين بنفس الاسم بالتوازي قد يتنافس على مجلد الوجهة | لم يُعالج؛ لا يوجد دليل على حدوثه في الاستخدام الحالي، ويُسجّل كخطر منخفض |
| Jetsam/TTFT/جلسات طويلة | BLOCKED — تتطلب جهازاً |
| إمكانية entitlement على Apple ID مجاني | UNKNOWN — AltStore 2.2+ يعلن الدعم، ويوجد تقرير مجتمع بـ AltStore 2.3، لكن profile الجهاز لم يُفحص هنا |
| توقيع ad-hoc مع entitlements فارغة للمصدر | UNVERIFIED — لم يُنفَّذ على macOS؛ ليس توقيع Apple ولا إثبات تثبيت، ولا يحمل صلاحية ممتازة |
| مسار الـ entitlement الممتاز | BLOCKED — يحتاج identity/profile مصرّحًا بهما؛ السكربت المنفصل يفشل عمدًا بدونهما |

---

## 21. ما الذي يتطلب iPhone حقيقياً

كل ما في `docs/phase12/DEVICE_TEST_LOG.md` (T01–T11)، إضافة إلى:

- تشغيل الـ XCTests الجديدة على Xcode 16.
- تثبيت IPA موقعّع فعلياً (Xcode أو إعادة توقيع AltStore).
- فحص الـ entitlement داخل التطبيق المثبّت، وتأكيد تجديد الـ 7 days.
- قياس الذاكرة/Jetsam وTTFT/Prefill/Decode.

---

## 22. بحث AltStore/free Apple ID — الوقائع والحكم

### 22.1 مصادر رسمية تمت مراجعتها

| المصدر | الوقائع المستخرجة |
|---|---|
| [AltStore Classic vs PAL](https://faq.altstore.io/altstore-pal-v.-altstore-classic) | Classic يقبل أي `.ipa`، يتطلب غالبًا كمبيوترًا للتثبيت، التطبيقات تنتهي بعد 7 days، وحد 3 تطبيقات نشطة؛ PAL مختلف تمامًا وليس مسار هذه المرحلة. |
| [AltServer](https://faq.altstore.io/altstore-classic/altserver.md) | AltServer هو الذي يعيد توقيع/تثبيت/تجديد التطبيقات؛ يجب أن يكون على نفس Wi-Fi أو USB. |
| [Installing AltStore (macOS)](https://faq.altstore.io/altstore-classic/how-to-install-altstore-macos.md) | إدخال Apple ID يتم داخل AltServer فقط؛ Trust the developer profile ثم Developer Mode على iOS 16+. |
| [App IDs](https://faq.altstore.io/altstore-classic/app-ids.md) و[Error Codes](https://faq.altstore.io/altstore-classic/error-codes.md) | 10 AppIDs/7 days، و3 تطبيقات نشطة بحساب مجاني، ورسائل خطأ 1009/2009/3013 عند التجاوز. |
| [AltStore release notes](https://faq.altstore.io/release-notes/altstore.md) | الإصدار 2.2 (16 April 2026) أضاف: `Supports sideloading apps with "Increased Memory Limit" entitlement`. الإصدار 2.3 رفع الحد الأدنى لـ iOS إلى 17.4. |
| [Make a Source](https://faq.altstore.io/developers/make-a-source.md) | مصدر AltStore يجب أن يعلن كل الـ entitlements في `appPermissions.entitlements`؛ AltStore يتحقق من تطابقها مع الـ IPA. |
| [Apple: com.apple.developer.kernel.increased-memory-limit](https://developer.apple.com/documentation/bundleresources/entitlements/com.apple.developer.kernel.increased-memory-limit) | الـ entitlement متاح على iOS 15+، وبعض طرازات الأجهزة فقط تحصل على الحد الأعلى. |
| [TN3125](https://developer.apple.com/documentation/technotes/tn3125-inside-code-signing-provisioning-profiles) | الـ provisioning profile هو allowlist؛ يجب أن يظهر المفتاح في توقيع التطبيق **و** في الـ profile كي يطبّقه iOS. |

### 22.2 حكم التوافق

- **AltStore Classic + free Apple ID**: مسار معقول لإعادة توقيع IPA الذي لا يملك شهادة Apple، بشرط أن يوفر المستخدم الحساب داخل AltServer على جهازه.
- **`com.apple.developer.kernel.increased-memory-limit`**: الدعم مُعلن رسميًا في AltStore 2.2+، وهناك تقرير مجتمع بأن AltStore 2.3 حافظ على المفتاح بحساب مجاني. لكن هذا **لا يثبت** أن الـ profile الناتج لحسابنا و جهازنا سيحتوي المفتاح، ولا أن الجهاز سيطبّق الحد الأعلى.
- **الحالة النهائية لهذا البند**: `UNKNOWN / BLOCKED` حتى يُفحص entitlement blob داخل التطبيق المثبّت بواسطة AltStore، ويُقاس سلوك الذاكرة على iPhone فعلي.

### 22.3 لماذا التوقيع ad-hoc في السكربت؟ وما الذي لا يعنيه؟

بناء IPA مع `CODE_SIGNING_ALLOWED=NO` ينتج تطبيقًا بلا entitlement blob موقعّع. لذلك يضع `scripts/package_altstore_ipa.sh` توقيع ad-hoc مع `ios/LittleBitLoader/LittleBitLoader.AltStore.entitlements` (فارغ) ليكون للـ IPA blob قابل للفحص. هذا التوقيع:

- لا يستخدم certificate أو private key أو Apple ID.
- لا يجعل التطبيق قابلاً للتثبيت على iOS.
- **لا يحمل** `com.apple.developer.kernel.increased-memory-limit`، ولا يجوز الاستدلال به على منح الصلاحية.
- يُستخدم فقط لحزم source IPA، ثم يستبدله AltServer بتوقيعه الخاص.

المسار الممتاز منفصل: `ios/LittleBitLoader/LittleBitLoader.entitlements` بقي كما هو، و`scripts/package_privileged_ipa.sh` يرفض العمل دون `LITTLEBIT_PRIVILEGED_SIGNING_AUTHORIZED=1` وidentity/profile مصرّح بهما يحوي المفتاح. لم يُنفَّذ هذا المسار هنا؛ حالته `BLOCKED`.

---

## 23. تقرير المتابعة: NEON4، فصل التوقيع، والذاكرة

أُضيف تقرير تفصيلي في `docs/phase12/NEON4_ROOT_CAUSE_AND_PACKAGING.md`. الخلاصة:

| البند | النتيجة |
|---|---|
| NEON4 `M=4 K=7 pattern=2 dirty=0` | أول انحراف في العملية الأخيرة عند `k=6` (`sum -= +qNaN`)؛ NEON يستخدم `fsub` مرتبًا، بينما قد يخفض Apple clang الطرحScalar إلى `fneg`+`fadd` ويقلب إشارة NaN |
| الإصلاح | مساعدات `ordered_add` / `ordered_sub` على ARM64 تُستخدم في الـ oracle والـ scalar tails؛ نواة NEON تبقى على `fadd`/`fsub` مرتبة، بدون fallback صامت |
| التشخيص الدائم | طباعة row، packed words/bytes، stride، `K_logical`/`K_padded`، expected/actual bits، وتتبع accumulators؛ Test 7c يتتبع أربعة lanes ويسمّي أول انحراف |
| Regression coverage | Test 7 لم يتغير (15,960 حالة)؛ Test 7c يضيف 308 حالة لحدود الـ tail وكل متغيرات NEON + engine |
| ARM64 محليًا | BLOCKED؛ البيئة `x86_64` ولا يوجد qemu أو cross compiler؛ يجب تشغيله على runner ARM64 |
| حزم IPA | فصل كامل بين AltStore-compatible source IPA بـ entitlements فارغة، ومسار privilege منفصل BLOCKED بدون identity/profile مصرّح بهما |
| ادعاء الـ entitlement | الـ ad-hoc blob الفارغ لا يثبت أي صلاحية؛ لا يُدعى أن AltStore منح `increased-memory-limit` |
| الذاكرة | مراجعة ساكنة + تحقق مبكر من رأس safetensors و`numel` ونطاقات البايتات، تقليل ذروة إعادة تحميل KV، reserve للـ tokenizer؛ القياس الفعلي/Jetsam يبقى BLOCKED |
| التحذيرات | بناء C++ مع `-Wall -Wextra`: صفر تحذيرات؛ حذف candidate helpers غير المستخدمة، ووسم قيم assert بـ `maybe_unused`، وإضافة `@unknown default` وحذف locals غير مستخدمة في Swift (الكود أُضيف؛ التجميع على macOS BLOCKED) |

جميع هذه النتائج محلية/ساكنة ما لم يُذكر غير ذلك؛ لم تُنفَّذ اختبارات iPhone أو Xcode أو AltStore هنا.

- اختبار Files + security scope على جهاز، وليس محاكياً.

---

## 24. التحقق متعدد المنصات لأحدث commit

التقرير الكامل: `docs/phase12/CROSS_PLATFORM_VERIFICATION.md`.

| البند | الحالة |
|---|---|
| ARM64 Linux hardware (`ubuntu-24.04-arm`) | WORKFLOW PREPARED / NOT EXECUTED؛ الآن يتحقق من `aarch64` وELF AArch64 و`Test 7 = 15960` و`Test 7c = 308` عبر 7 kernels |
| macOS simulator (`macos-15`, `iphonesimulator`) | WORKFLOW PREPARED / NOT EXECUTED؛ أُضيف build كامل للجهاز بـ Release/بدون توقيع قبل `build-for-testing` وXCTest |
| AltStore source IPA | WORKFLOW PREPARED / NOT EXECUTED؛ تتم قراءة entitlement blob من التطبيق المستخرج من الـ IPA النهائي، مع فحص غياب المفتاح الممتاز صراحةً |
| Signed/installed app | BLOCKED؛ لا يوجد IPA موقعّع من Apple ولا تطبيق مثبّت لفحص blob الخاص به |
| Physical iPhone | BLOCKED |
| Actual run URLs | **لا توجد**؛ `9451eb9` غير مدفوع و`origin` يشير إلى `161b78f`، و`gh` غير مسجّل الدخول |
| Local x86 evidence only | Release build بلا تحذيرات، CTest 8/8، YAML 4/4، `run:` 42/42، سكربتات الحزم 2/2 `bash -n`، `git diff --check` نظيف |

لا يُدعى نجاح ARM64 أو macOS أو IPA خارجي قبل توفر run URLs وتقارير فعلية.
