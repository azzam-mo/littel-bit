# المرحلة السابعة — جاهزية التشغيل الفعلي على iOS

## 1. مراجعة التوافق مع iOS / ARM64

| البند | الحالة | الملاحظة |
|---|---|---|
| C++17 + ARM64 | متوافق | لا توجد افتراضات x86؛ `bf16_to_fp32` يعتمد على تخطيط البتات فقط. |
| `-march=native` | موجود في CMake فقط | يُستخدم على Linux؛ مشروع Xcode يستخدم `-O3` و `ARCHS[sdk=iphoneos*] = arm64` فقط، لذلك لا يوجد علم معالج غير صالح عند الترجمة العرضية. |
| `mmap`/`munmap` | متوافق مع iOS | `PROT_READ + MAP_SHARED` مدعوم؛ يُفتح الملف من مجلد التطبيق أو Files. |
| قياس الذاكرة | متوافق | `__APPLE__` يستخدم `mach_task_basic_info`؛ Linux يستخدم `/proc/self/statm`. |
| Foundation/Dispatch | متوافق | الجسر `Objective-C++` يستخدم GCD فقط. |
| Swift 5.9 / iOS 16 | مضبوط في `Package.swift` ومشروع Xcode المولّد. | |

**إصلاحات هذه المرحلة:**
- إزالة أبعاد GQA المكتوبة يدويًا (`32/8/128`) من `TransformerBlock` وتمريرها من `ModelConfig`؛ هذا يمنع خروجًا عن الحدود إذا اختلف `config.json` ويتيح نماذج اختبار صغيرة.
- منع تجاوز `max_seq_len` أثناء الـ prefill (كان RoPE يقرأ خارج الجدول).
- التحقق من صحة معمارية النموذج و`data_offsets` قبل إنشاء المؤشرات.
- فشل صريح عند غياب ملفات الـ tokenizer بدل الاستمرار بإخراج غير صحيح.
- قياس زمن التحميل فعليًا بدل ترك `loadTimeMs = 0`.
- حماية الجسر من تحميل/إلغاء/تفريغ متزامن مع الاستدلال (`_busy`)، وفك ترميز UTF-8 تدريجي للتدفق، والحفاظ على صلاحية Files scope طوال عمر النموذج.

## 2. قراءة model.safetensors

- الرأس يُقرأ من أول 8 بايت (طول JSON)، ثم يُوزّن JSON ويتم تعيين `data_ptr` داخل الـ mmap بدون نسخ.
- أُضيف تحقق أن `offset_begin <= offset_end <= حجم البيانات`؛ أي ملف تالف يُرفض بدل إرجاع مؤشر خارج النطاق.
- جداول `embed_tokens` و`lm_head` تبقى `mmap` مباشر (BF16 → FP32 عند الاستخدام فقط).
- أسماء التنسورات المطلوبة مطابقة لتخطيط LittleBit: `U_packed/V_packed/u1/u2/v1/v2` لكل طبقة، مع مسار متبقٍ اختياري `*_R`.

## 3. Tokenizer والإدخال/الإخراج

-رموز التحكم Qwen (`<|im_start|>`، `<|im_end|>`، `<|endoftext|>`) تُمرّر مباشرة دون byte-level BPE.
- أُضيف `Tokenizer::apply_chat_template` ويُستخدم تلقائيًا في `generate`.
- **تحذير معروف:** الـ BPE الحالي يقسّم حسب حروف UTF-8 فقط ولا يطبّق regex الكامل لـ Qwen (`\p{L}`/`\p{N}`)؛ قد يختلف الترميز عن HF على الحواشي وعلامات الترقيم المركّبة. لم يتوفر `tokenizer.json` حقيقي للتحقق النهائي، لذا يجب اعتباره نقطة تحقق متبقية عند توفر الأوزان.
-إخراج التدفق يُفكّ الآن عبر مخزن UTF-8 تدريجي في الجسر بدل `stringWithUTF8String` لكل رمز على حدة.

## 4. مسار الاستدلال والتدفق (تم التحقق عليه على Linux)

أُضيف اختبار قابل للتنفيذ بالكامل:

```
cmake -S . -B build && cmake --build build -j && ctest --test-dir build --output-on-failure
```

- `tests/make_tiny_model.py` يولّد نموذجًا صناعيًا صغيرًا بشكل Qwen3 (طبقة واحدة، GQA 4/2، head_dim 8، vocab 256).
- `tests/test_model_inference` يتحقق من: تحميل mmap، الأبعاد من `config.json`، الـ prefill، التوليد، إيقاف مبكر من callback، خلو اللوغاريتمات من NaN، إعادة استخدام KV بعد `clear_cache`.
- النتيجة الحالية: **7/7 اختبارات ناجحة** (6 الأصلية + اختبار المسار الكامل).

## 5. الذاكرة والإيقاف

- KV Cache النظري للنموذج الحقيقي: `36 × 2 × 1024 × 8 × 128 × 4 ≈ 288 MB`، ضمن سقف iPhone 13 المتوقع (2800 MB) مع الأوزان المحمّلة عبر mmap.
- `stopGeneration` يضبط عل atomic ويُستجاب له بين الرموز (بعد كل callback).
- `unloadModel` أثناء التوليد يعيد جدولة الإلغاء ولا يحرر النموذج حتى ينتهي الخيط الخلفي.

## 6. مشروع Xcode وملفات البناء

```
python3 scripts/generate_xcode_project.py
open LittleBitLoader.xcodeproj
```

- الهدف `LittleBitLoader`: يضم واجهات SwiftUI، الجسر `.mm`، ومحرك C++ مع `HEADER_SEARCH_PATHS` و`SWIFT_OBJC_BRIDGING_HEADER`.
- الهدف `LittleBitLoaderTests`: اختبارات XCTest للجسر ودورة حياة المحرك.
- **لم يتم فتح أو بناء المشروع على macOS**؛ تحققت فقط من سلامة بنية `project.pbxproj` (مراجع وعُدد متوازنة). يجب تعيين `DEVELOPMENT_TEAM` وتشغيل البناء على جهاز حقيقي.

## 7. ما تم وما يتبقى

**تم فعليًا:** إصلاحات المحرك/الجسر، اختبار استدلال كامل على Linux، مولّد مشروع Xcode، bridging header، اختبارات XCTest، توثيق الفجوات.

**يتبقى على macOS (عائق فعلي):**
1. لا يوجد Xcode ولا جهاز iPhone في بيئة Linux الحالية → لا ادعاء بنجاح بناء أو تشغيل iOS.
2. يحتاج المشروع إلى `DEVELOPMENT_TEAM` وتوقيع لتشغيل الاختبارات على جهاز.
3. مطلوب مجلد نموذج حقيقي (`model.safetensors` + `config.json` + `vocab.json` + `merges.txt`) للتحقق من تطابق الأسماء بنسبة 100% مع HF.
4. مراجعة tokenizer مقابل `tokenizer.json` الرسمي عند توفره.
5. قياس الذاكرة الفعلية على iPhone 13 (Jetsam) وتعبئة IPA موقّع للاختبار.

لم تُلمس إعدادات API أو مزودي النماذج (بما فيهم DeepSeek).
