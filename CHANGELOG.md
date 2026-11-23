# Changelog — LittleBit Loader

## [Unreleased] - 2026-11-23

### المرحلة الأولى: تحليل النموذج والمواصفات
- تحليل نموذج `littlebit-qwen3-8b-0.55bpw` (Qwen3-8B base, 36 layers, 32 Q heads, 8 KV heads, head dim 128).
- فحص بنية ملف `model.safetensors` واستخراج بنية الطبقات (4935 tensor، طبقات ثنائية مضغوطة I32 مع مقاييس BF16).
- دراسة جدوى استهلاك الذاكرة على iPhone 13 (Apple A15 Bionic, 4GB RAM) واعتماد استراتيجية `mmap` مع تفادي فك الضغط المسبق في RAM.

### المرحلة الثانية: إعداد بيئة المشروع والمستودع
- تهيئة مستودع Git محلي بفرع `main` وربطه بمستودع GitHub (`git@github.com:azzam-mo/littel-bit.git`).
- إنشاء نسخة احتياطية على GitHub في فرع `backup-initial`.
- إنشاء نسخة احتياطية محلية كاملة للمشروع (`backups/LittleBit-Loader-main-backup-...tar.gz`).
- تفريع فرع التطوير `dev` وتفعيله كفرع العمل الأساسي.

### المرحلة الثالثة والرابعة: تطوير محرك LittleBit Core والـ Tokenizer
- **`binary_packer`**: تطبيق خوارزمية فك وضغط مصفوفات البتات الثنائية (`{-1, +1}`) وحسابات `binary_matvec_mult` المباشرة دون تخصيص ذاكرة وسيطة.
- **`LittleBitLinear`**: محاكاة حسابية متطابقة لطبقة LittleBitLinear تشمل المسار الأساسي (Primary) والمسار المتبقي (Residual) ومقاييس الحجم ($u_1, u_2, v_1, v_2$).
- **`SafeTensorsFile`**: قارئ تنسورات بذاكرة معنونة `mmap` مع تحويل تلقائي من BF16/FP16 إلى FP32 وقراءة مباشرة لهيكل الطبقات.
- **`Tokenizer`**: محرك Byte-Pair Encoding (BPE) مكتمل يدعم تشفير وفك تشفير النصوص الإنجليزية والعربية ومعالجة الرموز الخاصة.

### المرحلة الخامسة: تطوير معمارية Transformer Block و Qwen3Model
- **العمليات الرياضية (`ops`)**:
  - `rmsnorm`: تطبيق دقيق للـ RMSNorm مع $\epsilon = 10^{-6}$.
  - `silu`: تفعيل SwiGLU.
  - `RoPEEmbeddings`: مصفوفات الجيب وجيب التمام مع قاعدة التردد $\theta = 1,000,000$ وأبعاد 128.
  - `softmax` وخوارزميات العينات: Greedy و Temperature + Top-P.
- **الانتباه والحسابات المجمعة (`Attention` & `GQA`)**:
  - 32 Query Heads مقابل 8 KV Heads (مجموعات رباعية).
  - دمج `q_norm` و `k_norm` قبل حساب مصفوفة الانتباه.
  - إدارة متقدمة لذاكرة `KVCache` لحفظ سياق التوكنز السابقة.
- **شبكة التغذية الأمامية (`MLP`)**:
  - معمارية SwiGLU: $\text{down\_proj}(\text{silu}(\text{gate\_proj}(x)) \odot \text{up\_proj}(x))$.
- **كتلة المحول (`TransformerBlock`)**:
  - الربط الكامل مع طبقات التطبيع المسبق والوصلات المتبقية (Residual Connections).
- **النموذج الكامل (`Qwen3Model`)**:
  - جدول التضمين (`embed_tokens`) عبر `mmap` فوري بدون استهلاك ذاكرة RAM مسبق.
  - رأس اللغة (`lm_head`) عبر `mmap`.
  - حلقة توليد النصوص التدفقي (Streaming Generation) مع قياس حي لسرعة التوليد وزمن الاستجابة واستهلاك الذاكرة الحية (RSS).

### المرحلة السادسة: تطوير واجهة iOS (SwiftUI + C++ Bridge)
- **واجهات SwiftUI الكاملة**:
  - `HomeView`: اسم التطبيق، حالة المحرك الحية، النموذج الحالي، وأزرار التنقل السريع.
  - `ModelManagerView`: استيراد النماذج من تطبيق Files على iPhone، فحص سلامة وتوافق الملفات، وإدارة وحذف النماذج.
  - `ChatView`: واجهة محادثة عصرية مع توليد تدريجي متدفق، زر إيقاف فوري، وإعدادات متقدمة للحرارة والسياق.
  - `PerformanceMonitorView`: مقياس بصري لذاكرة RAM (مضبوط على السقف الآمن لجهاز iPhone 13 وهو 2800 MB)، سرعة Tokens/sec، زمن الاستجابة، وحجم KV Cache.
- **جسر الربط C++/Objective-C (`LittleBitEngineBridge.h/.mm`)**:
  - تشغيل الاستدلال في خيوط خلفية غير متزامنة لحماية سلاسة الواجهة.
  - قياس استهلاك ذاكرة النظام الفعلية في الوقت الحقيقي.
- **حزمة إدارة الاعتماديات (`Package.swift`)** و ملف التهيئة `Info.plist` لدعم مشاركة الملفات واستيراد المستندات المباشر.

### المرحلة السابعة: جاهزية التشغيل الفعلي على iOS
- توافق ARM64/iOS: إزالة أبعاد GQA المكتوبة وتمريرها من `ModelConfig`، منع تجاوز `max_seq_len`، التحقق من `config` و`data_offsets`، فشل صريح عند غياب الـ tokenizer، وتسجيل زمن التحميل.
- Tokenizer: تمييز رموز تحكم Qwen، قالب محادثة `apply_chat_template`، وفك UTF-8 تدريجي في الجسر أثناء التدفق.
- الجسر: حماية من العمليات المتزامنة (`_busy`)، إلغاء آمن، والحفاظ على صلاحية Files scope.
- بناء iOS: مولّد `scripts/generate_xcode_project.py`، bridging header، واختبارات `LittleBitLoaderTests`.
- اختبار مسار كامل على Linux عبر نموذج صناعي صغير (`tests/make_tiny_model.py` + `tests/test_model_inference`) يشمل prefill وتدفقًا وإيقافًا مبكرًا.
- توثيق كامل في `docs/phase7-ios-readiness.md` لما تم التحقق منه وما يحتاج macOS/جهازًا حقيقيًا.

### المرحلة الثامنة: Real Model Validation
- استُخدم نموذج حقيقي محلي (`littlebit-qwen3-8b-0.55bpw`، 2.76GiB، 4935 tensor) دون أي أوزان وهمية.
- `config.json` يطابق `ModelConfig` في الحقول العشرة الرئيسية؛ 4431 تنسورًا مطلوبًا موجودة و لا مفقود؛ جداول التضمين/rأس اللغة BF16 بالأبعاد الصحيحة.
- تحققت صحة ترتيب LittleBit-2 (المسار الأساسي + المتبقي) بمقارنة C++ مع مرجع NumPy مستقل: خطأ نسبي ~1.1e-6.
- مقارنة لوغاريتمات كاملة على الـ36 طبقة مقابل مرجع مبني على `transformers v4.51.3`: `max_rel=3.56e-6`، نفس top-1 وتطابق top-5 بالكامل.
- Tokenizer: تحميل `tokenizer.json` الرسمي (يحتوي 96 دمجًا ناقصة في `merges.txt`)، تطبيق pre-tokenizer الخاص بـ Qwen3 ورموز `added_tokens`؛ 11/11 عينة تطابقت مع مكتبة Hugging Face `tokenizers` (عربي/إنجليزي/أرقام/رموز/قالب محادثة).
- توليد حقيقي ناجح على النموذج الكامل مع لوغاريتمات منتهية، مع توثيق عائق الأداء (`ttft≈158s`، `0.015 tok/s` على CPU اللينكس).
- ميزانية الذاكرة: 288KiB/token للـ KV، 144MiB عند 512 و288MiB عند 1024؛ RSS بعد التحميل ~1011MiB وبعد التوليد ~2330MiB؛ لا افتراض بقدرة iPhone 13 على أي حجم.
- Xcode: تحقق إنشائي (`scripts/check_xcode_project.py`)؛ Deployment iOS 16، arm64، bridging header، Automatic signing مع إمكانية تمرير `DEVELOPMENT_TEAM`.
- توثيق كامل في `docs/phase8-real-model-validation.md`. لم يُبدأ أي نقل منصات قبل إثبات صحة المحرك.

### المرحلة التاسعة — Profiling (بدون أي تحسين بعد)
- أُضيف `tests/perf_benchmark.cpp` و`scripts/benchmark_phase9.py` كـ benchmark قابل لإعادة التشغيل على الأوزان الحقيقية؛ يُقس TTFT وPrefill وDecode ولكل طبقة ومكوّن، ويتحقق أن نسخة القياس تعطي لوغاريتمات المحرك نفسها (`max_abs=0` على 14 forward).
- خط الأساس على 4 vCPU: TTFT=142.6s لـ11 رمزًا، decode=13.39s/token، خيط واحد فقط (99.7% من نواة ≈24.9% من الآلة).
- أكبر ثلاثة اختناقات مقاسة: MLP binary projections 9.87s/forward (75.5%)، attention projections 2.68s/forward (20.5%)، `lm_head` 0.52s/forward (4.0%). تصحّح فرضية المرحلة الثامنة التي كانت تتهم `lm_head` وحده.
- Bandwidth فعلي: 0.039 GB/s للطبقات الثنائية (حلقة لكل بت) مقابل 2.39 GB/s لـ`lm_head`؛ AVX2/AVX512 متوفرة لكن غير مستخدمة.
- ترتيب الحلول المقترحة قبل التنفيذ: (1) تحسين نواة `binary_matvec_mult`، (2) multithreading على صفوف المصفوفة، (3) SIMD لـ`lm_head`، ثم تأجيل GQA/KV وtop-p لانخفاض عائدهما الحالي.
- التوثيق: `docs/phase9-profiling-baseline.md` و`docs/phase9-profiles/baseline.{json,md}`. لم يُغيّر أي كود محرك، ولم تبدأ المرحلة الثانية بعد.

### نتائج الاختبارات (100% نجاح لكافة الاختبارات الثمانية):
1. `test_binary_packer`: نجح.
2. `test_littlebit_linear`: نجح.
3. `test_safetensors`: نجح.
4. `test_tokenizer`: نجح.
5. `test_ops`: نجح.
6. `test_transformer_block`: نجح.
7. `test_model_inference`: نجح (مسار استدلال كامل + تدفق + إيقاف مبكر).
8. `check_xcode_project`: نجح (فحص إنشائي فقط؛ لا يعني بناءً على macOS).
