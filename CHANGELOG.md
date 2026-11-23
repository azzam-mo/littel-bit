# Changelog — LittleBit Loader

## [Unreleased] - 2026-11-22

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

### نتائج الاختبارات (100% نجاح لكافة الاختبارات الستة):
1. `test_binary_packer`: نجح.
2. `test_littlebit_linear`: نجح.
3. `test_safetensors`: نجح.
4. `test_tokenizer`: نجح.
5. `test_ops`: نجح.
6. `test_transformer_block`: نجح.
