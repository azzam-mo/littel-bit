# Changelog — LittleBit Loader

## [Unreleased] - 2026-11-22

### المرحلة الأولى: تحليل النموذج والمواصفات
- تحليل نموذج `littlebit-qwen3-8b-0.55bpw` (Qwen3-8B base, 36 layers, 32 Q heads, 8 KV heads, head dim 128).
- فحص بنية ملف `model.safetensors` واستخراج بنية الطبقات (4935 tensor، طبقات ثنائية مضغوطة I32 مع مقاييس BF16).
- دراسة جدوى استهلاك الذاكرة على iPhone 13 (Apple A15 Bionic, 4GB RAM) واعتماد استراتيجية `mmap` مع تفادي فك الضغط المسبق في RAM.

### المرحلة الثانية: إعداد بيئة المشروع والمستودع
- تهيئة مستودع Git محلي بفرع `main`.
- إنشاء نسخة احتياطية محلية كاملة للمشروع (`backups/LittleBit-Loader-main-backup-...tar.gz`).
- تفريع فرع التطوير `dev`.
- إنشاء بنية محرك C++ الأساسية ومكتبات الفحص والـ Unit Tests.

### المرحلة الثالثة والرابعة: تطوير محرك LittleBit Core والـ Tokenizer
- **`binary_packer`**: تطبيق خوارزمية فك وضغط مصفوفات البتات الثنائية (`{-1, +1}`) وحسابات `binary_matvec_mult` المباشرة دون تخصيص ذاكرة وسيطة.
- **`LittleBitLinear`**: محاكاة حسابية متطابقة لطبقة LittleBitLinear تشمل المسار الأساسي (Primary) والمسار المتبقي (Residual) ومقاييس الحجم ($u_1, u_2, v_1, v_2$).
- **`SafeTensorsFile`**: قارئ تنسورات بذاكرة معنونة `mmap` مع تحويل تلقائي من BF16/FP16 إلى FP32 وقراءة مباشرة لهيكل الطبقات.
- **`Tokenizer`**: محرك Byte-Pair Encoding (BPE) مكتمل يدعم تشفير وفك تشفير النصوص الإنجليزية والعربية ومعالجة الرموز الخاصة.
- **Unit Tests**: نجاح كافة اختبارات الوحدة (4 من أصل 4 بنسبة 100%):
  - `test_binary_packer`: نجح.
  - `test_littlebit_linear`: نجح (تطابق حسابي 100% مع المرجع الرياضي).
  - `test_safetensors`: نجح.
  - `test_tokenizer`: نجح (تشفير وفك تشفير كامل باللغتين).
- **Packaging & CI**:
  - تطوير سكربت `scripts/package_ipa.py` و `scripts/package_ipa.sh` لتغليف حزم `.ipa` غير الموقعة على Linux بهيكل `Payload/`.
  - إعداد مسار GitHub Actions (`.github/workflows/ci.yml`) لاختبار وتجميع المحرك على Ubuntu.
