# المرحلة الثامنة — Real Model Validation

## 0. مصدر النموذج المستخدم

لم يُنشأ أي نموذج وهمي للتحقق. استُخدم ملف الأوزان الحقيقي الموجود محليًا:

```
/home/mrdp_24193/Documents/deepseek-harness/default-workspace/littel-bit/artifacts/
  config.json          (Qwen3-For-CausalLM / LittleBit-2 0.55bpw)
  model.safetensors    2,968,804,064 bytes ≈ 2.76 GiB
  tokenizer.json       11,422,654 bytes
  tokenizer_config.json
```

ملفات `vocab.json` و`merges.txt` موجودة في مسار منفصل (`~/.gemini/antigravity/scratch/model-configs`) واستُخدمت فقط عند المقارنة مع المسار القديم. لم يُحمّل أي نموذج ضخم جديد؛ فُحصت المساحة والذاكرة أولًا (73GiB قرص متاح، ~9.9GiB ذاكرة متاحة).

## 1. Model Validation

### 1.1 config.json مقابل ModelConfig

`scripts/validate_real_model.py` يقارن القيم الحقيقية بالقيم الافتراضية:

| الحقل | القيمة الحقيقية | الحالة |
|---|---|---|
| model_type | qwen3 | MATCH |
| vocab_size | 151936 | MATCH |
| hidden_size | 4096 | MATCH |
| num_hidden_layers | 36 | MATCH |
| num_attention_heads | 32 | MATCH |
| num_key_value_heads | 8 | MATCH |
| head_dim | 128 | MATCH |
| intermediate_size | 12288 | MATCH |
| rms_norm_eps | 1e-6 | MATCH |
| rope_theta | 1000000 | MATCH |

### 1.2 أسماء وأبعاد التنسورات

```
[config] MATCH
[tensors] 4431 required, missing 0 []
[embeddings] MATCH
```

الرأس يحتوي 4935 tensor + `__metadata__`، منها 252 طبقة خطية LittleBit لكل نمط (أساسي + متبقٍ). أشكال الطبقة 0:

| Linear | U_shape | V_shape | split_dim |
|---|---:|---:|---:|
| q_proj | 4096 | 544 | 544 |
| k_proj | 1024 | 208 | 208 |
| v_proj | 1024 | 208 | 208 |
| o_proj | 4096 | 544 | 544 |
| gate_proj | 12288 | 824 | 824 |
| up_proj | 12288 | 824 | 824 |
| down_proj | 4096 | 824 | 824 |

### 1.3 ترتيب الأوزان والتحويل

مقارنة `model.layers.0.self_attn.q_proj` بين C++ ومرجع NumPy مشتق من `quantization/modules/littlebit.py`:

```
in=4096 out=4096 max_abs=4.29153e-06 rel=1.10562e-06 MATCH
```

الصيغة المؤكدة:

```
y = (((x * v2) @ Vqᵀ) * (v1 * u2)) @ Uqᵀ * u1
    + (((x * v2_R) @ Vq_Rᵀ) * (v1_R * u2_R)) @ Uq_Rᵀ * u1_R
```

حيث `Vq/Uq = 1 - 2*bit` من حزم `I32` (padding = +1 → bit 0). لا يوجد bias في النموذج.

### 1.4 Attention / GQA / RoPE / RMSNorm / MLP

راجعت الكود مقابل `transformers v4.51.3 modeling_qwen3.py` (مصدر رسمي):

- QK-RMSNorm قبل RoPE، على رأس بطول `head_dim` — مطابق.
- `rotate_half` بنمط split-half، `theta=1e6` — مطابق.
- GQA بمجموعة `32/8 = 4` — مطابق.
- SwiGLU: `silu(gate) * up` ثم `down_proj` — مطابق.
- RMSNorm نهائي و`lm_head` غير مشترك (`tie_word_embeddings=false`) — مطابق.

### 1.5 مقارنة اللوغاريتمات (مرجع مستقل)

`scripts/compare_logits.py` ينفّذ forward كاملًا بـ NumPy من الأوزان الحقيقية، ثم يقارنه بمخرجات `tests/logits_probe.cpp`.

```
python3 scripts/compare_logits.py <model_dir> <logits_probe> 151644,9707 36
[logits] layers=36 tokens=2 max_abs=6.10352e-05 max_rel=3.56484e-06 mean_rel=4.15283e-07
[top1] got=11 ref=11 MATCH
[top5] overlap=5/5
RESULT: PASS
```

أيضًا نجح الاختبار بـ 3 طبقات و4 رموز (يختبر إعادة استخدام KV عبر عدة مواضع): `max_rel=3.36731e-06`.

> هذا مرجع مستقل مبني على كود Hugging Face الرسمي، وليس تشغيلًا لـ `transformers` (غير متاح في البيئة). لا أدّعي مطابقة HF Runtime؛ أدّعي مطابقة الحسابات المرجعية المنشورة.

### 1.6 توليد حقيقي

```
real_model_smoke <model_dir> 3 512
[SMOKE] prompt=مرحبا generated=3 tokens=262,220,16,
[SMOKE] decoded='    1'
[SMOKE] load=1124.46 ms ttft=158208 ms rate=0.0151244 tok/s
        kv=144 MB rss_before=1137 MB rss_after=2330 MB
All real-model smoke checks passed.
```

المخرجات ليست "إجابة ذكية"؛ الهدف إثبات أن المحرك يمرّر الأوزان الحقيقية عبر كل الطبقات وينتج لوغاريتمات منتهية. النص الضعيف متوقع ل checkpoint مبكر (0.5 epoch حسب README الخاص بالنموذج).

## 2. Tokenizer

### 2.1 الإصلاحات

- `Tokenizer::load_from_json` أصبح مسار التحميل المفضل في `Qwen3Model::load`؛ يقرأ `model.vocab` و`model.merges` و`added_tokens` من `tokenizer.json`.
- `merges.txt` كان ينقصه **96 دمجًا** موجودة في `tokenizer.json` (أبرزها رموز تبدأ بـ `#`)؛ التحميل من JSON أزال هذا الفارق.
- طُبّق pre-tokenizer الخاص بـ Qwen3:
  `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
- جداول فئات Unicode (L/N) مولّدة في `include/littlebit/unicode_ranges.h` من Unicode 16، بدون ICU/اعتماد خارجي على iOS.
- الرموز الخاصة تُمرّر حرفيًا من `added_tokens` بدل قائمة ثابتة.

### 2.2 المقارنة مع Hugging Face

`scripts/validate_tokenizer.py` يستخدم مكتبة `tokenizers` الرسمية مع `tokenizer.json` نفسه:

```
11 / 11 عينة MATCH — RESULT: PASS
```

العينات: الإنجليزية، العربية، Arabic-Indic digits، أرقام/تواريخ، رموز، نصوص مختلطة، رموز تحكم، و3 قوالب محادثة كاملة.

**ملاحظة متبقية:** لا يوجد تطبيع NFC في C++ (`tokenizer.json` يحدّد `normalizer: NFC`). العينات العربية الشائعة تطابقت، لكن النصوص المفككة (decomposed) قد تختلف؛ تحتاج اختبارًا إضافيًا عند توفر حالات استخدام فعلية.

## 3. Memory

### 3.1 قياسات فعلية على Linux

```
[file] 2.76 GiB, header 536,328 bytes, tensors 4,935
  BF16: 2386.1 MiB   (أغلبها embed_tokens + lm_head: ~1.19 GiB لكل جدول)
  I32:  444.7 MiB    (الأوزان المكمّمة derivate تُنسخ إلى RAM)
RSS بعد تحميل كامل: 1011 MiB
RSS بعد التوليد:    2330 MiB (Peak 2385 MiB)
```

### 3.2 KV Cache

```
288 KiB per token = 36 layers × 2 (K,V) × 8 KV heads × 128 × 4 bytes
```

| السياق | KV |
|---:|---:|
| 512 | 144 MiB |
| 1024 | 288 MiB |
| 2048 | 576 MiB |
| 4096 | 1152 MiB |
| 8192 | 2304 MiB |
| 40960 (config) | 11520 MiB |

`config.json` يسمح بـ 40960 token، لكن المحرك يستخدم افتراضيًا 1024. **لا يجوز افتراض أن iPhone 13 يستطيع حمل أي حجم**؛ 288 MiB KV + ~1GiB أوزان anonymous + صفحات mmap التي لمستها حلقة `lm_head` قد تتجاوز ميزانية التطبيق بسهولة. يجب قياس Jetsam على الجهاز قبل تحديد `maxSeqLen` النهائي.

### 3.3 عائق أداء فعلي

على CPU اللينكس: `ttft ≈ 158s` و`≈0.015 tok/s` للنموذج الكامل. السبب الرئيسي مسح `lm_head` بالكامل (151936 × 4096 BF16) لكل token، إضافة إلى غياب NEON/Accelerate. هذا ليس فشلًا صحيحًا (correctness)، لكنه عائق أداء يجب معالجته قبل تجربة iOS مريحة. لم أبدأ أي نقل_platform أو تحسين غير مثبت.

## 4. iOS

- `scripts/generate_xcode_project.py` أُعيد توليده بعد المراجعة؛ `scripts/check_xcode_project.py` يعطي:
  `[xcode] PASS: structure, sources, deployment target and bridging header`
- Deployment Target: iOS 16.0؛ `ARCHS[sdk=iphoneos*] = arm64`؛ لا يوجد `-march=native`.
- Signing: `CODE_SIGN_STYLE = Automatic`، لكن `DEVELOPMENT_TEAM` غير مضمّن لغياب حساب Apple. يمكن تمريره:
  `DEVELOPMENT_TEAM=XXXXXXXXXX python3 scripts/generate_xcode_project.py`
  أو اختياره يدويًا في Xcode.
- Info.plist يفعّل `UIFileSharingEnabled` و`LSSupportsOpeningDocumentsInPlace` ويتطلب `arm64`.
- **لم يتم فتح أو بناء أو توقيع المشروع على macOS**: لا Xcode ولا جهاز في هذه البيئة. لا ادعاء بنجاح iOS.

### اختبارات الجهاز الحقيقي المطلوبة

1. `xcodebuild test` لهدف `LittleBitLoaderTests` على iPhone (XCTest bridge lifecycle).
2. استيراد مجلد النموذج من Files والتحقق من بقاء security scope أثناء mmap.
3. توليد عربي/إنجليزي مع التحقق من سلامة UTF-8 في التدفق.
4. قياس Jetsam/الذاكرة عند `maxSeqLen` 512 ثم 1024، ومراقبةMemory Warning.
5. إيقاف التوليد وتفريغ النموذج أثناء الاستدلال.
6. تعبئة IPA موقّع وتثبيته على الجهاز.

## 5. Cross-Platform Preparation

قابل للفصل فعلًا بدون أي تعديل: `include/littlebit/`, `src/`, `tests/`, `scripts/*` (محرك C++17 + POSIX mmap + أدوات validates). خاص بـ iOS فقط: `ios/` (SwiftUI، الجسر Objective-C++) و`LittleBitLoader.xcodeproj`. لم يبدأ أي نقل إلى منصات أخرى قبل إثبات صحة المحرك، كما هو مطلوب.

## 6. النتيجة

**تم التحقق فعليًا على Linux:** config، 4431 tensor، ترتيب LittleBit-2، طبقة حقيقية، GQA/RoPE/RMSNorm/MLP، لوغاريتمات كاملة 36 طبقة، tokenizer مقابل HF، توليد حقيقي، ميزانية الذاكرة، سلامة مشروع Xcode إنشائيًا.

**عوائق فعلية:** لا Xcode/جهاز Apple؛ لا HF PyTorch runtime محلي؛ أداء CPU بطيء جدًا؛ تطبيع NFC غير منفذ؛ استهلاك الذاكرة عند السياقات الطويلة يحتاج قياسًا على الجهاز.

لم تُلمس إعدادات API أو مزودي النماذج (بما فيهم DeepSeek).
