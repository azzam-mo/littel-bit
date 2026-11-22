# LittleBit Loader (iOS & Linux Inference Engine)

**LittleBit Loader** هو محرك استدلال عالي الكفاءة مخصص لتشغيل نماذج اللغة ذات الضغط الفائق دون 1 بت (Sub-1-Bit LLMs)، وبالأخص عائلة **LittleBit-2** (0.55 BPW) المبنية على معمارية Qwen3-8B، مع الاستهداف الأساسي لأجهزة iPhone 13 (Apple A15 Bionic, 4GB RAM) ومحطة التطوير على Linux.

## معمارية المحرك

- **C++ Inference Engine:**
  - `safetensors` zero-copy parser & memory-mapped I/O (`mmap`).
  - فك ضغط وحسابات LittleBit-2 الثنائية (`{-1, +1}`) المحسّنة بدون استهلاك ذاكرة مسبق.
  - دعم مسارات التقريب الأساسية والمتبقية (`Primary` + `Residual Paths`).
  - RMSNorm، SiLU Activation، RoPE، وGrouped-Query Attention (GQA).
  - إدارة ديناميكية للـ KV Cache لتفادي تجاوز سقف الذاكرة (Jetsam Memory Limit).
- **iOS UI Layer (SwiftUI / C++ Bridge):**
  - Model Manager (استيراد وإدارة ملفات النماذج والتحقق من التوافق).
  - Chat Interface مع توليد تدريجي متدفق (Token Streaming).
  - Performance & Memory Monitor (قياس الذاكرة الحية وTokens/sec وزمن الاستجابة).
- **CI / Packaging:**
  - GitHub Actions على Ubuntu لاختبارات C++ وبناء حزم IPA غير الموقعة دون الحاجة إلى macOS runners.
