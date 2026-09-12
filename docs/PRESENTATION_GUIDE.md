# 🎤 Presentation Guide / دليل العرض

This document is a **step-by-step demo script** followed by **technical Q&A** you may be asked.
Arabic explanations are given under each step.

---

## Part 1 – Live demo script (≈ 10 minutes)

### 0. Start
```
streamlit run app.py
```
Sidebar shows the loaded model and the GPU (name, CUDA version, VRAM).

> 🇸🇦 نبدأ بتشغيل التطبيق، ونشير إلى الـ Sidebar: الموديل المحمَّل + كرت الشاشة + استهلاك الذاكرة. المشروع يعمل حصراً على GPU بصيغة bfloat16.

### 1. Ask several questions & demonstrate memory (💬 Chat tab)
1. Click **"What is Machine Learning?"** → answer streams token-by-token.
2. Click **"What are its main types?"** → the model resolves *"its"* thanks to memory.
3. Open **"Full prompt sent to the Transformer"** on the right → show that the previous
   question and answer are inside the prompt, plus the token count.
4. Turn **OFF** "Use conversation history" in the sidebar, ask *"What are its main types?"* again
   → the model no longer knows what *"its"* refers to. Turn memory back **ON**.

> 🇸🇦 الترانسفورمر نفسه بلا ذاكرة (stateless). الذاكرة نحن نصنعها: نخزّن كل الرسائل في `session_state` ونعيد إرسال آخر N جولات داخل الـ prompt باستخدام الـ chat template. عند إيقاف الذاكرة يرى الموديل السؤال الجديد فقط فلا يفهم كلمة "its".

### 2. Tokens and Token IDs (🔤 Tokenization tab)
* The last question is already filled in. Show the coloured tokens and the IDs under them.
* Point at a split word (e.g. `Token` + `ization`) and at the `␣` marker (leading space).
* Show the *Round-trip* line and the *embedding vectors* expander.

> 🇸🇦 النص → قطع فرعية (sub-words) بطريقة BPE → كل قطعة لها رقم (ID) في القاموس. الشبكة لا ترى حروفاً أبداً، فقط الأرقام. كل ID يختار صفاً من مصفوفة الـ embedding بحجم (vocab × embedding_dim) ليتحول إلى متجه.

### 3. Model information (🧠 Model Info tab)
Read the four required numbers: **layers, heads, embedding dim, vocab size**; then parameters,
context length, head dimension, and the module list of one decoder block.

> 🇸🇦 Qwen2.5-0.5B: 24 طبقة، 14 رأس انتباه، بُعد التضمين 896، قاموس 151,936 رمزاً، ~494 مليون معامل. هذه القيم تُقرأ مباشرة من `model.config` وليست مكتوبة يدوياً.

### 4. Next-token prediction (🎯 tab)
* Context *"The capital of France is"* → top-10 distribution (` Paris` on top).
* Right chart: the same logits after the sidebar's temperature/top-k/top-p.
* Three temperature charts (0.3 / 1.0 / 2.0): the peak flattens as T grows.
* Bottom: the trace of the last chat answer – each row is one generated token with its
  probability and the alternatives that were *not* chosen.

> 🇸🇦 الموديل لا "يكتب جملة"، بل يحسب احتمالاً لكل رمز في القاموس، يُختار رمز واحد، يُضاف للسياق، ثم تتكرر العملية. الجدول يوضح هذا حرفياً لكل خطوة من الإجابة الأخيرة.

### 5. Self-attention (🔍 tab)
* Sentence *"The cat sat on the mat because it was tired."*
* Heat-map: rows = query token, columns = key token, upper triangle empty (**causal mask**).
* Choose the token **it** in *"Focus on one token"* and browse a few layers/heads – some heads
  give weight to **cat**.
* Explain Q, K, V and the formula shown at the bottom.

> 🇸🇦 كل رمز (query) يوزّع أوزاناً على الرموز السابقة (keys) ويأخذ متوسطاً موزوناً من قيمها (values). المثلث العلوي فارغ لأن الرمز لا يُسمح له برؤية المستقبل (causal mask). الرؤوس المختلفة تتعلم علاقات مختلفة.

### 6. Temperature experiment (🧪 Parameter Lab)
* Parameter = **Temperature**, values `0.0, 0.7, 1.5` → **Run**.
* Compare the three outputs: T=0 factual/deterministic, T=0.7 natural, T=1.5 drifts.
* Point at the *mean confidence* ↓ and *distinct-token ratio* ↑ in the chart.

> 🇸🇦 الحرارة تقسم الـ logits قبل الـ softmax. حرارة منخفضة → توزيع حاد → دائماً الرمز الأعلى احتمالاً. حرارة عالية → توزيع مسطح → رموز نادرة تُختار → إبداع ثم هلوسة.

### 7. Top-K experiment (🧪 Parameter Lab)
* Parameter = **Top-K**, values `1, 5, 100` → **Run**.
* K=1 equals greedy; K=100 leaves the choice to temperature.

> 🇸🇦 Top-K يقصّ القائمة ويُبقي أفضل K رمز فقط ثم يعيد التطبيع. K=1 يساوي greedy. القيم الكبيرة لا تُفلتر شيئاً تقريباً. عيبه أنه قصّ ثابت لا يراعي شكل التوزيع، ولهذا يُستخدم Top-P معه.

### 8. Dashboard (📊) and Architecture (🏗️)
* Dashboard: questions asked, generated tokens, average response time, turns, current parameters, charts.
* Architecture: the diagram (numbers come from the live model) + the pipeline list + Q&A expanders.

---

## Part 2 – Technical Q&A

**Q: How does the Transformer generate a response?**
The history is rendered into a prompt and tokenized. IDs → embeddings (+ positions) → *L* decoder
blocks (masked self-attention + FFN) → LM head → one logit per vocabulary token for the last
position → softmax → sampling (temperature / top-k / top-p) → the chosen token is appended and the
model runs again using the KV-cache, until EOS or max tokens → IDs are decoded to text.

> 🇸🇦 يُبنى الـ prompt من التاريخ ويُرمَّز، ثم تمر الأرقام عبر التضمين والطبقات حتى رأس اللغة الذي يعطي logit لكل رمز، softmax يحوّلها لاحتمالات، نختار رمزاً، نضيفه، ونكرر حتى رمز النهاية.

**Q: What is the difference between tokens and token IDs?**
A token is a sub-word string from the vocabulary (`"␣learning"`); its ID is its integer index
(`6832`). The model only consumes IDs; the tokenizer maps both directions.

**Q: Why sub-word tokenization (BPE) instead of words or characters?**
Words → huge vocabulary and no way to handle unseen words. Characters → very long sequences.
Sub-words are the compromise: a fixed 50k–150k vocabulary can spell anything.

**Q: What are Q, K and V?**
Three linear projections of each token vector. Query asks "what am I looking for", Key says
"what do I contain", Value is the information passed on. Weights = softmax(QKᵀ/√d_k).

**Q: Why divide by √d_k?**
Dot products grow with the dimension; without scaling the softmax saturates and gradients vanish.

**Q: What is the causal mask?**
−∞ added to attention scores of future positions, so their softmax weight is exactly 0.
This keeps generation autoregressive.

**Q: Why multiple heads?**
Each head attends in a different sub-space and learns a different relation (previous token,
syntax, co-reference). Their outputs are concatenated and projected.

**Q: What are positional encodings and why are they needed?**
Self-attention is permutation-invariant; positions (learned embeddings in GPT-2, rotary
embeddings – RoPE – in Qwen) inject word order.

**Q: What does the feed-forward network do?**
A per-token MLP (`d → 4d → d`, SwiGLU in Qwen) that transforms the attended information;
much of the model's factual knowledge is stored here.

**Q: Temperature vs Top-K vs Top-P?**
All act on the final distribution only. Temperature rescales logits (sharp ↔ flat), Top-K keeps
the K best tokens, Top-P keeps the smallest set whose cumulative probability ≥ P (adaptive).
T=0 or K=1 both give greedy decoding.

**Q: How does conversation memory work if the model is stateless?**
The app stores turns in `st.session_state` and re-inserts the last *N* turns into every prompt
via the chat template; old turns are dropped when the context budget is exceeded.

**Q: What is the KV-cache?**
Keys/values of previous tokens are stored so each new step processes only the newest token:
O(n) per step instead of O(n²) – essential for interactive speed.

**Q: Why GPU?**
Every generated token is a full forward pass: hundreds of matrix multiplications with
millions of parameters. GPUs execute them in parallel; the project runs in bfloat16 on CUDA.

**Q: Why `attn_implementation="eager"`?**
Fused attention kernels (SDPA / Flash-Attention) never materialise the attention matrix, so
they cannot return it. The eager implementation computes it explicitly and returns it for
the visualisation.

**Q: What is the difference between a chat model and GPT-2 here?**
Qwen2.5-Instruct was fine-tuned on conversations and has a chat template with roles; GPT-2 is a
raw language model, so the app wraps the history in a `User:/AI:` transcript and stops when
the model starts a new `User:` line.

**Q: What is the EOS token?**
A special vocabulary entry the model learned to emit when the answer is finished; generation
stops when it is sampled (`stopped by: eos` in the chat caption).
