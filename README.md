# 🤖 Intelligent Transformer-Based Chatbot

An interactive **Streamlit** chatbot built on a **pre-trained decoder-only Transformer**
(default: `Qwen/Qwen2.5-0.5B-Instruct`). The goal is not only to chat, but to make every
stage of a Transformer visible: **tokenization → embeddings → self-attention → next-token
prediction → sampling → conversation memory**.

> ⚡ **GPU only.** The application is designed to run exclusively on a CUDA GPU (weights are
> loaded in `bfloat16`). If no GPU is detected the app stops with an explanatory error.

---

## 1. Features ↔ project requirements

| # | Requirement | Where in the app | Implementation |
|---|-------------|------------------|----------------|
| 1 | **Tokenization** – tokens, token IDs, explanation | 🔤 *Tokenization* tab | `chatbot/tokenization.py` – coloured token chips with IDs, table, round-trip check, peek at the embedding vectors, explanation of text → tokens → IDs → embeddings |
| 2 | **Transformer model info** – layers, heads, embedding dim, vocab | 🧠 *Model Info* tab | `chatbot/model_manager.py` reads `model.config` (layers, heads, hidden size, vocab, params, context length, FFN size, KV-heads, …) and lists the modules of one decoder block |
| 3 | **Text generation + next-token prediction** | 💬 *Chat* + 🎯 *Next-Token Prediction* tabs | `chatbot/generation.py` – a hand-written generation loop (no `model.generate`) that records, at **every step**, the chosen token, its probability and the top alternatives. Top-10 distribution charts, temperature comparison, top-k walk-through |
| 4 | **Self-attention visualisation** | 🔍 *Self-Attention* tab | `chatbot/attention.py` – forward pass with `output_attentions=True`; heat-map per layer/head (or mean of heads), "where does token X look" bar chart, per-head entropy table, the attention formula |
| 5 | **Generation parameters** – temperature, top-k, max tokens (+ top-p, repetition penalty) | Sidebar + 🧪 *Parameter Lab* tab | Sliders apply instantly; the Lab runs the **same prompt** with several values side by side and computes confidence / diversity metrics with a written analysis |
| 6 | **Conversation memory** | 💬 *Chat* tab, sidebar toggle | `chatbot/memory.py` – the last *N* (question, answer) pairs are rendered with the model's chat template into the prompt. The exact prompt is shown in the *"What the model actually saw"* panel. Memory can be switched **off** to demonstrate the difference |
| 7 | **Streamlit interface** – chat, input, send button, history, model info, settings | whole app | `app.py` + `ui/*.py` |
| 8 | **Dashboard** – questions, generated tokens, avg response time, parameters, turns | 📊 *Dashboard* tab | `chatbot/stats.py` + charts (response time, tokens, confidence vs temperature) |
| – | **System architecture** | 🏗️ *Architecture* tab | Pipeline diagram drawn in Plotly (filled with the loaded model's numbers) + step-by-step pipeline + technical Q&A |

---

## 2. Installation

```bash
# 1) CUDA build of PyTorch (pick the index matching your driver: cu126 / cu128 / cu130)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 2) the rest
pip install -r requirements.txt
```

Verify the GPU is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 3. Run

The project ships in **two equivalent versions**:

| Version | Entry point | When to use |
|---------|-------------|-------------|
| **Modular** (source of truth) | `streamlit run app.py` | development, reading the code module by module (`chatbot/` = logic, `ui/` = tabs) |
| **Single file** | `streamlit run single_file/transformer_chatbot_app.py` | submission / running from one file; generated from the modular code by `python tools/build_single_file.py` |

```bash
streamlit run app.py
```
or double-click `run.bat` (it checks for a GPU first). The first start downloads the model
weights (~1 GB for Qwen2.5-0.5B-Instruct) into the Hugging Face cache.

Optional core smoke-test without the UI:

```bash
python test_core.py
```

---

## 4. Project structure

```
Transformer_Chatbot/
├── app.py                    Streamlit entry point (sidebar + 8 tabs)
├── run.bat                   Windows launcher (GPU check + streamlit run)
├── test_core.py              smoke test of the core package
├── requirements.txt
├── chatbot/                  core logic – independent of Streamlit
│   ├── config.py             model registry, default parameters, prompts
│   ├── model_manager.py      GPU-only loading, model/GPU information
│   ├── tokenization.py       text → tokens → IDs, coloured chips, embedding preview
│   ├── generation.py         logit processing (rep. penalty, temperature, top-k, top-p),
│   │                         streaming generation loop with next-token trace
│   ├── attention.py          attention extraction + plotly heat-maps
│   ├── memory.py             conversation history → prompt (chat template / plain)
│   └── stats.py              dashboard statistics
├── ui/                       one module per tab
│   ├── sidebar.py  chat_tab.py  tokenization_tab.py  model_tab.py
│   ├── next_token_tab.py  attention_tab.py  lab_tab.py  dashboard_tab.py
│   └── architecture_tab.py  state.py
├── single_file/
│   └── transformer_chatbot_app.py   the whole application in ONE file (generated)
├── tools/
│   └── build_single_file.py         regenerates the single-file version from the modules
└── docs/
    └── PRESENTATION_GUIDE.md demo script + technical Q&A (English / العربية)
```

---

## 5. System architecture

```
 User question ─┐
                ├─► Prompt builder ─► Tokenizer ─► Embeddings + positions
 Chat memory  ──┘   (chat template,   (BPE)          │
   ▲                 last N turns)                    ▼
   │                                    L × [ masked multi-head self-attention → FFN ]
   │                                                  │
   │                                                  ▼
   │                                   LM head → logits (one per vocab token)
   │                                                  │
   │                     repetition penalty → temperature → top-k → top-p → softmax → sample
   │                                                  │
   │                    append token & repeat (KV-cache) ──┘   until EOS / max tokens
   │                                                  │
   └───────── store turn ◄── Detokenizer ◄────────────┘ ─► streamed answer + dashboard
```

1. **Memory** – every turn is kept in `st.session_state.messages`.
2. **Prompt builder** – the last *N* turns + new question are rendered with the tokenizer's
   chat template (`system / user / assistant`) and trimmed to the context budget.
3. **Tokenizer** – Byte-Pair Encoding: text → sub-words → integer IDs.
4. **Transformer** – IDs → embedding vectors → *L* decoder blocks (self-attention + feed-forward)
   → LM head → a logit for each of the *V* vocabulary tokens.
5. **Sampler** – the logits are reshaped by the generation parameters and one token is drawn.
   Every step is stored in the *next-token trace*.
6. **Loop** – the chosen token is fed back (using the KV-cache) until the end-of-sequence
   token or `max_new_tokens`.
7. **Detokenizer** – IDs → text, streamed to the chat; statistics go to the dashboard.

---

## 6. Generation parameters – what we observed

| Parameter | Low value | High value | Effect |
|-----------|-----------|------------|--------|
| **Temperature** `softmax(logits / T)` | `0` → greedy: deterministic, repeatable, sometimes repetitive | `1.5+` → nearly flat distribution: creative, then incoherent | Controls *randomness*. Mean confidence of chosen tokens falls and distinct-token ratio rises as T grows |
| **Top-K** | `1` → identical to greedy | `100+` → practically no filtering | Hard cut-off of the candidate set. Prevents the long tail of nonsense tokens; too small ⇒ dull answers |
| **Top-P** | `0.3` → very conservative | `1.0` → disabled | Adaptive cut-off: keeps 1 token when the model is sure and many when it is unsure |
| **Max new tokens** | short → answers cut mid-sentence (`stopped by: max_tokens`) | long → only costs time | Length limit, does not change *which* tokens are chosen |
| **Repetition penalty** | `1.0` → off | `1.5` → strongly avoids repeated tokens | Divides the logits of tokens already generated |

Use the 🧪 *Parameter Lab* to reproduce these observations with your own prompt.

---

## 7. Supported models

| Model | Chat template | Size | Notes |
|-------|---------------|------|-------|
| `Qwen/Qwen2.5-0.5B-Instruct` (default) | ✅ | ~1 GB | best quality / speed trade-off |
| `HuggingFaceTB/SmolLM2-360M-Instruct` | ✅ | ~0.7 GB | fast |
| `HuggingFaceTB/SmolLM2-135M-Instruct` | ✅ | ~0.3 GB | fastest |
| `Qwen/Qwen2.5-1.5B-Instruct` | ✅ | ~3 GB | better answers |
| `gpt2`, `distilgpt2` | ❌ | ~0.3–0.5 GB | classic raw LMs, wrapped in a `User:/AI:` transcript |

All models are loaded with `attn_implementation="eager"` so that attention weights can be returned.
