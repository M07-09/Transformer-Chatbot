"""🏗️ System architecture (Plotly diagram, pure Python) + technical Q&A."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from chatbot import LoadedModel, get_model_info


def architecture_figure(num_layers: int, num_heads: int, vocab_size: int) -> go.Figure:
    """Pipeline diagram drawn with Plotly shapes + annotations (no DOT / HTML)."""
    # key: (x, y, width, height, title, subtitle, colour)
    boxes = {
        "user":   (1.05, 4.35, 1.8, 0.8, "User question", "", "#FFD6A5"),
        "memory": (1.05, 3.05, 1.8, 0.9, "Conversation memory", "session_state.messages", "#FFD6A5"),
        "prompt": (3.35, 3.7, 2.0, 0.9, "Prompt builder", "chat template, last N turns", "#CAFFBF"),
        "tok":    (5.65, 3.7, 2.0, 0.9, "Tokenizer (BPE)", "text → tokens → IDs", "#9BF6FF"),
        "emb":    (7.85, 3.7, 1.7, 0.9, "Embeddings", "+ positions", "#BDB2FF"),
        "blocks": (10.2, 3.7, 2.4, 1.0, f"{num_layers} × Decoder block",
                   f"self-attention ({num_heads} heads) → FFN", "#BDB2FF"),
        "head":   (10.2, 1.6, 2.4, 0.9, "LM head", f"vector → {vocab_size:,} logits", "#BDB2FF"),
        "samp":   (7.45, 1.6, 2.5, 0.9, "Sampler", "softmax → T → top-k → top-p", "#FFC6FF"),
        "detok":  (4.85, 1.6, 1.8, 0.9, "Detokenizer", "IDs → text", "#9BF6FF"),
        "answer": (2.35, 1.6, 2.0, 0.9, "Streamed answer", "+ dashboard", "#FFD6A5"),
    }
    # (from, to, label, dashed)
    edges = [
        ("user", "prompt", "", False),
        ("memory", "prompt", "", False),
        ("prompt", "tok", "", False),
        ("tok", "emb", "", False),
        ("emb", "blocks", "", False),
        ("blocks", "head", "", False),
        ("head", "samp", "", False),
        ("samp", "detok", "EOS / max tokens", False),
        ("detok", "answer", "", False),
        ("answer", "memory", "store turn", True),
        ("samp", "blocks", "append token, repeat (KV-cache)", True),
    ]

    fig = go.Figure()
    for x, y, w, h, title, subtitle, color in boxes.values():
        fig.add_shape(type="rect", x0=x - w / 2, x1=x + w / 2, y0=y - h / 2, y1=y + h / 2,
                      fillcolor=color, line=dict(color="#555555", width=1))
        fig.add_annotation(x=x, y=y + (0.13 if subtitle else 0), text=title, showarrow=False,
                           font=dict(size=12, color="#111111"))
        if subtitle:
            fig.add_annotation(x=x, y=y - 0.2, text=subtitle, showarrow=False,
                               font=dict(size=10, color="#333333"))

    def _port(src: str, dst: str) -> tuple[float, float, float, float]:
        """Start/end points on the box borders facing each other."""
        sx, sy, sw, sh, *_ = boxes[src]
        dx, dy, dw, dh, *_ = boxes[dst]
        if abs(dx - sx) >= abs(dy - sy):            # mostly horizontal
            sign = 1 if dx > sx else -1
            return sx + sign * sw / 2, sy, dx - sign * dw / 2, dy
        sign = 1 if dy > sy else -1                 # mostly vertical
        return sx, sy + sign * sh / 2, dx, dy - sign * dh / 2

    for src, dst, label, dashed in edges:
        x0, y0, x1, y1 = _port(src, dst)
        if dashed:
            fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1,
                          line=dict(color="#888888", width=1.5, dash="dash"))
            # short solid arrow-head at the end of the dashed line
            fig.add_annotation(x=x1, y=y1, ax=x0 + 0.9 * (x1 - x0), ay=y0 + 0.9 * (y1 - y0),
                               xref="x", yref="y", axref="x", ayref="y", text="", showarrow=True,
                               arrowhead=3, arrowsize=1.3, arrowwidth=1.5, arrowcolor="#888888")
        else:
            fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                               text="", showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=1.6,
                               arrowcolor="#444444")
        if label:
            fig.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2 + 0.18, text=label, showarrow=False,
                               font=dict(size=10, color="#444444"), bgcolor="rgba(255,255,255,0.85)")

    fig.update_xaxes(visible=False, range=[0, 11.6], fixedrange=True)
    fig.update_yaxes(visible=False, range=[0.9, 5.0], fixedrange=True)
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode=False)
    return fig


QA = [
    ("How does the Transformer generate a response?",
     "The chat history is rendered into a prompt and tokenized into IDs. The IDs are embedded, pass through "
     "all decoder blocks, and the LM head returns one logit per vocabulary entry for the *last position*. "
     "Softmax converts the logits to probabilities, the sampler picks one token, the token is appended to the "
     "context and the model runs again (only on the new token thanks to the KV-cache). The loop stops at the "
     "end-of-sequence token or the max-tokens limit; the IDs are decoded back to text."),
    ("What is the role of tokenization?",
     "Neural networks work on numbers. Byte-Pair Encoding splits text into sub-word units from a fixed "
     "vocabulary and maps each to an integer ID. The same vocabulary is used in reverse to turn output IDs into "
     "text. Sub-words let a ~150k vocabulary spell any word, including rare or misspelled ones."),
    ("What is self-attention and why is it needed?",
     "Self-attention lets every token build its representation from *all previous tokens*, weighted by relevance "
     "(softmax(Q·Kᵀ/√d)·V). It is how the model links 'its' to 'Machine Learning' from a previous turn, and it is "
     "fully parallel, unlike an RNN that reads one token at a time."),
    ("Why multiple heads and multiple layers?",
     "Each head learns a different relation (previous word, subject-verb, co-reference…). Stacking layers lets the "
     "model compose these relations into higher-level meaning: lower layers capture syntax, upper layers semantics."),
    ("What is the causal mask?",
     "During training and generation a token must not see the future. The mask sets attention scores to −∞ "
     "for positions after the current one, so the softmax gives them weight 0 – that is the empty upper triangle "
     "in the attention heat-map."),
    ("What do temperature, top-k and top-p change?",
     "Nothing inside the network – they only reshape the final probability distribution before sampling. "
     "Temperature scales the logits (sharper vs flatter), top-k keeps the K best tokens, top-p keeps the smallest "
     "set with cumulative probability P. Greedy decoding (T=0) always takes the arg-max."),
    ("How does conversation memory work?",
     "The Transformer itself is stateless. The app stores every turn in Streamlit's session_state and, for each new "
     "question, re-sends the last N turns inside the prompt (using the model's chat template). Turning memory off "
     "sends only the new question, so follow-ups like 'its main types' can no longer be resolved."),
    ("What is the KV-cache?",
     "Keys and values of already-processed tokens are stored so that each new step only computes attention for the "
     "newest token instead of recomputing the whole sequence. This makes generation O(n) per step instead of O(n²)."),
    ("Why run on the GPU?",
     "Every generated token needs a full forward pass through all layers – hundreds of large matrix multiplications. "
     "GPUs execute these in parallel; this project is configured to run exclusively on CUDA with bfloat16 weights."),
]


def render(lm: LoadedModel) -> None:
    st.subheader("🏗️ System architecture")
    info = get_model_info(lm)
    st.plotly_chart(
        architecture_figure(info["num_layers"], info["num_attention_heads"], info["vocab_size"]),
        width="stretch",
    )

    st.markdown(
        """
#### Pipeline, step by step
1. **Conversation memory** – all previous turns live in `st.session_state.messages`.
2. **Prompt builder** (`chatbot/memory.py`) – the last *N* turns + the new question are rendered with the
   model's chat template (system / user / assistant roles) and trimmed to the context budget.
3. **Tokenizer** (`chatbot/tokenization.py`) – text → sub-word tokens → integer IDs.
4. **Transformer** (`chatbot/model_manager.py`) – embeddings + positional info → *L* decoder blocks
   (masked multi-head self-attention → feed-forward) → LM head → logits for every vocabulary token.
5. **Sampler** (`chatbot/generation.py`) – repetition penalty → temperature → top-k → top-p → softmax → choose
   a token. Every step is recorded (the *next-token trace*).
6. **Loop** – the chosen token is appended and the model is run again with the KV-cache until EOS / max tokens.
7. **Detokenizer** – IDs → text, streamed to the chat window; statistics go to the dashboard.
8. **Attention viewer** (`chatbot/attention.py`) – an extra forward pass with `output_attentions=True`
   exposes the weights of every layer and head for the heat-map.

#### Project layout
```
app.py                 Streamlit entry point (tabs + sidebar)
chatbot/config.py      model registry, default parameters, prompts
chatbot/model_manager  GPU-only loading, architecture facts
chatbot/tokenization   tokens / IDs / embedding preview
chatbot/generation     hand-written sampling loop with trace
chatbot/attention      attention extraction + plotly heat-maps
chatbot/memory         chat history → prompt
chatbot/stats          dashboard statistics
ui/*.py                one module per tab
```
"""
    )

    st.markdown("#### ❓ Technical questions & answers")
    for q, a in QA:
        with st.expander(q):
            st.markdown(a)
