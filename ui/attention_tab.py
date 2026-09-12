"""🔍 Self-attention visualisation tab."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from chatbot import LoadedModel, attention_heatmap, compute_attention, head_summary, token_focus_bar

DEFAULT_TEXT = "The cat sat on the mat because it was tired."


def render(lm: LoadedModel) -> None:
    st.subheader("🔍 Self-attention: how tokens look at each other")
    st.markdown(
        "For every token (a **query**) the model computes a weight for every earlier token (the **keys**). "
        "The weights of one row sum to 1 and say *where this token gathers information from*. "
        "Decoder models are **causal**: a token can only attend to itself and the tokens before it, "
        "which is why the upper-right triangle is always empty."
    )

    text = st.text_input("Sentence to analyse (max 48 tokens)", value=DEFAULT_TEXT, key="att_text")
    if not text.strip():
        return

    cache_key = (lm.name, text)
    cache = st.session_state.attention_cache
    if cache_key not in cache:
        cache.clear()
        with st.spinner("Running a forward pass with output_attentions=True …"):
            cache[cache_key] = compute_attention(lm, text)
    data = cache[cache_key]

    c1, c2 = st.columns([1, 1])
    layer = c1.slider("Layer", 0, data.num_layers - 1, min(data.num_layers - 1, max(0, data.num_layers // 3)))
    head_opts = ["mean of all heads"] + [f"head {h}" for h in range(data.num_heads)]
    head_choice = c2.selectbox("Head", head_opts, index=0)
    head = None if head_choice.startswith("mean") else int(head_choice.split()[-1])

    st.plotly_chart(attention_heatmap(data, layer, head), width="stretch")

    st.markdown("#### 🎯 Focus on one token")
    q = st.selectbox("Query token", list(range(len(data.tokens))),
                     index=len(data.tokens) - 1,
                     format_func=lambda i: f"{i}: {data.tokens[i]}")
    st.plotly_chart(token_focus_bar(data, layer, head, q), width="stretch")

    with st.expander("📊 Which heads in this layer are 'focused' vs 'spread out'?"):
        ent = head_summary(data, layer)
        df = pd.DataFrame({"head": list(range(len(ent))), "mean attention entropy": ent.round(3)})
        df["behaviour"] = ["focused (few tokens)" if e < ent.mean() else "diffuse (many tokens)" for e in ent]
        st.dataframe(df.sort_values("mean attention entropy"), width="stretch", hide_index=True)
        st.caption("Low entropy = the head puts most of its weight on one or two tokens; "
                   "high entropy = it averages over many tokens.")

    st.markdown("---")
    st.markdown("### The maths behind the picture")
    st.markdown(
        "Every token vector **x_i** is projected three times: "
        "**Query** q_i = x_i · W_Q, **Key** k_i = x_i · W_K, **Value** v_i = x_i · W_V."
    )
    st.code("Attention(Q, K, V) = softmax( (Q · Kᵀ) / sqrt(d_k) + M ) · V", language="text")
    st.markdown(
        """
* **Q · Kᵀ** measures how much each query *matches* each key (a similarity score).
* **sqrt(d_k)** keeps the scores in a sane range; **M** is the **causal mask** (−∞ above the diagonal).
* **softmax** turns the scores of one row into the weights you see in the heat-map.
* The output of a token is the **weighted average of the values** — it now contains information from the tokens it attended to.
* This happens in **every head of every layer**; different heads specialise (previous token, syntax, co-reference such as *it → cat*, …).
"""
    )
