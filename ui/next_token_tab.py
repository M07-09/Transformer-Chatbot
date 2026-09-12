"""🎯 Next-token prediction tab."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from chatbot import (
    GenerationSettings,
    LoadedModel,
    distribution_from_logits,
    next_token_logits,
    process_logits,
    top_candidates,
)

from .chat_tab import trace_dataframe

DEFAULT_PROMPT = "The capital of France is"


def _bar(cands, title: str, color: str = "#636EFA") -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=[c.prob for c in cands][::-1],
            y=[repr(c.text) for c in cands][::-1],
            orientation="h",
            marker_color=color,
            text=[f"{c.prob:.1%}" for c in cands][::-1],
            textposition="outside",
            hovertext=[f"id {c.token_id}  p = {c.prob:.3%}" for c in cands][::-1],
            hoverinfo="y+text",
            name="probability",
        )
    )
    fig.update_layout(title=title, xaxis_title="probability", height=360,
                      margin=dict(l=10, r=40, t=50, b=30), xaxis=dict(range=[0, 1.05]))
    return fig


def render(lm: LoadedModel, settings: GenerationSettings) -> None:
    st.subheader("🎯 Next-token prediction: the only thing a language model does")
    st.markdown(
        "A decoder-only Transformer never *writes a sentence*. It computes, for the current "
        "context, a probability for **every token in the vocabulary**, one token is chosen, it is "
        "appended to the context, and the process repeats. Try it:"
    )

    text = st.text_input("Context (the model predicts what comes next)", value=DEFAULT_PROMPT, key="nt_text")
    if not text.strip():
        return

    ids = lm.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    logits = next_token_logits(lm, ids)                 # [vocab]

    # ---------- top-10 raw distribution ----------
    col_a, col_b = st.columns([1, 1])
    with col_a:
        cands = distribution_from_logits(lm.tokenizer, logits, temperature=1.0, top_n=10)
        st.plotly_chart(_bar(cands, "Top-10 next tokens  (softmax of raw logits, T = 1)"), width="stretch")
    with col_b:
        st.markdown("**Same logits, with the current sidebar settings applied**")
        probs = process_logits(logits, settings, [])
        surv = int((probs > 0).sum())
        kept = top_candidates(lm.tokenizer, probs, min(10, surv))
        st.plotly_chart(
            _bar(kept, f"T={settings.temperature}, top-k={settings.top_k}, top-p={settings.top_p} → "
                       f"{surv:,} of {logits.numel():,} tokens can be sampled", "#EF553B"),
            width="stretch",
        )

    # ---------- temperature comparison ----------
    st.markdown("#### 🌡️ Effect of temperature on the same logits")
    st.caption("softmax(logits / T): low T sharpens the peak (deterministic), high T flattens it (random).")
    temps = [0.3, 1.0, 2.0]
    cols = st.columns(len(temps))
    for c, T in zip(cols, temps):
        d = distribution_from_logits(lm.tokenizer, logits, temperature=T, top_n=8)
        with c:
            st.plotly_chart(_bar(d, f"T = {T}   (top-1 = {d[0].prob:.0%})"), width="stretch")

    # ---------- top-k walk-through ----------
    st.markdown("#### 🔪 Effect of Top-K")
    k_demo = st.slider("Top-K to preview", 1, 50, 5, key="nt_k")
    d = distribution_from_logits(lm.tokenizer, logits, temperature=1.0, top_n=50)
    mass = sum(c.prob for c in d[:k_demo])
    st.markdown(
        f"With **K = {k_demo}**, sampling is restricted to the tokens "
        + ", ".join(f"`{c.text!r}`" for c in d[:k_demo])
        + f" which together hold **{mass:.1%}** of the probability mass; all other "
          f"{logits.numel() - k_demo:,} tokens get probability 0."
    )

    # ---------- last answer trace ----------
    st.markdown("#### 🧾 Trace of the last chat answer")
    result = st.session_state.last_result
    if result is None:
        st.caption("Ask something in the Chat tab first.")
        return
    st.dataframe(trace_dataframe(result), width="stretch", hide_index=True, height=360)

    # probability of chosen token along the answer
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=[s.raw_prob for s in result.trace], mode="lines+markers",
                             name="raw P(chosen)", line=dict(color="#636EFA")))
    fig.add_trace(go.Scatter(y=[s.chosen.prob for s in result.trace], mode="lines+markers",
                             name="P(chosen) after T / top-k / top-p", line=dict(color="#EF553B")))
    fig.update_layout(title="How confident was the model at each generated token?",
                      xaxis_title="generation step", yaxis_title="probability",
                      height=320, margin=dict(l=10, r=10, t=50, b=30), yaxis=dict(range=[0, 1.02]))
    st.plotly_chart(fig, width="stretch")
