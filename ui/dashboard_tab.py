"""📊 Dashboard tab: session statistics."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from chatbot import GenerationSettings, LoadedModel, SessionStats


def render(lm: LoadedModel, settings: GenerationSettings, memory: dict) -> None:
    st.subheader("📊 Chatbot dashboard")
    stats: SessionStats = st.session_state.stats
    n_msgs = len(st.session_state.messages)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Questions asked", stats.questions_asked)
    c2.metric("Generated tokens", f"{stats.generated_tokens:,}")
    c3.metric("Avg response time", f"{stats.avg_response_time:.2f} s")
    c4.metric("Conversation turns", n_msgs // 2, help="one turn = one question + one answer")
    c5.metric("Avg speed", f"{stats.avg_tokens_per_second:.1f} tok/s")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### ⚙️ Current generation parameters")
        params = settings.as_dict()
        params.update({
            "conversation memory": "ON" if memory["use_memory"] else "OFF",
            "past turns remembered": memory["max_turns"] if memory["use_memory"] else 0,
            "model": lm.name,
            "device / dtype": f"{lm.device} / {lm.dtype}",
        })
        st.table(pd.DataFrame({"parameter": list(params.keys()), "value": [str(v) for v in params.values()]}))

    with right:
        st.markdown("#### 🗂️ Session totals")
        st.table(pd.DataFrame({
            "metric": ["Messages in history", "Prompt tokens sent (total)", "Generated tokens (total)",
                       "Total generation time"],
            "value": [str(n_msgs), f"{stats.prompt_tokens:,}", f"{stats.generated_tokens:,}",
                      f"{sum(t.seconds for t in stats.turns):.2f} s"],
        }))

    if not stats.turns:
        st.info("No questions yet — go to the Chat tab and ask something.")
        return

    df = pd.DataFrame([t.__dict__ for t in stats.turns])
    x = df["question_no"]

    g1, g2 = st.columns(2)
    with g1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=df["seconds"], mode="lines+markers", name="response time (s)",
                                 line=dict(color="#636EFA")))
        fig.update_layout(title="Response time per question", xaxis_title="question #", yaxis_title="seconds",
                          height=300, margin=dict(l=10, r=10, t=40, b=30))
        st.plotly_chart(fig, width="stretch")
    with g2:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=x, y=df["generated_tokens"], name="generated", marker_color="#00CC96"))
        fig.add_trace(go.Bar(x=x, y=df["prompt_tokens"], name="prompt (context)", marker_color="#AB63FA"))
        fig.update_layout(title="Tokens per question", barmode="group", xaxis_title="question #",
                          yaxis_title="tokens", height=300, margin=dict(l=10, r=10, t=40, b=30))
        st.plotly_chart(fig, width="stretch")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=df["mean_confidence"], mode="lines+markers", name="mean confidence",
                             line=dict(color="#EF553B")))
    fig.add_trace(go.Scatter(x=x, y=df["temperature"], mode="lines+markers", name="temperature used",
                             line=dict(color="#FFA15A", dash="dot")))
    fig.update_layout(title="Model confidence vs. temperature per question", xaxis_title="question #",
                      height=300, margin=dict(l=10, r=10, t=40, b=30))
    st.plotly_chart(fig, width="stretch")

    with st.expander("📋 Per-question log"):
        st.dataframe(df.rename(columns={
            "question_no": "#", "prompt_tokens": "prompt tokens", "generated_tokens": "generated tokens",
            "seconds": "seconds", "memory_turns": "past turns used", "mean_confidence": "mean confidence",
        }).round(3), width="stretch", hide_index=True)
