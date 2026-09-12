"""🧪 Parameter Lab: run the same prompt with different temperature / top-k values."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from chatbot import GenerationSettings, LoadedModel, build_prompt, generate

DEFAULT_PROMPT = "Explain in two sentences what a neural network is."

EXPERIMENTS = {
    "Temperature": {"field": "temperature", "values": "0.0, 0.7, 1.5", "cast": float},
    "Top-K": {"field": "top_k", "values": "1, 5, 100", "cast": int},
    "Top-P": {"field": "top_p", "values": "0.3, 0.9, 1.0", "cast": float},
    "Max new tokens": {"field": "max_new_tokens", "values": "20, 60, 150", "cast": int},
}

EXPLANATIONS = {
    "temperature": """
**Temperature** divides the logits before the softmax: `p = softmax(logits / T)`.
* **T → 0** – the highest logit dominates completely → *greedy decoding*: deterministic, factual, but can be repetitive.
* **T ≈ 0.7** – the distribution keeps its shape but small differences still matter → fluent yet varied.
* **T ≥ 1.5** – the distribution becomes almost flat → rare tokens get picked, sentences drift or become nonsense.
Watch the **mean confidence** column drop and the **distinct-token ratio** rise as T grows.
""",
    "top_k": """
**Top-K** keeps only the K most probable tokens and re-normalises.
* **K = 1** – identical to greedy decoding (only the arg-max survives).
* **K = 5–50** – blocks the long tail of unlikely tokens while still allowing variety.
* **K = 100+** – almost no filtering; the result depends mostly on the temperature.
Top-K is a *hard* cut-off: it ignores how the probability is shaped, so with a very peaked distribution K=50 may include tokens of probability 0.001 %.
""",
    "top_p": """
**Top-P (nucleus sampling)** keeps the smallest set of tokens whose cumulative probability reaches P.
Unlike Top-K the number of kept tokens **adapts** to the situation: 1 token when the model is sure, many when it is unsure.
* **P = 0.3** – very conservative, close to greedy.
* **P = 0.9** – the common default.
* **P = 1.0** – disabled.
""",
    "max_new_tokens": """
**Max new tokens** is a hard upper bound on the generation loop. It does not change *which* token is picked,
only *how many* steps are run – unless the model emits the end-of-sequence token first.
Short limits cut answers mid-sentence (`stopped by: max_tokens`); long limits only cost time.
""",
}


def _distinct_ratio(ids: list[int]) -> float:
    return len(set(ids)) / len(ids) if ids else 0.0


def render(lm: LoadedModel, settings: GenerationSettings) -> None:
    st.subheader("🧪 Parameter Lab: same prompt, different generation parameters")
    st.caption("Runs use the current sidebar settings as a base, and only the chosen parameter is changed. "
               "Conversation memory is NOT used here so the comparison is fair.")

    prompt = st.text_area("Prompt", value=DEFAULT_PROMPT, height=70, key="lab_prompt")
    c1, c2 = st.columns([1, 2])
    exp_name = c1.selectbox("Parameter to vary", list(EXPERIMENTS.keys()))
    exp = EXPERIMENTS[exp_name]
    values_str = c2.text_input("Values to try (comma separated)", value=exp["values"], key=f"lab_vals_{exp_name}")

    try:
        values = [exp["cast"](v.strip()) for v in values_str.split(",") if v.strip()]
    except ValueError:
        st.error("Could not parse the values.")
        return
    if not values:
        return

    if st.button("▶️ Run experiment", type="primary"):
        built = build_prompt(lm, [{"role": "user", "content": prompt}], use_memory=False,
                             reserve_tokens=max(settings.max_new_tokens, max(
                                 (v for v in values if exp["field"] == "max_new_tokens"), default=0)))
        results = []
        prog = st.progress(0.0, text="Generating …")
        for i, v in enumerate(values):
            s = replace(settings, **{exp["field"]: v})
            r = generate(lm, built.input_ids, s, built.stop_strings)
            results.append((v, r))
            prog.progress((i + 1) / len(values), text=f"Generated {i + 1}/{len(values)}")
        prog.empty()
        st.session_state.lab_results = {"param": exp["field"], "label": exp_name, "prompt": prompt, "runs": results}

    res = st.session_state.lab_results
    if not res:
        return

    st.markdown(f"#### Results — varying **{res['label']}** for: *{res['prompt']}*")
    cols = st.columns(len(res["runs"]))
    for c, (v, r) in zip(cols, res["runs"]):
        with c:
            st.markdown(f"**{res['label']} = {v}**")
            st.info(r.text or "(empty)")
            st.caption(
                f"{r.generated_tokens} tokens · {r.elapsed:.2f}s · "
                f"confidence {r.mean_confidence:.0%} · distinct {_distinct_ratio(r.new_token_ids):.0%} · "
                f"stop: {r.finished_by}"
            )

    # ---- comparison table + chart ----
    df = pd.DataFrame(
        [
            {
                res["label"]: v,
                "tokens": r.generated_tokens,
                "seconds": round(r.elapsed, 2),
                "mean confidence": round(r.mean_confidence, 3),
                "distinct-token ratio": round(_distinct_ratio(r.new_token_ids), 3),
                "stopped by": r.finished_by,
            }
            for v, r in res["runs"]
        ]
    )
    st.dataframe(df, width="stretch", hide_index=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(name="mean confidence", x=[str(v) for v, _ in res["runs"]],
                         y=[r.mean_confidence for _, r in res["runs"]], marker_color="#636EFA"))
    fig.add_trace(go.Bar(name="distinct-token ratio", x=[str(v) for v, _ in res["runs"]],
                         y=[_distinct_ratio(r.new_token_ids) for _, r in res["runs"]], marker_color="#EF553B"))
    fig.update_layout(barmode="group", xaxis_title=res["label"], yaxis=dict(range=[0, 1.05]),
                      height=320, margin=dict(l=10, r=10, t=30, b=30))
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### 📝 Analysis")
    st.markdown(EXPLANATIONS[res["param"]])
