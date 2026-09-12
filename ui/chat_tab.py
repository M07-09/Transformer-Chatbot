"""💬 Chat tab: the conversation itself + memory / next-token evidence."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from chatbot import (
    EXAMPLE_QUESTIONS,
    GenerationResult,
    GenerationSettings,
    LoadedModel,
    TurnRecord,
    build_prompt,
    generate_stream,
)


def trace_dataframe(result: GenerationResult, max_rows: int | None = None) -> pd.DataFrame:
    rows = []
    for s in result.trace[:max_rows] if max_rows else result.trace:
        alts = "  |  ".join(f"{repr(c.text)} {c.prob:.0%}" for c in s.candidates)
        rows.append(
            {
                "step": s.step + 1,
                "chosen token": repr(s.chosen.text),
                "id": s.chosen.token_id,
                "P(chosen) after T/top-k/top-p": round(s.chosen.prob, 3),
                "P(chosen) raw": round(s.raw_prob, 3),
                "top alternatives": alts,
            }
        )
    return pd.DataFrame(rows)


def _answer(lm: LoadedModel, prompt: str, settings: GenerationSettings, memory: dict) -> None:
    """Append the user message, build the prompt (with memory), stream the answer."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    built = build_prompt(
        lm,
        st.session_state.messages,
        use_memory=memory["use_memory"],
        max_turns=memory["max_turns"],
        reserve_tokens=settings.max_new_tokens,
    )
    result = GenerationResult()
    with st.chat_message("assistant"):
        text = st.write_stream(
            generate_stream(lm, built.input_ids, settings, result, built.stop_strings)
        )
        st.caption(
            f"⏱ {result.elapsed:.2f}s · {result.generated_tokens} tokens · "
            f"{result.tokens_per_second:.1f} tok/s · prompt {built.num_tokens} tokens · "
            f"memory: {built.turns_included} past turns · stopped by: {result.finished_by}"
        )

    final_text = (text if isinstance(text, str) else result.text).strip() or "(empty answer)"
    st.session_state.messages.append({"role": "assistant", "content": final_text})
    st.session_state.last_result = result
    st.session_state.last_prompt = built
    st.session_state.stats.add(
        TurnRecord(
            question_no=st.session_state.stats.questions_asked + 1,
            prompt_tokens=built.num_tokens,
            generated_tokens=result.generated_tokens,
            seconds=result.elapsed,
            temperature=settings.temperature,
            top_k=settings.top_k,
            memory_turns=built.turns_included,
            mean_confidence=result.mean_confidence,
        )
    )
    st.rerun()


def render(lm: LoadedModel, settings: GenerationSettings, memory: dict) -> None:
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("💬 Conversation")
        if not lm.is_chat:
            st.info(
                "This model has **no chat template** (raw language model). The app wraps the "
                "history in a `User:` / `AI:` transcript and stops when the model starts a new "
                "`User:` line. Expect less coherent answers than a chat-tuned model."
            )

        # quick demo questions – they show conversation memory in action
        cols = st.columns(len(EXAMPLE_QUESTIONS))
        for c, q in zip(cols, EXAMPLE_QUESTIONS):
            if c.button(q, width="stretch", key=f"ex_{q}"):
                st.session_state.pending_prompt = q
                st.rerun()

        # history
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        prompt = st.chat_input("Ask me anything…  (press Enter or click ➤ to send)")
        if st.session_state.pending_prompt:
            prompt = st.session_state.pending_prompt
            st.session_state.pending_prompt = None
        if prompt and prompt.strip():
            _answer(lm, prompt.strip(), settings, memory)

    with right:
        st.subheader("🔬 What the model actually saw")
        built = st.session_state.last_prompt
        result: GenerationResult | None = st.session_state.last_result
        if built is None or result is None:
            st.caption("Send a message to see the prompt (with memory) and the next-token trace here.")
            return

        c1, c2, c3 = st.columns(3)
        c1.metric("Prompt tokens", built.num_tokens)
        c2.metric("Past turns in context", built.turns_included,
                  help="(user, assistant) pairs included thanks to conversation memory")
        c3.metric("Generated tokens", result.generated_tokens)
        if built.turns_dropped:
            st.warning(f"{built.turns_dropped} old turn(s) were dropped to fit the context window.")

        with st.expander("🧠 Full prompt sent to the Transformer (conversation memory)", expanded=False):
            st.code(built.text, language="text")

        with st.expander("🎯 Next-token prediction trace of the last answer", expanded=True):
            st.caption(
                "Each row is one step of the generation loop: the model produced a probability "
                "distribution over the whole vocabulary and ONE token was chosen. "
                "`raw` = softmax(logits); the other column is after temperature / top-k / top-p."
            )
            st.dataframe(trace_dataframe(result, max_rows=40), width="stretch", height=320, hide_index=True)
            st.caption(f"Mean confidence of chosen tokens: **{result.mean_confidence:.1%}**")
