"""Sidebar: model selection, generation settings, memory settings, GPU status."""

from __future__ import annotations

import streamlit as st

from chatbot import DEFAULT_SETTINGS, MODEL_REGISTRY, GenerationSettings, gpu_info

from .state import clear_conversation


def render() -> tuple[GenerationSettings, dict]:
    with st.sidebar:
        st.title("🤖 Transformer Chatbot")
        st.caption("Tokenization → Transformer → Next-token prediction")

        # ---------------- model ----------------
        st.subheader("🧠 Model")
        names = list(MODEL_REGISTRY.keys())
        current = st.session_state.model_name
        choice = st.selectbox(
            "Pre-trained model",
            names,
            index=names.index(current) if current in names else 0,
            format_func=lambda n: MODEL_REGISTRY[n]["label"],
            help="Chat-tuned models follow instructions; GPT-2 models are raw language models.",
        )
        if choice != current:
            st.session_state.model_name = choice
            st.session_state.attention_cache = {}
            st.rerun()

        # ---------------- generation settings ----------------
        st.subheader("🎛️ Generation settings")
        temperature = st.slider(
            "Temperature", 0.0, 2.0, DEFAULT_SETTINGS.temperature, 0.05,
            help="0 = greedy (always the most likely token). Higher = flatter distribution = more random.",
        )
        top_k = st.slider(
            "Top-K", 0, 200, DEFAULT_SETTINGS.top_k, 1,
            help="Keep only the K most likely tokens before sampling. 0 = disabled, 1 = greedy.",
        )
        top_p = st.slider(
            "Top-P (nucleus)", 0.1, 1.0, DEFAULT_SETTINGS.top_p, 0.05,
            help="Keep the smallest set of tokens whose probabilities add up to P. 1.0 = disabled.",
        )
        max_new_tokens = st.slider(
            "Max new tokens", 10, 400, DEFAULT_SETTINGS.max_new_tokens, 10,
            help="Upper bound on the length of the generated answer.",
        )
        repetition_penalty = st.slider(
            "Repetition penalty", 1.0, 2.0, DEFAULT_SETTINGS.repetition_penalty, 0.05,
            help="> 1 discourages repeating tokens that already appeared.",
        )
        settings = GenerationSettings(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )

        # ---------------- memory ----------------
        st.subheader("🧠 Conversation memory")
        use_memory = st.toggle(
            "Use conversation history as context", value=True,
            help="OFF = the model only sees the current question (it will forget everything).",
        )
        max_turns = st.slider(
            "Past turns to remember", 1, 20, 6, 1, disabled=not use_memory,
            help="How many previous (question, answer) pairs are included in the prompt.",
        )
        memory = {"use_memory": use_memory, "max_turns": max_turns}

        if st.button("🗑️ Clear conversation", width="stretch"):
            clear_conversation()
            st.rerun()

        # ---------------- GPU status ----------------
        st.subheader("⚡ GPU")
        g = gpu_info()
        if g:
            st.success(f"**{g['name']}**  \nCUDA {g['cuda_version']} · compute {g['compute_capability']}")
            st.progress(
                min(1.0, g["reserved_gb"] / g["total_gb"]),
                text=f"VRAM {g['reserved_gb']:.2f} / {g['total_gb']:.1f} GB",
            )
        else:
            st.error("No CUDA GPU detected")

    return settings, memory
