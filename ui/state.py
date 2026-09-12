"""Session state initialisation and cached model loading."""

from __future__ import annotations

import streamlit as st

from chatbot import DEFAULT_MODEL, GPUNotAvailableError, LoadedModel, SessionStats, load_model


def init_state() -> None:
    defaults = {
        "messages": [],            # [{"role": "user"|"assistant", "content": str}]
        "stats": SessionStats(),
        "last_result": None,       # GenerationResult of the last answer
        "last_prompt": None,       # BuiltPrompt of the last answer
        "pending_prompt": None,    # set by the example-question buttons
        "model_name": DEFAULT_MODEL,
        "lab_results": None,
        "attention_cache": {},
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


@st.cache_resource(show_spinner=False)
def _cached_load(model_name: str) -> LoadedModel:
    return load_model(model_name)


def get_model(model_name: str) -> LoadedModel:
    """Load (once) the selected model onto the GPU, or stop the app with a clear error."""
    try:
        with st.spinner(f"Loading **{model_name}** onto the GPU … (the first time downloads the weights)"):
            return _cached_load(model_name)
    except GPUNotAvailableError as e:
        st.error("🚫 **GPU required** — this application is designed to run exclusively on a CUDA GPU.")
        st.code(str(e))
        st.stop()
    except Exception as e:  # download / network / OOM problems
        st.error(f"❌ Could not load `{model_name}`: {e}")
        st.stop()


def clear_conversation() -> None:
    st.session_state.messages = []
    st.session_state.stats = SessionStats()
    st.session_state.last_result = None
    st.session_state.last_prompt = None
    st.session_state.pending_prompt = None
