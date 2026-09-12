"""Intelligent Transformer-Based Chatbot — Streamlit entry point.

Run with EITHER:
    python app.py           (recommended -- this file launches Streamlit itself)
    streamlit run app.py    (the classic way, still works)
"""

import subprocess
import sys

import streamlit.runtime as _st_runtime

if not _st_runtime.exists():
    # Started as a plain script (`python app.py`), not via `streamlit run`.
    # Relaunch this same file under Streamlit, then exit with its exit code.
    raise SystemExit(subprocess.call(
        [sys.executable, "-m", "streamlit", "run", __file__, "--", *sys.argv[1:]]
    ))

import streamlit as st

st.set_page_config(
    page_title="Transformer Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui import (  # noqa: E402  (must come after set_page_config)
    architecture_tab,
    attention_tab,
    chat_tab,
    dashboard_tab,
    lab_tab,
    model_tab,
    next_token_tab,
    sidebar,
    tokenization_tab,
)
from ui.state import get_model, init_state  # noqa: E402

init_state()
settings, memory = sidebar.render()
lm = get_model(st.session_state.model_name)

st.title("🤖 Intelligent Transformer-Based Chatbot")
st.caption(
    f"Model: **{lm.name}** · device: **{lm.device.upper()}** ({lm.dtype}) · "
    "every answer is produced token-by-token by next-token prediction."
)

tabs = st.tabs([
    "💬 Chat",
    "🔤 Tokenization",
    "🧠 Model Info",
    "🎯 Next-Token Prediction",
    "🔍 Self-Attention",
    "🧪 Parameter Lab",
    "📊 Dashboard",
    "🏗️ Architecture",
])

with tabs[0]:
    chat_tab.render(lm, settings, memory)
with tabs[1]:
    tokenization_tab.render(lm)
with tabs[2]:
    model_tab.render(lm)
with tabs[3]:
    next_token_tab.render(lm, settings)
with tabs[4]:
    attention_tab.render(lm)
with tabs[5]:
    lab_tab.render(lm, settings)
with tabs[6]:
    dashboard_tab.render(lm, settings, memory)
with tabs[7]:
    architecture_tab.render(lm)
