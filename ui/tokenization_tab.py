"""🔤 Tokenization tab: text → tokens → token IDs → embeddings."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from chatbot import LoadedModel, embedding_preview, tokenize_text, tokens_figure

DEFAULT_TEXT = "Machine learning is a subfield of artificial intelligence. Tokenization splits text into sub-words!"


def _last_user_message() -> str | None:
    for m in reversed(st.session_state.messages):
        if m["role"] == "user":
            return m["content"]
    return None


def render(lm: LoadedModel) -> None:
    st.subheader("🔤 Tokenization: how text becomes numbers")
    tok = lm.tokenizer

    default = _last_user_message() or DEFAULT_TEXT
    text = st.text_area("Text to tokenize", value=default, height=90, key="tok_text")
    add_special = st.checkbox("Add special tokens (BOS/EOS) if the tokenizer uses them", value=False)

    if not text.strip():
        st.info("Type some text above.")
        return

    tokens = tokenize_text(tok, text, add_special_tokens=add_special)

    # ---- headline numbers ----
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Characters", len(text))
    c2.metric("Words (whitespace)", len(text.split()))
    c3.metric("Tokens", len(tokens))
    c4.metric("Tokenizer vocabulary", f"{len(tok):,}",
              help="Distinct tokens the tokenizer can produce (base BPE vocabulary + special tokens). "
                   "The model's embedding matrix may be padded to a slightly larger size – see Model Info.")

    # ---- coloured chips ----
    st.markdown("**Tokens** (␣ = leading space, ↵ = newline) with their **Token IDs** underneath:")
    st.plotly_chart(tokens_figure(tokens), width="stretch", config={"displayModeBar": False})

    st.markdown("**Token IDs as the model receives them:**")
    st.code("[" + ", ".join(str(t.token_id) for t in tokens) + "]", language="python")

    # ---- table ----
    df = pd.DataFrame(
        {
            "position": [t.index for t in tokens],
            "token (vocab entry)": [t.pretty for t in tokens],
            "decoded text": [repr(t.text) for t in tokens],
            "token ID": [t.token_id for t in tokens],
        }
    )
    st.dataframe(df, width="stretch", hide_index=True, height=min(400, 38 * (len(df) + 1)))

    # ---- round trip ----
    ids = [t.token_id for t in tokens]
    decoded = tok.decode(ids, skip_special_tokens=True)
    st.markdown(
        f"**Round-trip check:** `decode(encode(text))` → `{decoded!r}` — "
        + ("✅ identical to the input" if decoded == text else "⚠️ differs slightly (normalisation)")
    )

    # ---- embeddings ----
    with st.expander("🔢 Peek at the embedding vectors (Token ID → vector)"):
        dims = 8
        emb = embedding_preview(lm.model, ids[:12], dims=dims)
        emb_df = pd.DataFrame(
            emb.numpy().round(4),
            index=[f"{t.pretty} (id {t.token_id})" for t in tokens[:12]],
            columns=[f"d{i}" for i in range(dims)],
        )
        hidden = lm.model.get_input_embeddings().weight.shape[1]
        n_rows = lm.model.get_input_embeddings().weight.shape[0]
        st.caption(
            f"Each ID is a row index into the embedding matrix of shape "
            f"**{n_rows:,} × {hidden}**. Only the first {dims} of {hidden} dimensions are shown."
        )
        st.dataframe(emb_df, width="stretch")

    # ---- explanation ----
    st.markdown("---")
    st.markdown(
        """
### Text → Tokens → Token IDs → Embeddings
| Stage | What it is | Example |
|---|---|---|
| **Text** | The raw string typed by the user | `"Machine learning"` |
| **Tokens** | Sub-word pieces produced by the tokenizer (Byte-Pair Encoding). Frequent words stay whole, rare words are split into pieces. A leading `␣` means *"there was a space before me"*. | `["Machine", "␣learning"]` |
| **Token IDs** | The integer index of every token in the vocabulary. This is the **only** thing the neural network receives. | `[38105, 6832]` |
| **Embeddings** | Each ID selects one row of the embedding matrix → a dense vector of size *embedding_dim*. Positional information is added so the model knows the order. | `[0.01, -0.23, …]` |

**Why not one token per word?**  A fixed vocabulary of ~50k–150k sub-words can spell *any* word
(even misspellings and new words) while keeping the embedding table small. The same tokenizer
maps the model's output IDs back to text (`decode`), which is how the answer becomes readable again.
"""
    )

    st.markdown("**Special tokens of this tokenizer:**")
    specials = {
        "BOS": (tok.bos_token, tok.bos_token_id),
        "EOS": (tok.eos_token, tok.eos_token_id),
        "PAD": (tok.pad_token, tok.pad_token_id),
    }
    st.table(pd.DataFrame(
        [{"role": k, "token": str(v[0]), "id": str(v[1])} for k, v in specials.items()]
    ))
