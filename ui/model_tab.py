"""🧠 Model info tab: architecture facts read from the loaded model."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from chatbot import LoadedModel, get_model_info, gpu_info, human_count, layer_structure


def render(lm: LoadedModel) -> None:
    st.subheader("🧠 Pre-trained Transformer model")
    info = get_model_info(lm)

    st.markdown(f"**`{info['model_name']}`** — `{info['architecture']}` (type: `{info['model_type']}`), "
                f"loaded on **{info['device'].upper()}** in **{info['dtype']}** "
                f"({info['load_seconds']:.1f} s)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transformer layers", info["num_layers"], help="Number of stacked decoder blocks")
    c2.metric("Attention heads / layer", info["num_attention_heads"],
              help="Each head learns a different way of relating tokens")
    c3.metric("Embedding dimension", info["embedding_dim"], help="Size of every token vector (d_model)")
    c4.metric("Vocabulary size", f"{info['vocab_size']:,}",
              help="Rows of the embedding matrix / outputs of the LM head (config.vocab_size). "
                   f"The tokenizer itself defines {len(lm.tokenizer):,} tokens; the rest is padding for GPU efficiency.")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Total parameters", human_count(info["total_params"]))
    c6.metric("Context length", f"{info['context_length']:,} tokens")
    c7.metric("Feed-forward dim", info["ffn_dim"])
    c8.metric("Head dimension", info["head_dim"], help="embedding_dim / heads")

    st.markdown("#### Details")
    rows = {
        "Key/Value heads (GQA)": info["num_kv_heads"],
        "Activation function": info["activation"],
        "Embedding matrix": f"{info['vocab_size']:,} × {info['embedding_dim']}  = {human_count(info['embedding_params'])} params",
        "Tied input/output embeddings": info["tie_word_embeddings"],
        "Chat template available": "yes" if info["chat_template"] else "no (plain LM)",
        "BOS / EOS / PAD tokens": f"{info['bos_token']} / {info['eos_token']} / {info['pad_token']}",
        "EOS token id(s)": ", ".join(map(str, info["eos_ids"])),
    }
    st.table(pd.DataFrame({"property": list(rows.keys()), "value": [str(v) for v in rows.values()]}))

    g = gpu_info()
    if g:
        st.markdown("#### GPU")
        st.table(pd.DataFrame({
            "property": ["Device", "CUDA", "Compute capability", "VRAM total", "VRAM reserved by PyTorch"],
            "value": [g["name"], g["cuda_version"], g["compute_capability"],
                      f"{g['total_gb']:.1f} GB", f"{g['reserved_gb']:.2f} GB"],
        }))

    with st.expander("🧱 Modules inside one Transformer block (from the live model)"):
        st.code("\n".join(layer_structure(lm)), language="text")

    with st.expander("📐 How the numbers fit together"):
        L, H, D, V = info["num_layers"], info["num_attention_heads"], info["embedding_dim"], info["vocab_size"]
        st.markdown(
            f"""
* A token ID is looked up in the **embedding matrix** (`{V:,} × {D}`) → a vector of **{D}** numbers.
* That vector goes through **{L} identical decoder blocks**. In each block:
  * **Multi-head self-attention** with **{H} heads**; every head works in a sub-space of
    `{D} / {H} = {info['head_dim']}` dimensions and produces its own attention pattern.
  * A **feed-forward network** that expands `{D} → {info['ffn_dim']} → {D}`.
  * Residual connections + normalisation keep training stable.
* The final vector is projected by the **LM head** back to the vocabulary size (**{V:,}** logits).
  `softmax` turns the logits into a probability for *every* possible next token.
"""
        )
