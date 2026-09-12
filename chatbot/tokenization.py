"""Text  →  tokens  →  token IDs helpers and a Plotly "token chips" figure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import plotly.graph_objects as go
import torch

# Palette used to colour token chips (cycles).
TOKEN_COLORS = [
    "#FFD6A5", "#CAFFBF", "#9BF6FF", "#BDB2FF", "#FFC6FF",
    "#FDFFB6", "#A0C4FF", "#FFADAD", "#B5EAD7", "#E2F0CB",
]


@dataclass
class TokenInfo:
    index: int
    token: str        # raw sub-word as stored in the vocabulary (e.g. "Ġmachine")
    pretty: str       # raw token with visible whitespace markers (e.g. "␣machine")
    text: str         # decoded text of that single id (e.g. " machine")
    token_id: int


def prettify(raw: str) -> str:
    """Make byte-level BPE markers visible."""
    return (
        raw.replace("Ġ", "␣")   # GPT-2 / Qwen / SmolLM: leading space
           .replace("▁", "␣")   # SentencePiece: leading space
           .replace("Ċ", "↵")   # newline
    )


def tokenize_text(tokenizer: Any, text: str, add_special_tokens: bool = False) -> list[TokenInfo]:
    ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    raws = tokenizer.convert_ids_to_tokens(ids)
    out = []
    for i, (tid, raw) in enumerate(zip(ids, raws)):
        out.append(
            TokenInfo(
                index=i,
                token=raw,
                pretty=prettify(raw),
                text=tokenizer.decode([tid]),
                token_id=int(tid),
            )
        )
    return out


def tokens_figure(tokens: list[TokenInfo], units_per_row: float = 80.0) -> go.Figure:
    """Coloured chips drawn with Plotly: one box per token, its token ID underneath.

    Boxes are laid out left-to-right and wrap to a new row when the row is full.
    (Pure Python / Plotly – no HTML or CSS.)
    """
    gap, row_height, box_height = 0.7, 3.2, 1.5
    placed: list[tuple[TokenInfo, str, float, int, float]] = []   # (token, label, x0, row, width)
    x, row = 0.0, 0
    for t in tokens:
        label = t.pretty if t.pretty.strip() else "␣"
        width = max(len(label), 2) * 0.72 + 1.2
        if x + width > units_per_row and x > 0:
            row, x = row + 1, 0.0
        placed.append((t, label, x, row, width))
        x += width + gap
    n_rows = row + 1

    fig = go.Figure()
    centers_x, centers_y, hover = [], [], []
    for t, label, x0, r, width in placed:
        top = -r * row_height
        color = TOKEN_COLORS[t.index % len(TOKEN_COLORS)]
        fig.add_shape(type="rect", x0=x0, x1=x0 + width, y0=top - box_height, y1=top,
                      fillcolor=color, line=dict(color=color, width=1))
        fig.add_annotation(x=x0 + width / 2, y=top - box_height / 2, text=label, showarrow=False,
                           font=dict(family="monospace", size=14, color="#111111"))
        fig.add_annotation(x=x0 + width / 2, y=top - box_height - 0.55, text=str(t.token_id),
                           showarrow=False, font=dict(size=10, color="#666666"))
        centers_x.append(x0 + width / 2)
        centers_y.append(top - box_height / 2)
        hover.append(f"position {t.index}  |  token {label}  |  id {t.token_id}  |  text {t.text!r}")

    # invisible markers give a hover tooltip per chip
    fig.add_trace(go.Scatter(x=centers_x, y=centers_y, mode="markers",
                             marker=dict(size=18, opacity=0), hovertext=hover, hoverinfo="text",
                             showlegend=False))
    fig.update_xaxes(visible=False, range=[-0.5, units_per_row + 0.5], fixedrange=True)
    fig.update_yaxes(visible=False, range=[-(n_rows * row_height) + 0.3, 0.5], fixedrange=True)
    fig.update_layout(height=40 + int(n_rows * 64), margin=dict(l=0, r=0, t=0, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def embedding_preview(model: Any, ids: list[int], dims: int = 8) -> torch.Tensor:
    """First `dims` values of the embedding vector for each id (vocab × hidden lookup)."""
    emb = model.get_input_embeddings().weight
    idx = torch.tensor(ids, device=emb.device)
    with torch.no_grad():
        return emb[idx, :dims].float().cpu()
