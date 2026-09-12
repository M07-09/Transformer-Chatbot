"""Self-attention extraction and visualisation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import torch

from .model_manager import LoadedModel
from .tokenization import prettify

MAX_ATTENTION_TOKENS = 48


@dataclass
class AttentionData:
    tokens: list[str]                 # pretty token labels
    weights: np.ndarray               # [layers, heads, seq, seq]

    @property
    def num_layers(self) -> int:
        return self.weights.shape[0]

    @property
    def num_heads(self) -> int:
        return self.weights.shape[1]

    def matrix(self, layer: int, head: int | None) -> np.ndarray:
        """One head, or the mean over all heads when head is None."""
        if head is None:
            return self.weights[layer].mean(axis=0)
        return self.weights[layer, head]


@torch.inference_mode()
def compute_attention(lm: LoadedModel, text: str) -> AttentionData:
    tok = lm.tokenizer
    ids = tok.encode(text, add_special_tokens=False)[:MAX_ATTENTION_TOKENS]
    input_ids = torch.tensor([ids], device=lm.device)
    out = lm.model(input_ids=input_ids, output_attentions=True, use_cache=False)
    # out.attentions: tuple(len = layers) of [batch, heads, seq, seq]
    att = torch.stack(out.attentions)[:, 0].float().cpu().numpy()
    # the model runs in bfloat16, whose rounding makes rows sum to 0.99–1.01;
    # renormalise so that every row of the heat-map sums to exactly 1
    att = att / att.sum(axis=-1, keepdims=True)
    labels = [prettify(t) for t in tok.convert_ids_to_tokens(ids)]
    # make duplicate labels unique so plotly axes do not merge them
    seen: dict[str, int] = {}
    unique = []
    for t in labels:
        seen[t] = seen.get(t, 0) + 1
        unique.append(t if seen[t] == 1 else f"{t}({seen[t]})")
    return AttentionData(tokens=unique, weights=att)


def attention_heatmap(data: AttentionData, layer: int, head: int | None) -> go.Figure:
    m = data.matrix(layer, head)
    title = f"Layer {layer} · " + ("mean of all heads" if head is None else f"Head {head}")
    fig = go.Figure(
        data=go.Heatmap(
            z=m,
            x=data.tokens,
            y=data.tokens,
            colorscale="Viridis",
            zmin=0.0,
            zmax=float(m.max()) if m.size else 1.0,
            colorbar=dict(title="attention"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Key tokens (attended TO)",
        yaxis_title="Query tokens (attending FROM)",
        yaxis=dict(autorange="reversed"),
        height=max(420, 22 * len(data.tokens) + 160),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def token_focus_bar(data: AttentionData, layer: int, head: int | None, query_index: int) -> go.Figure:
    """Bar chart: how much token `query_index` attends to every other token."""
    m = data.matrix(layer, head)
    row = m[query_index]
    colors = ["#EF553B" if i == query_index else "#636EFA" for i in range(len(row))]
    fig = go.Figure(go.Bar(x=data.tokens, y=row, marker_color=colors, name="attention weight"))
    fig.update_layout(
        title=f"Where does token “{data.tokens[query_index]}” look?  (row {query_index} of the attention matrix)",
        yaxis_title="attention weight (sums to 1)",
        height=340,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def head_summary(data: AttentionData, layer: int) -> np.ndarray:
    """Per-head 'entropy' — how spread out each head's attention is (for a small table)."""
    w = data.weights[layer]                        # [heads, seq, seq]
    ent = -(w * np.log(w + 1e-9)).sum(axis=-1)     # [heads, seq]
    return ent.mean(axis=-1)
