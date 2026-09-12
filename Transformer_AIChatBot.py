"""Intelligent Transformer-Based Chatbot — SINGLE-FILE version.

    streamlit run transformer_chatbot_app.py

This file is generated from the modular project (chatbot/ + ui/ + app.py) by
tools/build_single_file.py and contains exactly the same code in one module:

    1. configuration & model registry          (chatbot/config.py)
    2. GPU-only model loading + model facts     (chatbot/model_manager.py)
    3. tokenization helpers                     (chatbot/tokenization.py)
    4. next-token prediction & sampling loop    (chatbot/generation.py)
    5. self-attention extraction & plots        (chatbot/attention.py)
    6. conversation memory → prompt             (chatbot/memory.py)
    7. dashboard statistics                     (chatbot/stats.py)
    8. Streamlit state, sidebar and the 8 tabs  (ui/*.py)
    9. main entry point                         (app.py)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import time
from typing import Any, Iterator
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


st.set_page_config(
    page_title="Transformer Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================================================================================================
# chatbot/config.py
# ================================================================================================
# Central configuration: model registry, default generation settings, prompts.
# ---------------------------------------------------------------------------
# Models the user can pick from the sidebar.
# "chat" = the tokenizer ships a chat template (instruction-tuned model).
# Non-chat models (GPT-2 family) use a manual "User:/AI:" prompt format.
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "Qwen/Qwen2.5-0.5B-Instruct": {
        "label": "Qwen2.5-0.5B-Instruct  (recommended, chat-tuned, ~1 GB)",
        "chat": True,
    },
    "HuggingFaceTB/SmolLM2-360M-Instruct": {
        "label": "SmolLM2-360M-Instruct  (small, chat-tuned, ~0.7 GB)",
        "chat": True,
    },
    "HuggingFaceTB/SmolLM2-135M-Instruct": {
        "label": "SmolLM2-135M-Instruct  (tiny, chat-tuned, ~0.3 GB)",
        "chat": True,
    },
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "label": "Qwen2.5-1.5B-Instruct  (better answers, slower, ~3 GB)",
        "chat": True,
    },
    "distilgpt2": {
        "label": "DistilGPT-2  (classic, NOT chat-tuned, ~0.3 GB)",
        "chat": False,
    },
    "gpt2": {
        "label": "GPT-2 small  (classic, NOT chat-tuned, ~0.5 GB)",
        "chat": False,
    },
}

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = (
    "You are a helpful and knowledgeable AI assistant. Answer clearly and concisely "
    "(a few sentences). Use the previous conversation to understand follow-up "
    "questions and references such as 'it', 'its', 'they' or 'that'."
)

# Prompt format for models without a chat template (GPT-2 family).
PLAIN_PREAMBLE = (
    "The following is a conversation between a curious User and a helpful, "
    "knowledgeable AI assistant. The AI answers accurately and concisely.\n\n"
)
PLAIN_USER_PREFIX = "User: "
PLAIN_AI_PREFIX = "AI: "
PLAIN_STOP_STRINGS = ["\nUser:", "User:", "\nAI:", "\n\n\n"]

# Upper bound on prompt tokens we send to the model (keeps CPU inference fast).
MAX_CONTEXT_BUDGET = 2048


@dataclass
class GenerationSettings:
    """All user-adjustable generation parameters."""

    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    max_new_tokens: int = 120
    repetition_penalty: float = 1.1

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULT_SETTINGS = GenerationSettings()

EXAMPLE_QUESTIONS = [
    "What is Machine Learning?",
    "What are its main types?",
    "Give me one real-world example of each type.",
]


# ================================================================================================
# chatbot/model_manager.py
# ================================================================================================
# Loading the pre-trained Transformer and reading its architecture facts.
@dataclass
class LoadedModel:
    name: str
    tokenizer: Any
    model: Any
    device: str
    dtype: str
    load_seconds: float
    is_chat: bool  # tokenizer ships a chat template

    @property
    def eos_ids(self) -> set[int]:
        """Every token id that should terminate generation."""
        ids: set[int] = set()
        if self.tokenizer.eos_token_id is not None:
            ids.add(int(self.tokenizer.eos_token_id))
        gen_eos = getattr(self.model.generation_config, "eos_token_id", None)
        if isinstance(gen_eos, int):
            ids.add(gen_eos)
        elif isinstance(gen_eos, (list, tuple)):
            ids.update(int(i) for i in gen_eos)
        return ids


class GPUNotAvailableError(RuntimeError):
    """Raised when no CUDA GPU can be used. This project is GPU-only by design."""


def pick_device() -> str:
    """GPU-only policy: refuse to run on CPU."""
    if not torch.cuda.is_available():
        raise GPUNotAvailableError(
            "No CUDA GPU detected. This chatbot runs exclusively on the GPU.\n"
            f"Installed torch: {torch.__version__} (CUDA build: {torch.version.cuda}).\n"
            "Install a CUDA build of PyTorch, e.g.:\n"
            "    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130\n"
            "and make sure the NVIDIA driver is up to date."
        )
    return "cuda"


def gpu_info() -> dict:
    """Name / memory of the active GPU (for the sidebar and model-info tab)."""
    if not torch.cuda.is_available():
        return {}
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    return {
        "name": props.name,
        "total_gb": props.total_memory / 1024**3,
        "allocated_gb": torch.cuda.memory_allocated(idx) / 1024**3,
        "reserved_gb": torch.cuda.memory_reserved(idx) / 1024**3,
        "cuda_version": torch.version.cuda,
        "compute_capability": f"{props.major}.{props.minor}",
    }


def load_model(model_name: str) -> LoadedModel:
    """Download (first time) and load tokenizer + causal LM onto the GPU.

    attn_implementation="eager" is required so that the model can return the
    attention weights (the fused SDPA kernels do not expose them).
    """
    t0 = time.perf_counter()
    device = pick_device()                      # raises if no GPU
    # bfloat16: half the memory of fp32, and numerically safer than fp16 for Qwen.
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()

    is_chat = bool(getattr(tokenizer, "chat_template", None))
    if model_name in MODEL_REGISTRY:
        is_chat = MODEL_REGISTRY[model_name]["chat"] and is_chat

    return LoadedModel(
        name=model_name,
        tokenizer=tokenizer,
        model=model,
        device=device,
        dtype=str(dtype).replace("torch.", ""),
        load_seconds=time.perf_counter() - t0,
        is_chat=is_chat,
    )


def _first_attr(obj: Any, names: list[str], default=None):
    for n in names:
        if hasattr(obj, n) and getattr(obj, n) is not None:
            return getattr(obj, n)
    return default


def get_model_info(lm: LoadedModel) -> dict:
    """Architecture facts read straight from the model config."""
    cfg = lm.model.config
    layers = _first_attr(cfg, ["num_hidden_layers", "n_layer"])
    heads = _first_attr(cfg, ["num_attention_heads", "n_head"])
    kv_heads = _first_attr(cfg, ["num_key_value_heads"], heads)
    hidden = _first_attr(cfg, ["hidden_size", "n_embd"])
    ffn = _first_attr(cfg, ["intermediate_size", "n_inner"])
    if ffn is None and hidden is not None:
        ffn = 4 * hidden  # GPT-2 default
    ctx = _first_attr(cfg, ["max_position_embeddings", "n_positions"])
    vocab = cfg.vocab_size
    total_params = sum(p.numel() for p in lm.model.parameters())
    emb_params = lm.model.get_input_embeddings().weight.numel()
    head_dim = hidden // heads if (hidden and heads) else None

    return {
        "model_name": lm.name,
        "model_type": cfg.model_type,
        "architecture": lm.model.__class__.__name__,
        "num_layers": layers,
        "num_attention_heads": heads,
        "num_kv_heads": kv_heads,
        "embedding_dim": hidden,
        "head_dim": head_dim,
        "ffn_dim": ffn,
        "vocab_size": vocab,
        "context_length": ctx,
        "total_params": total_params,
        "embedding_params": emb_params,
        "activation": _first_attr(cfg, ["hidden_act", "activation_function"], "?"),
        "tie_word_embeddings": getattr(cfg, "tie_word_embeddings", None),
        "device": lm.device,
        "dtype": lm.dtype,
        "chat_template": lm.is_chat,
        "load_seconds": lm.load_seconds,
        "bos_token": lm.tokenizer.bos_token,
        "eos_token": lm.tokenizer.eos_token,
        "pad_token": lm.tokenizer.pad_token,
        "eos_ids": sorted(lm.eos_ids),
    }


def layer_structure(lm: LoadedModel, max_lines: int = 40) -> list[str]:
    """Human-readable listing of the first Transformer block's sub-modules."""
    lines: list[str] = []
    for name, module in lm.model.named_modules():
        # keep only the first block (…layers.0… or …h.0…) plus embeddings / head
        if (".layers.0." in name or ".h.0." in name) and name.count(".") <= 4:
            lines.append(f"{name}  →  {module.__class__.__name__}")
        if len(lines) >= max_lines:
            break
    return lines


def human_count(n: int | None) -> str:
    if n is None:
        return "?"
    if n >= 1e9:
        return f"{n/1e9:.2f} B"
    if n >= 1e6:
        return f"{n/1e6:.1f} M"
    if n >= 1e3:
        return f"{n/1e3:.1f} K"
    return str(n)


# ================================================================================================
# chatbot/tokenization.py
# ================================================================================================
# Text  →  tokens  →  token IDs helpers and a Plotly "token chips" figure.
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


# ================================================================================================
# chatbot/generation.py
# ================================================================================================
# Next-token prediction and sampling.
#
# The generation loop is written by hand (instead of model.generate) so that the
# application can *show* what happens at every step:
#
#     logits  →  repetition penalty  →  temperature  →  top-k  →  top-p
#             →  softmax  →  sample / argmax  →  append token  →  repeat
# ---------------------------------------------------------------------------
# Data structures recorded during generation
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    token_id: int
    text: str
    prob: float


@dataclass
class StepTrace:
    step: int
    chosen: Candidate
    candidates: list[Candidate]          # top alternatives *after* temperature/top-k/top-p
    raw_prob: float                      # probability of chosen token before any processing


@dataclass
class GenerationResult:
    text: str = ""
    new_token_ids: list[int] = field(default_factory=list)
    trace: list[StepTrace] = field(default_factory=list)
    prompt_tokens: int = 0
    elapsed: float = 0.0
    finished_by: str = ""                # "eos" | "stop_string" | "max_tokens"
    settings: GenerationSettings | None = None

    @property
    def generated_tokens(self) -> int:
        return len(self.new_token_ids)

    @property
    def tokens_per_second(self) -> float:
        return self.generated_tokens / self.elapsed if self.elapsed > 0 else 0.0

    @property
    def mean_confidence(self) -> float:
        """Average probability of the chosen tokens (after processing)."""
        if not self.trace:
            return 0.0
        return sum(s.chosen.prob for s in self.trace) / len(self.trace)


# ---------------------------------------------------------------------------
# Logit processing (the heart of the "generation parameters" concept)
# ---------------------------------------------------------------------------
def apply_repetition_penalty(logits: torch.Tensor, generated_ids: list[int], penalty: float) -> torch.Tensor:
    if penalty == 1.0 or not generated_ids:
        return logits
    ids = torch.tensor(sorted(set(generated_ids)), device=logits.device)
    scores = logits[ids]
    scores = torch.where(scores < 0, scores * penalty, scores / penalty)
    logits = logits.clone()
    logits[ids] = scores
    return logits


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.numel():
        return logits
    threshold = torch.topk(logits, k).values[-1]
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    remove = cum - F.softmax(sorted_logits, dim=-1) > p   # keep tokens until mass ≥ p
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    out = torch.full_like(logits, float("-inf"))
    out[sorted_idx] = sorted_logits
    return out


def process_logits(logits: torch.Tensor, settings: GenerationSettings, generated_ids: list[int]) -> torch.Tensor:
    """Return the final probability distribution used for choosing the next token."""
    logits = logits.float()
    logits = apply_repetition_penalty(logits, generated_ids, settings.repetition_penalty)
    if settings.temperature <= 0:                 # greedy: one-hot on the arg-max
        probs = torch.zeros_like(logits)
        probs[torch.argmax(logits)] = 1.0
        return probs
    logits = logits / settings.temperature
    logits = top_k_filter(logits, settings.top_k)
    logits = top_p_filter(logits, settings.top_p)
    return F.softmax(logits, dim=-1)


def choose_token(probs: torch.Tensor, settings: GenerationSettings) -> int:
    if settings.temperature <= 0:
        return int(torch.argmax(probs))
    return int(torch.multinomial(probs, num_samples=1))


def top_candidates(tokenizer: Any, probs: torch.Tensor, n: int) -> list[Candidate]:
    n = min(n, int((probs > 0).sum()))
    if n == 0:
        return []
    vals, idx = torch.topk(probs, n)
    return [
        Candidate(int(i), tokenizer.decode([int(i)]), float(v))
        for v, i in zip(vals, idx)
    ]


# ---------------------------------------------------------------------------
# Single forward pass helpers
# ---------------------------------------------------------------------------
@torch.inference_mode()
def next_token_logits(lm: LoadedModel, input_ids: torch.Tensor) -> torch.Tensor:
    """Logits for the position after the last token  (shape: [vocab])."""
    out = lm.model(input_ids=input_ids.to(lm.device), use_cache=False)
    return out.logits[0, -1].float().cpu()


def distribution_from_logits(tokenizer: Any, logits: torch.Tensor, temperature: float, top_n: int = 10) -> list[Candidate]:
    """Top-n tokens of softmax(logits / T)  – used by the Next-Token tab."""
    if temperature <= 0:
        temperature = 1e-4
    probs = F.softmax(logits / temperature, dim=-1)
    return top_candidates(tokenizer, probs, top_n)


# ---------------------------------------------------------------------------
# Streaming generation loop
# ---------------------------------------------------------------------------
def _safe_emit_len(text: str, stop_strings: list[str]) -> int:
    """Length of text that can be emitted without leaking a partial stop string."""
    n = len(text)
    for s in stop_strings:
        for k in range(min(len(s), len(text)), 0, -1):
            if text.endswith(s[:k]):
                n = min(n, len(text) - k)
                break
    return n


@torch.inference_mode()
def generate_stream(
    lm: LoadedModel,
    input_ids: torch.Tensor,
    settings: GenerationSettings,
    result: GenerationResult,
    stop_strings: list[str] | None = None,
    n_alternatives: int = 5,
) -> Iterator[str]:
    """Yield text chunks while filling `result` with the next-token trace."""
    tokenizer, model = lm.tokenizer, lm.model
    stop_strings = stop_strings or []
    eos_ids = lm.eos_ids

    result.settings = settings
    result.prompt_tokens = int(input_ids.shape[1])
    result.new_token_ids = []
    result.trace = []
    t0 = time.perf_counter()

    cur = input_ids.to(lm.device)
    past = None
    emitted = ""
    full = ""
    finished_by = "max_tokens"

    for step in range(settings.max_new_tokens):
        out = model(input_ids=cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1]

        raw_probs = F.softmax(logits.float(), dim=-1)
        probs = process_logits(logits, settings, result.new_token_ids)
        next_id = choose_token(probs, settings)

        result.trace.append(
            StepTrace(
                step=step,
                chosen=Candidate(next_id, tokenizer.decode([next_id]), float(probs[next_id])),
                candidates=top_candidates(tokenizer, probs, n_alternatives),
                raw_prob=float(raw_probs[next_id]),
            )
        )

        if next_id in eos_ids:
            finished_by = "eos"
            break

        result.new_token_ids.append(next_id)
        cur = torch.tensor([[next_id]], device=lm.device)

        full = tokenizer.decode(result.new_token_ids, skip_special_tokens=True)
        if full.endswith("�"):          # incomplete multi-byte character – wait
            continue

        # stop strings (for models without a chat template)
        cut = -1
        for s in stop_strings:
            pos = full.find(s)
            if pos != -1 and (cut == -1 or pos < cut):
                cut = pos
        if cut != -1:
            full = full[:cut]
            finished_by = "stop_string"
            if len(full) > len(emitted):
                yield full[len(emitted):]
            emitted = full
            break

        safe = _safe_emit_len(full, stop_strings)
        if safe > len(emitted):
            yield full[len(emitted):safe]
            emitted = full[:safe]

    if finished_by != "stop_string":
        full = tokenizer.decode(result.new_token_ids, skip_special_tokens=True)
        if len(full) > len(emitted):
            yield full[len(emitted):]

    result.text = full.strip()
    result.elapsed = time.perf_counter() - t0
    result.finished_by = finished_by


def generate(lm: LoadedModel, input_ids: torch.Tensor, settings: GenerationSettings,
             stop_strings: list[str] | None = None) -> GenerationResult:
    """Non-streaming convenience wrapper (used by the Parameter Lab)."""
    result = GenerationResult()
    for _ in generate_stream(lm, input_ids, settings, result, stop_strings):
        pass
    return result


# ================================================================================================
# chatbot/attention.py
# ================================================================================================
# Self-attention extraction and visualisation.
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
    # the model runs in bfloat16, whose rounding makes rows sum to 0.99-1.01;
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


# ================================================================================================
# chatbot/memory.py
# ================================================================================================
# Conversation memory: turn the chat history into the prompt the model sees.
@dataclass
class BuiltPrompt:
    input_ids: torch.Tensor
    text: str
    num_tokens: int
    turns_included: int          # number of past (user, assistant) pairs kept
    turns_dropped: int           # pairs dropped to fit the context budget
    stop_strings: list[str]


def _render(lm: LoadedModel, messages: list[dict], system_prompt: str) -> str:
    """Render messages to a single prompt string."""
    if lm.is_chat:
        msgs = [{"role": "system", "content": system_prompt}] + messages
        return lm.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    # Plain-text format for models without a chat template (GPT-2 family)
    parts = [PLAIN_PREAMBLE]
    for m in messages:
        prefix = PLAIN_USER_PREFIX if m["role"] == "user" else PLAIN_AI_PREFIX
        parts.append(f"{prefix}{m['content'].strip()}\n")
    parts.append(PLAIN_AI_PREFIX.rstrip())
    return "".join(parts)


def build_prompt(
    lm: LoadedModel,
    history: list[dict],
    use_memory: bool = True,
    max_turns: int = 6,
    reserve_tokens: int = 150,
    system_prompt: str = SYSTEM_PROMPT,
) -> BuiltPrompt:
    """history = [{"role": "user"|"assistant", "content": str}, ...] ending with a user msg.

    Memory ON  → the last `max_turns` (user, assistant) pairs are prepended to the
                 new question so the model can resolve references ("its types").
    Memory OFF → only the new question is sent.
    """
    assert history and history[-1]["role"] == "user", "history must end with a user message"
    current = history[-1]
    past = history[:-1]

    # group the past into (user, assistant) pairs
    pairs: list[list[dict]] = []
    for m in past:
        if m["role"] == "user":
            pairs.append([m])
        elif pairs and len(pairs[-1]) == 1:
            pairs[-1].append(m)
    pairs = [p for p in pairs if len(p) == 2]

    if not use_memory:
        pairs = []
    else:
        pairs = pairs[-max_turns:] if max_turns > 0 else []
    total_pairs = len(pairs)

    ctx_limit = getattr(lm.model.config, "max_position_embeddings", None) or \
        getattr(lm.model.config, "n_positions", MAX_CONTEXT_BUDGET)
    budget = min(int(ctx_limit), MAX_CONTEXT_BUDGET) - reserve_tokens

    while True:
        messages = [m for p in pairs for m in p] + [current]
        text = _render(lm, messages, system_prompt)
        ids = lm.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
        if ids.shape[1] <= budget or not pairs:
            break
        pairs = pairs[1:]                     # drop the oldest pair and retry

    return BuiltPrompt(
        input_ids=ids,
        text=text,
        num_tokens=int(ids.shape[1]),
        turns_included=len(pairs),
        turns_dropped=total_pairs - len(pairs),
        stop_strings=[] if lm.is_chat else list(PLAIN_STOP_STRINGS),
    )


# ================================================================================================
# chatbot/stats.py
# ================================================================================================
# Session statistics shown on the dashboard.
@dataclass
class TurnRecord:
    question_no: int
    prompt_tokens: int
    generated_tokens: int
    seconds: float
    temperature: float
    top_k: int
    memory_turns: int
    mean_confidence: float


@dataclass
class SessionStats:
    turns: list[TurnRecord] = field(default_factory=list)

    @property
    def questions_asked(self) -> int:
        return len(self.turns)

    @property
    def generated_tokens(self) -> int:
        return sum(t.generated_tokens for t in self.turns)

    @property
    def prompt_tokens(self) -> int:
        return sum(t.prompt_tokens for t in self.turns)

    @property
    def avg_response_time(self) -> float:
        return sum(t.seconds for t in self.turns) / len(self.turns) if self.turns else 0.0

    @property
    def avg_tokens_per_second(self) -> float:
        secs = sum(t.seconds for t in self.turns)
        return self.generated_tokens / secs if secs > 0 else 0.0

    def add(self, record: TurnRecord) -> None:
        self.turns.append(record)

    def reset(self) -> None:
        self.turns.clear()


# ================================================================================================
# ui/state.py
# ================================================================================================
# Session state initialisation and cached model loading.
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


# ================================================================================================
# ui/sidebar.py
# ================================================================================================
# Sidebar: model selection, generation settings, memory settings, GPU status.
def render_sidebar() -> tuple[GenerationSettings, dict]:
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


# ================================================================================================
# ui/chat_tab.py
# ================================================================================================
# 💬 Chat tab: the conversation itself + memory / next-token evidence.
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


def render_chat(lm: LoadedModel, settings: GenerationSettings, memory: dict) -> None:
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


# ================================================================================================
# ui/tokenization_tab.py
# ================================================================================================
# 🔤 Tokenization tab: text → tokens → token IDs → embeddings.
TOK_DEFAULT_TEXT = "Machine learning is a subfield of artificial intelligence. Tokenization splits text into sub-words!"


def _last_user_message() -> str | None:
    for m in reversed(st.session_state.messages):
        if m["role"] == "user":
            return m["content"]
    return None


def render_tokenization(lm: LoadedModel) -> None:
    st.subheader("🔤 Tokenization: how text becomes numbers")
    tok = lm.tokenizer

    default = _last_user_message() or TOK_DEFAULT_TEXT
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


# ================================================================================================
# ui/model_tab.py
# ================================================================================================
# 🧠 Model info tab: architecture facts read from the loaded model.
def render_model(lm: LoadedModel) -> None:
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


# ================================================================================================
# ui/next_token_tab.py
# ================================================================================================
# 🎯 Next-token prediction tab.
NT_DEFAULT_PROMPT = "The capital of France is"


def _bar(cands, title: str, color: str = "#636EFA") -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=[c.prob for c in cands][::-1],
            y=[repr(c.text) for c in cands][::-1],
            orientation="h",
            marker_color=color,
            text=[f"{c.prob:.1%}" for c in cands][::-1],
            textposition="outside",
            hovertext=[f"id {c.token_id}  p = {c.prob:.3%}" for c in cands][::-1],
            hoverinfo="y+text",
            name="probability",
        )
    )
    fig.update_layout(title=title, xaxis_title="probability", height=360,
                      margin=dict(l=10, r=40, t=50, b=30), xaxis=dict(range=[0, 1.05]))
    return fig


def render_next_token(lm: LoadedModel, settings: GenerationSettings) -> None:
    st.subheader("🎯 Next-token prediction: the only thing a language model does")
    st.markdown(
        "A decoder-only Transformer never *writes a sentence*. It computes, for the current "
        "context, a probability for **every token in the vocabulary**, one token is chosen, it is "
        "appended to the context, and the process repeats. Try it:"
    )

    text = st.text_input("Context (the model predicts what comes next)", value=NT_DEFAULT_PROMPT, key="nt_text")
    if not text.strip():
        return

    ids = lm.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    logits = next_token_logits(lm, ids)                 # [vocab]

    # ---------- top-10 raw distribution ----------
    col_a, col_b = st.columns([1, 1])
    with col_a:
        cands = distribution_from_logits(lm.tokenizer, logits, temperature=1.0, top_n=10)
        st.plotly_chart(_bar(cands, "Top-10 next tokens  (softmax of raw logits, T = 1)"), width="stretch")
    with col_b:
        st.markdown("**Same logits, with the current sidebar settings applied**")
        probs = process_logits(logits, settings, [])
        surv = int((probs > 0).sum())
        kept = top_candidates(lm.tokenizer, probs, min(10, surv))
        st.plotly_chart(
            _bar(kept, f"T={settings.temperature}, top-k={settings.top_k}, top-p={settings.top_p} → "
                       f"{surv:,} of {logits.numel():,} tokens can be sampled", "#EF553B"),
            width="stretch",
        )

    # ---------- temperature comparison ----------
    st.markdown("#### 🌡️ Effect of temperature on the same logits")
    st.caption("softmax(logits / T): low T sharpens the peak (deterministic), high T flattens it (random).")
    temps = [0.3, 1.0, 2.0]
    cols = st.columns(len(temps))
    for c, T in zip(cols, temps):
        d = distribution_from_logits(lm.tokenizer, logits, temperature=T, top_n=8)
        with c:
            st.plotly_chart(_bar(d, f"T = {T}   (top-1 = {d[0].prob:.0%})"), width="stretch")

    # ---------- top-k walk-through ----------
    st.markdown("#### 🔪 Effect of Top-K")
    k_demo = st.slider("Top-K to preview", 1, 50, 5, key="nt_k")
    d = distribution_from_logits(lm.tokenizer, logits, temperature=1.0, top_n=50)
    mass = sum(c.prob for c in d[:k_demo])
    st.markdown(
        f"With **K = {k_demo}**, sampling is restricted to the tokens "
        + ", ".join(f"`{c.text!r}`" for c in d[:k_demo])
        + f" which together hold **{mass:.1%}** of the probability mass; all other "
          f"{logits.numel() - k_demo:,} tokens get probability 0."
    )

    # ---------- last answer trace ----------
    st.markdown("#### 🧾 Trace of the last chat answer")
    result = st.session_state.last_result
    if result is None:
        st.caption("Ask something in the Chat tab first.")
        return
    st.dataframe(trace_dataframe(result), width="stretch", hide_index=True, height=360)

    # probability of chosen token along the answer
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=[s.raw_prob for s in result.trace], mode="lines+markers",
                             name="raw P(chosen)", line=dict(color="#636EFA")))
    fig.add_trace(go.Scatter(y=[s.chosen.prob for s in result.trace], mode="lines+markers",
                             name="P(chosen) after T / top-k / top-p", line=dict(color="#EF553B")))
    fig.update_layout(title="How confident was the model at each generated token?",
                      xaxis_title="generation step", yaxis_title="probability",
                      height=320, margin=dict(l=10, r=10, t=50, b=30), yaxis=dict(range=[0, 1.02]))
    st.plotly_chart(fig, width="stretch")


# ================================================================================================
# ui/attention_tab.py
# ================================================================================================
# 🔍 Self-attention visualisation tab.
ATT_DEFAULT_TEXT = "The cat sat on the mat because it was tired."


def render_attention(lm: LoadedModel) -> None:
    st.subheader("🔍 Self-attention: how tokens look at each other")
    st.markdown(
        "For every token (a **query**) the model computes a weight for every earlier token (the **keys**). "
        "The weights of one row sum to 1 and say *where this token gathers information from*. "
        "Decoder models are **causal**: a token can only attend to itself and the tokens before it, "
        "which is why the upper-right triangle is always empty."
    )

    text = st.text_input("Sentence to analyse (max 48 tokens)", value=ATT_DEFAULT_TEXT, key="att_text")
    if not text.strip():
        return

    cache_key = (lm.name, text)
    cache = st.session_state.attention_cache
    if cache_key not in cache:
        cache.clear()
        with st.spinner("Running a forward pass with output_attentions=True …"):
            cache[cache_key] = compute_attention(lm, text)
    data = cache[cache_key]

    c1, c2 = st.columns([1, 1])
    layer = c1.slider("Layer", 0, data.num_layers - 1, min(data.num_layers - 1, max(0, data.num_layers // 3)))
    head_opts = ["mean of all heads"] + [f"head {h}" for h in range(data.num_heads)]
    head_choice = c2.selectbox("Head", head_opts, index=0)
    head = None if head_choice.startswith("mean") else int(head_choice.split()[-1])

    st.plotly_chart(attention_heatmap(data, layer, head), width="stretch")

    st.markdown("#### 🎯 Focus on one token")
    q = st.selectbox("Query token", list(range(len(data.tokens))),
                     index=len(data.tokens) - 1,
                     format_func=lambda i: f"{i}: {data.tokens[i]}")
    st.plotly_chart(token_focus_bar(data, layer, head, q), width="stretch")

    with st.expander("📊 Which heads in this layer are 'focused' vs 'spread out'?"):
        ent = head_summary(data, layer)
        df = pd.DataFrame({"head": list(range(len(ent))), "mean attention entropy": ent.round(3)})
        df["behaviour"] = ["focused (few tokens)" if e < ent.mean() else "diffuse (many tokens)" for e in ent]
        st.dataframe(df.sort_values("mean attention entropy"), width="stretch", hide_index=True)
        st.caption("Low entropy = the head puts most of its weight on one or two tokens; "
                   "high entropy = it averages over many tokens.")

    st.markdown("---")
    st.markdown("### The maths behind the picture")
    st.markdown(
        "Every token vector **x_i** is projected three times: "
        "**Query** q_i = x_i · W_Q, **Key** k_i = x_i · W_K, **Value** v_i = x_i · W_V."
    )
    st.code("Attention(Q, K, V) = softmax( (Q · Kᵀ) / sqrt(d_k) + M ) · V", language="text")
    st.markdown(
        """
* **Q · Kᵀ** measures how much each query *matches* each key (a similarity score).
* **sqrt(d_k)** keeps the scores in a sane range; **M** is the **causal mask** (−∞ above the diagonal).
* **softmax** turns the scores of one row into the weights you see in the heat-map.
* The output of a token is the **weighted average of the values** — it now contains information from the tokens it attended to.
* This happens in **every head of every layer**; different heads specialise (previous token, syntax, co-reference such as *it → cat*, …).
"""
    )


# ================================================================================================
# ui/lab_tab.py
# ================================================================================================
# 🧪 Parameter Lab: run the same prompt with different temperature / top-k values.
LAB_DEFAULT_PROMPT = "Explain in two sentences what a neural network is."

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


def render_lab(lm: LoadedModel, settings: GenerationSettings) -> None:
    st.subheader("🧪 Parameter Lab: same prompt, different generation parameters")
    st.caption("Runs use the current sidebar settings as a base, and only the chosen parameter is changed. "
               "Conversation memory is NOT used here so the comparison is fair.")

    prompt = st.text_area("Prompt", value=LAB_DEFAULT_PROMPT, height=70, key="lab_prompt")
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


# ================================================================================================
# ui/dashboard_tab.py
# ================================================================================================
# 📊 Dashboard tab: session statistics.
def render_dashboard(lm: LoadedModel, settings: GenerationSettings, memory: dict) -> None:
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


# ================================================================================================
# ui/architecture_tab.py
# ================================================================================================
# 🏗️ System architecture (Plotly diagram, pure Python) + technical Q&A.
def architecture_figure(num_layers: int, num_heads: int, vocab_size: int) -> go.Figure:
    """Pipeline diagram drawn with Plotly shapes + annotations (no DOT / HTML)."""
    # key: (x, y, width, height, title, subtitle, colour)
    boxes = {
        "user":   (1.05, 4.35, 1.8, 0.8, "User question", "", "#FFD6A5"),
        "memory": (1.05, 3.05, 1.8, 0.9, "Conversation memory", "session_state.messages", "#FFD6A5"),
        "prompt": (3.35, 3.7, 2.0, 0.9, "Prompt builder", "chat template, last N turns", "#CAFFBF"),
        "tok":    (5.65, 3.7, 2.0, 0.9, "Tokenizer (BPE)", "text → tokens → IDs", "#9BF6FF"),
        "emb":    (7.85, 3.7, 1.7, 0.9, "Embeddings", "+ positions", "#BDB2FF"),
        "blocks": (10.2, 3.7, 2.4, 1.0, f"{num_layers} × Decoder block",
                   f"self-attention ({num_heads} heads) → FFN", "#BDB2FF"),
        "head":   (10.2, 1.6, 2.4, 0.9, "LM head", f"vector → {vocab_size:,} logits", "#BDB2FF"),
        "samp":   (7.45, 1.6, 2.5, 0.9, "Sampler", "softmax → T → top-k → top-p", "#FFC6FF"),
        "detok":  (4.85, 1.6, 1.8, 0.9, "Detokenizer", "IDs → text", "#9BF6FF"),
        "answer": (2.35, 1.6, 2.0, 0.9, "Streamed answer", "+ dashboard", "#FFD6A5"),
    }
    # (from, to, label, dashed)
    edges = [
        ("user", "prompt", "", False),
        ("memory", "prompt", "", False),
        ("prompt", "tok", "", False),
        ("tok", "emb", "", False),
        ("emb", "blocks", "", False),
        ("blocks", "head", "", False),
        ("head", "samp", "", False),
        ("samp", "detok", "EOS / max tokens", False),
        ("detok", "answer", "", False),
        ("answer", "memory", "store turn", True),
        ("samp", "blocks", "append token, repeat (KV-cache)", True),
    ]

    fig = go.Figure()
    for x, y, w, h, title, subtitle, color in boxes.values():
        fig.add_shape(type="rect", x0=x - w / 2, x1=x + w / 2, y0=y - h / 2, y1=y + h / 2,
                      fillcolor=color, line=dict(color="#555555", width=1))
        fig.add_annotation(x=x, y=y + (0.13 if subtitle else 0), text=title, showarrow=False,
                           font=dict(size=12, color="#111111"))
        if subtitle:
            fig.add_annotation(x=x, y=y - 0.2, text=subtitle, showarrow=False,
                               font=dict(size=10, color="#333333"))

    def _port(src: str, dst: str) -> tuple[float, float, float, float]:
        """Start/end points on the box borders facing each other."""
        sx, sy, sw, sh, *_ = boxes[src]
        dx, dy, dw, dh, *_ = boxes[dst]
        if abs(dx - sx) >= abs(dy - sy):            # mostly horizontal
            sign = 1 if dx > sx else -1
            return sx + sign * sw / 2, sy, dx - sign * dw / 2, dy
        sign = 1 if dy > sy else -1                 # mostly vertical
        return sx, sy + sign * sh / 2, dx, dy - sign * dh / 2

    for src, dst, label, dashed in edges:
        x0, y0, x1, y1 = _port(src, dst)
        if dashed:
            fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1,
                          line=dict(color="#888888", width=1.5, dash="dash"))
            # short solid arrow-head at the end of the dashed line
            fig.add_annotation(x=x1, y=y1, ax=x0 + 0.9 * (x1 - x0), ay=y0 + 0.9 * (y1 - y0),
                               xref="x", yref="y", axref="x", ayref="y", text="", showarrow=True,
                               arrowhead=3, arrowsize=1.3, arrowwidth=1.5, arrowcolor="#888888")
        else:
            fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                               text="", showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=1.6,
                               arrowcolor="#444444")
        if label:
            fig.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2 + 0.18, text=label, showarrow=False,
                               font=dict(size=10, color="#444444"), bgcolor="rgba(255,255,255,0.85)")

    fig.update_xaxes(visible=False, range=[0, 11.6], fixedrange=True)
    fig.update_yaxes(visible=False, range=[0.9, 5.0], fixedrange=True)
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode=False)
    return fig


QA = [
    ("How does the Transformer generate a response?",
     "The chat history is rendered into a prompt and tokenized into IDs. The IDs are embedded, pass through "
     "all decoder blocks, and the LM head returns one logit per vocabulary entry for the *last position*. "
     "Softmax converts the logits to probabilities, the sampler picks one token, the token is appended to the "
     "context and the model runs again (only on the new token thanks to the KV-cache). The loop stops at the "
     "end-of-sequence token or the max-tokens limit; the IDs are decoded back to text."),
    ("What is the role of tokenization?",
     "Neural networks work on numbers. Byte-Pair Encoding splits text into sub-word units from a fixed "
     "vocabulary and maps each to an integer ID. The same vocabulary is used in reverse to turn output IDs into "
     "text. Sub-words let a ~150k vocabulary spell any word, including rare or misspelled ones."),
    ("What is self-attention and why is it needed?",
     "Self-attention lets every token build its representation from *all previous tokens*, weighted by relevance "
     "(softmax(Q·Kᵀ/√d)·V). It is how the model links 'its' to 'Machine Learning' from a previous turn, and it is "
     "fully parallel, unlike an RNN that reads one token at a time."),
    ("Why multiple heads and multiple layers?",
     "Each head learns a different relation (previous word, subject-verb, co-reference…). Stacking layers lets the "
     "model compose these relations into higher-level meaning: lower layers capture syntax, upper layers semantics."),
    ("What is the causal mask?",
     "During training and generation a token must not see the future. The mask sets attention scores to −∞ "
     "for positions after the current one, so the softmax gives them weight 0 – that is the empty upper triangle "
     "in the attention heat-map."),
    ("What do temperature, top-k and top-p change?",
     "Nothing inside the network – they only reshape the final probability distribution before sampling. "
     "Temperature scales the logits (sharper vs flatter), top-k keeps the K best tokens, top-p keeps the smallest "
     "set with cumulative probability P. Greedy decoding (T=0) always takes the arg-max."),
    ("How does conversation memory work?",
     "The Transformer itself is stateless. The app stores every turn in Streamlit's session_state and, for each new "
     "question, re-sends the last N turns inside the prompt (using the model's chat template). Turning memory off "
     "sends only the new question, so follow-ups like 'its main types' can no longer be resolved."),
    ("What is the KV-cache?",
     "Keys and values of already-processed tokens are stored so that each new step only computes attention for the "
     "newest token instead of recomputing the whole sequence. This makes generation O(n) per step instead of O(n²)."),
    ("Why run on the GPU?",
     "Every generated token needs a full forward pass through all layers – hundreds of large matrix multiplications. "
     "GPUs execute these in parallel; this project is configured to run exclusively on CUDA with bfloat16 weights."),
]


def render_architecture(lm: LoadedModel) -> None:
    st.subheader("🏗️ System architecture")
    info = get_model_info(lm)
    st.plotly_chart(
        architecture_figure(info["num_layers"], info["num_attention_heads"], info["vocab_size"]),
        width="stretch",
    )

    st.markdown(
        """
#### Pipeline, step by step
1. **Conversation memory** – all previous turns live in `st.session_state.messages`.
2. **Prompt builder** (`chatbot/memory.py`) – the last *N* turns + the new question are rendered with the
   model's chat template (system / user / assistant roles) and trimmed to the context budget.
3. **Tokenizer** (`chatbot/tokenization.py`) – text → sub-word tokens → integer IDs.
4. **Transformer** (`chatbot/model_manager.py`) – embeddings + positional info → *L* decoder blocks
   (masked multi-head self-attention → feed-forward) → LM head → logits for every vocabulary token.
5. **Sampler** (`chatbot/generation.py`) – repetition penalty → temperature → top-k → top-p → softmax → choose
   a token. Every step is recorded (the *next-token trace*).
6. **Loop** – the chosen token is appended and the model is run again with the KV-cache until EOS / max tokens.
7. **Detokenizer** – IDs → text, streamed to the chat window; statistics go to the dashboard.
8. **Attention viewer** (`chatbot/attention.py`) – an extra forward pass with `output_attentions=True`
   exposes the weights of every layer and head for the heat-map.

#### Project layout
```
app.py                 Streamlit entry point (tabs + sidebar)
chatbot/config.py      model registry, default parameters, prompts
chatbot/model_manager  GPU-only loading, architecture facts
chatbot/tokenization   tokens / IDs / embedding preview
chatbot/generation     hand-written sampling loop with trace
chatbot/attention      attention extraction + plotly heat-maps
chatbot/memory         chat history → prompt
chatbot/stats          dashboard statistics
ui/*.py                one module per tab
```
"""
    )

    st.markdown("#### ❓ Technical questions & answers")
    for q, a in QA:
        with st.expander(q):
            st.markdown(a)

# ================================================================================================
# app.py  (main entry point)
# ================================================================================================
init_state()
settings, memory = render_sidebar()
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
    render_chat(lm, settings, memory)
with tabs[1]:
    render_tokenization(lm)
with tabs[2]:
    render_model(lm)
with tabs[3]:
    render_next_token(lm, settings)
with tabs[4]:
    render_attention(lm)
with tabs[5]:
    render_lab(lm, settings)
with tabs[6]:
    render_dashboard(lm, settings, memory)
with tabs[7]:
    render_architecture(lm)