"""Loading the pre-trained Transformer and reading its architecture facts."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import MODEL_REGISTRY


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
