"""Core logic of the Transformer chatbot (independent of Streamlit)."""

from .config import (
    DEFAULT_MODEL,
    DEFAULT_SETTINGS,
    EXAMPLE_QUESTIONS,
    MODEL_REGISTRY,
    SYSTEM_PROMPT,
    GenerationSettings,
)
from .model_manager import (
    GPUNotAvailableError,
    LoadedModel,
    get_model_info,
    gpu_info,
    human_count,
    layer_structure,
    load_model,
    pick_device,
)
from .tokenization import TokenInfo, embedding_preview, tokenize_text, tokens_figure
from .generation import (
    Candidate,
    GenerationResult,
    StepTrace,
    distribution_from_logits,
    generate,
    generate_stream,
    next_token_logits,
    process_logits,
    top_candidates,
)
from .attention import AttentionData, attention_heatmap, compute_attention, head_summary, token_focus_bar
from .memory import BuiltPrompt, build_prompt
from .stats import SessionStats, TurnRecord

__all__ = [name for name in dir() if not name.startswith("_")]
