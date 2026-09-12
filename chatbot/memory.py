"""Conversation memory: turn the chat history into the prompt the model sees."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import (
    MAX_CONTEXT_BUDGET,
    PLAIN_AI_PREFIX,
    PLAIN_PREAMBLE,
    PLAIN_STOP_STRINGS,
    PLAIN_USER_PREFIX,
    SYSTEM_PROMPT,
)
from .model_manager import LoadedModel


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
