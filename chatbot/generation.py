"""Next-token prediction and sampling.

The generation loop is written by hand (instead of model.generate) so that the
application can *show* what happens at every step:

    logits  →  repetition penalty  →  temperature  →  top-k  →  top-p
            →  softmax  →  sample / argmax  →  append token  →  repeat
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterator

import torch
import torch.nn.functional as F

from .config import GenerationSettings
from .model_manager import LoadedModel


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
