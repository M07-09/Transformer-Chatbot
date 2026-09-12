"""Session statistics shown on the dashboard."""

from __future__ import annotations

from dataclasses import dataclass, field


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
