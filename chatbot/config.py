"""Central configuration: model registry, default generation settings, prompts."""

from dataclasses import dataclass, asdict

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
