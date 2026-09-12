"""Smoke test of the core package without Streamlit.

    python test_core.py [model_name]
"""

import sys
import time

import torch

from chatbot import (
    DEFAULT_MODEL,
    GenerationSettings,
    build_prompt,
    compute_attention,
    distribution_from_logits,
    generate,
    get_model_info,
    load_model,
    next_token_logits,
    tokenize_text,
)


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
    lm = load_model(name)
    info = get_model_info(lm)
    print(f"\n== {name} on {lm.device} ({lm.dtype}) loaded in {lm.load_seconds:.1f}s")
    for k in ["num_layers", "num_attention_heads", "embedding_dim", "vocab_size", "total_params", "context_length"]:
        print(f"   {k:22s} {info[k]}")

    # 1) tokenization
    text = "Machine learning is a subfield of AI."
    toks = tokenize_text(lm.tokenizer, text)
    print("\n== tokens:", [t.pretty for t in toks])
    print("   ids   :", [t.token_id for t in toks])

    # 2) next-token prediction
    ids = lm.tokenizer("The capital of France is", return_tensors="pt", add_special_tokens=False).input_ids
    logits = next_token_logits(lm, ids)
    print("\n== next token after 'The capital of France is':")
    for c in distribution_from_logits(lm.tokenizer, logits, 1.0, 5):
        print(f"   {c.text!r:14s} id={c.token_id:<7d} p={c.prob:.3f}")

    # 3) conversation memory + generation
    history = [{"role": "user", "content": "What is Machine Learning?"}]
    built = build_prompt(lm, history)
    settings = GenerationSettings(temperature=0.7, top_k=50, top_p=0.9, max_new_tokens=60)
    r1 = generate(lm, built.input_ids, settings, built.stop_strings)
    print(f"\n== Q1 ({built.num_tokens} prompt tokens) -> {r1.generated_tokens} tokens in {r1.elapsed:.2f}s "
          f"({r1.tokens_per_second:.1f} tok/s), stop={r1.finished_by}")
    print("   A1:", r1.text)

    history += [{"role": "assistant", "content": r1.text}, {"role": "user", "content": "What are its main types?"}]
    built2 = build_prompt(lm, history)
    r2 = generate(lm, built2.input_ids, settings, built2.stop_strings)
    print(f"\n== Q2 with memory ({built2.num_tokens} prompt tokens, {built2.turns_included} past turns)")
    print("   A2:", r2.text)

    built3 = build_prompt(lm, history, use_memory=False)
    r3 = generate(lm, built3.input_ids, settings, built3.stop_strings)
    print(f"\n== Q2 WITHOUT memory ({built3.num_tokens} prompt tokens)")
    print("   A2':", r3.text)

    # 4) trace
    s = r1.trace[0]
    print(f"\n== first step trace: chosen {s.chosen.text!r} p={s.chosen.prob:.3f}; "
          f"alternatives: {[(c.text, round(c.prob, 3)) for c in s.candidates]}")

    # 5) attention
    att = compute_attention(lm, "The cat sat on the mat because it was tired.")
    print(f"\n== attention tensor: layers={att.num_layers} heads={att.num_heads} seq={len(att.tokens)}")
    print("   row sums (should be 1):", att.matrix(0, 0).sum(axis=1)[:5].round(3))

    # 6) greedy determinism
    g = GenerationSettings(temperature=0.0, max_new_tokens=20)
    a = generate(lm, built.input_ids, g, built.stop_strings).text
    b = generate(lm, built.input_ids, g, built.stop_strings).text
    print("\n== greedy deterministic:", a == b)
    print("\nALL OK")


if __name__ == "__main__":
    t = time.perf_counter()
    main()
    print(f"total {time.perf_counter() - t:.1f}s")
