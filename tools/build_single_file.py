"""Build the SINGLE-FILE version of the chatbot from the modular sources.

    python tools/build_single_file.py

Output: single_file/transformer_chatbot_app.py  (run with `streamlit run single_file/transformer_chatbot_app.py`)

The modular version (chatbot/ + ui/ + app.py) is the source of truth; this script
concatenates the modules, removes the package-internal imports, hoists the external
imports to the top and renames the few identifiers that would collide in one namespace.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "single_file" / "transformer_chatbot_app.py"

# (module path, tag used to rename its `render` function)
ORDER = [
    ("chatbot/config.py", None),
    ("chatbot/model_manager.py", None),
    ("chatbot/tokenization.py", None),
    ("chatbot/generation.py", None),
    ("chatbot/attention.py", None),
    ("chatbot/memory.py", None),
    ("chatbot/stats.py", None),
    ("ui/state.py", None),
    ("ui/sidebar.py", "sidebar"),
    ("ui/chat_tab.py", "chat"),
    ("ui/tokenization_tab.py", "tokenization"),
    ("ui/model_tab.py", "model"),
    ("ui/next_token_tab.py", "next_token"),
    ("ui/attention_tab.py", "attention"),
    ("ui/lab_tab.py", "lab"),
    ("ui/dashboard_tab.py", "dashboard"),
    ("ui/architecture_tab.py", "architecture"),
]

# identifiers that exist in more than one module → make them unique
RENAMES = {
    "ui/tokenization_tab.py": {"DEFAULT_TEXT": "TOK_DEFAULT_TEXT"},
    "ui/attention_tab.py": {"DEFAULT_TEXT": "ATT_DEFAULT_TEXT"},
    "ui/next_token_tab.py": {"DEFAULT_PROMPT": "NT_DEFAULT_PROMPT"},
    "ui/lab_tab.py": {"DEFAULT_PROMPT": "LAB_DEFAULT_PROMPT"},
}

FUTURE = "from __future__ import annotations"
INTERNAL_PREFIXES = (".", "chatbot", "ui")

# parenthesised import lists may contain a comment with nested parentheses → allow one nesting level
IMPORT_RE = re.compile(
    r"^(?:from\s+(\S+)\s+import\s+(?:\((?:[^()]|\([^()]*\))*\)|[^\n]+)|import\s+(\S+)[^\n]*)\n", re.M
)
DOCSTRING_RE = re.compile(r'^\s*("""|\'\'\')(.*?)\1\n', re.S)


def is_internal(module: str) -> bool:
    return module.startswith(".") or module.split(".")[0] in ("chatbot", "ui")


def process(path: Path, tag: str | None, ext_imports: list[str]) -> str:
    text = path.read_text(encoding="utf-8")

    # module docstring → comment banner
    m = DOCSTRING_RE.match(text)
    doc = ""
    if m:
        doc = "\n".join("# " + line if line.strip() else "#" for line in m.group(2).strip().splitlines())
        text = text[m.end():]

    # pull out every top-level import
    def _grab(match: re.Match) -> str:
        stmt = match.group(0)
        mod = match.group(1) or match.group(2)
        if FUTURE in stmt:
            return ""
        if is_internal(mod):
            return ""
        if stmt not in ext_imports:
            ext_imports.append(stmt)
        return ""

    text = IMPORT_RE.sub(_grab, text)

    # rename colliding identifiers
    for old, new in RENAMES.get(path.relative_to(ROOT).as_posix(), {}).items():
        text = re.sub(rf"\b{old}\b", new, text)
    if tag:
        text = re.sub(r"\bdef render\(", f"def render_{tag}(", text)

    banner = "# " + "=" * 96 + f"\n# {path.relative_to(ROOT).as_posix()}\n" + "# " + "=" * 96
    return f"{banner}\n{doc}\n{text.strip()}\n"


def build_main(ext_imports: list[str]) -> tuple[str, str]:
    """Return (set_page_config statement, main body) derived from app.py."""
    text = (ROOT / "app.py").read_text(encoding="utf-8")
    m = DOCSTRING_RE.match(text)
    if m:
        text = text[m.end():]
    cfg = re.search(r"st\.set_page_config\((?:[^()]|\([^()]*\))*\)", text, re.S)
    page_config = cfg.group(0) if cfg else ""
    text = text.replace(page_config, "") if page_config else text
    text = IMPORT_RE.sub(lambda mm: "" if is_internal(mm.group(1) or mm.group(2)) else (ext_imports.append(mm.group(0)) or ""), text)
    # `from ui.state import get_model, init_state  # noqa` and similar lines are already stripped
    text = re.sub(r"^\s*#\s*noqa.*$", "", text, flags=re.M)
    text = text.replace("sidebar.render()", "render_sidebar()")
    for _, tag in ORDER:
        if tag and tag != "sidebar":
            text = text.replace(f"{tag}_tab.render(", f"render_{tag}(")
    return page_config, text.strip()


def main() -> None:
    ext_imports: list[str] = []
    sections = [process(ROOT / rel, tag, ext_imports) for rel, tag in ORDER]
    page_config, main_body = build_main(ext_imports)

    # keep a stable, readable import order: stdlib-ish first, then third-party
    def _key(stmt: str) -> tuple[int, str]:
        mod = re.match(r"(?:from|import)\s+(\S+)", stmt).group(1)
        third = mod.split(".")[0] in {"torch", "transformers", "streamlit", "plotly", "pandas", "numpy"}
        return (1 if third else 0, mod)

    # merge `from X import a` + `from X import b, c`  →  `from X import a, b, c`
    merged: dict[str, list[str]] = {}
    others: list[str] = []
    for stmt in dict.fromkeys(ext_imports):
        m = re.match(r"from\s+(\S+)\s+import\s+([^\n(]+)\n", stmt)
        if m:
            names = merged.setdefault(m.group(1), [])
            for n in m.group(2).split(","):
                n = n.strip()
                if n and n not in names:
                    names.append(n)
        else:
            others.append(stmt)
    stmts = others + [f"from {mod} import {', '.join(sorted(names))}\n" for mod, names in merged.items()]
    imports = "".join(sorted(stmts, key=_key))

    header = f'''"""Intelligent Transformer-Based Chatbot — SINGLE-FILE version.

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

{FUTURE}

{imports}

{page_config}

'''
    footer = "\n# " + "=" * 96 + "\n# app.py  (main entry point)\n# " + "=" * 96 + "\n" + main_body + "\n"
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(header + "\n\n".join(sections) + footer, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.1f} KB, {sum(1 for _ in open(OUT, encoding='utf-8'))} lines)")


if __name__ == "__main__":
    main()
