"""Mermaid diagram support for chat answers — Streamlit-free by design.

The model writes diagrams as fenced ```mermaid blocks (Mermaid is text, so
no vision model or new model is needed — the regular chat model generates
the syntax). This module does the three things that do not need Streamlit:

- split a message into markdown segments and Mermaid code segments,
- validate a Mermaid flowchart with cheap heuristic checks (this is NOT a
  full Mermaid parser — it catches the breakage small local models produce
  most often, and anything it cannot prove valid is shown as code instead
  of rendered),
- build the self-contained HTML document the UI renders in an iframe.

Rendering itself lives in app.py (st.components.v1.html). The raw code is
always shown next to any rendered diagram, so a failed render — or an
offline browser that cannot reach the mermaid.js CDN — is never a dead end.
"""

import html
import re

# Trigger for the "Show as diagram" button: the pending question is the
# original user question with this prefix. The question block in app.py
# strips the prefix before asking and appends DIAGRAM_INSTRUCTION.
DIAGRAM_PREFIX = "📊 Diagram: "

# Keywords that make a plain chat question a diagram request (explicit
# trigger only — v1 never auto-detects "this deserves a diagram").
_DIAGRAM_WORDS = ("diagram", "flowchart", "flow chart", "mermaid")

# Instruction appended to the question actually sent to the model when a
# diagram is requested. It is deliberately strict about syntax: small
# local models break Mermaid most often with unquoted labels carrying
# parentheses/brackets and with stray special characters. Grounding rule
# included: a flow the passages do not support must not be drawn.
DIAGRAM_INSTRUCTION = (
    "Also include ONE Mermaid flowchart that pictorially shows the process "
    "or relationships from the retrieved passages. Rules: start the block "
    "with 'flowchart TD' on its own line; one node per step; every label in "
    "double quotes; no parentheses, brackets, semicolons, or special "
    "characters inside labels; no HTML except <br/>; at most 12 nodes; use "
    "only steps supported by the retrieved passages — if the passages do "
    "not support a flow, say so instead of drawing one."
)

_MERMAID_CDN = "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"

# First content line must be a flowchart/graph header with a direction.
_HEADER_RE = re.compile(r"^(flowchart|graph)\s+(TD|TB|LR|RL|BT)\s*$")
# Any HTML tag other than <br/> inside the code (script included).
_TAG_RE = re.compile(r"<(?!br\s*/?)[a-zA-Z]")
_BRACKETS = {"[": "]", "(": ")", "{": "}"}


def is_diagram_request(question: str) -> bool:
    """True when a chat question explicitly asks for a diagram."""
    lowered = question.lower()
    return any(word in lowered for word in _DIAGRAM_WORDS)


def split_message(text: str):
    """Split message text into ("md", chunk) / ("mermaid", code) segments.

    Only CLOSED ```mermaid fences become diagram segments. An unclosed
    fence, and every non-mermaid fence, stays markdown — the UI then shows
    it as code, exactly as Streamlit would have without this feature.
    """
    segments = []
    md_lines = []
    mermaid_lines = None  # None = outside a mermaid fence
    fence_line = ""

    def flush_md():
        if md_lines:
            segments.append(("md", "".join(md_lines)))
            md_lines.clear()

    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if mermaid_lines is None:
            if stripped.startswith("```mermaid"):
                flush_md()
                mermaid_lines = []
                fence_line = line
            else:
                md_lines.append(line)
        else:
            if stripped.startswith("```"):
                segments.append(("mermaid", "".join(mermaid_lines).rstrip("\n")))
                mermaid_lines = None
            else:
                mermaid_lines.append(line)
    if mermaid_lines is not None:
        # Unclosed fence: not a diagram. Hand the fence back as markdown.
        md_lines.append(fence_line)
        md_lines.extend(mermaid_lines)
        mermaid_lines = None
    flush_md()
    return segments


def extract_mermaid_blocks(text: str):
    """All closed ```mermaid blocks in a message, in order."""
    return [code for kind, code in split_message(text) if kind == "mermaid"]


def validate_mermaid(code: str):
    """Heuristic validation of a Mermaid flowchart.

    Returns (ok, reason). True means "safe to attempt a render", not
    "guaranteed to render" — only the browser's mermaid.js knows that for
    sure, which is why the raw code is always shown next to the diagram.
    """
    lines = [line for line in code.strip().splitlines() if line.strip()]
    if not lines:
        return False, "empty diagram"
    if not _HEADER_RE.match(lines[0].strip()):
        return False, "must start with 'flowchart TD' (or TB, LR, RL, BT)"
    if code.count('"') % 2 != 0:
        return False, "unbalanced double quotes in labels"
    if "<script" in code.lower() or "javascript:" in code.lower():
        return False, "unsafe content in diagram"
    if _TAG_RE.search(code):
        return False, "HTML tags other than <br/> are not allowed"
    stack = []
    pairs = {v: k for k, v in _BRACKETS.items()}
    for ch in code:
        if ch in _BRACKETS:
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack[-1] != pairs[ch]:
                return False, "unbalanced brackets"
            stack.pop()
    if stack:
        return False, "unbalanced brackets"
    return True, ""


def render_html(code: str) -> str:
    """Self-contained HTML document rendering one Mermaid diagram.

    The code is HTML-escaped before embedding: the browser decodes the
    entities back when building the DOM, so mermaid.js sees the original
    text (including <br/>), while a stray </script> inside the code can
    never break out of the embedding.
    """
    escaped = html.escape(code)
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<script src='{_MERMAID_CDN}'></script>"
        "<script>mermaid.initialize({startOnLoad:true});</script>"
        "</head><body>"
        f"<div class='mermaid'>{escaped}</div>"
        "</body></html>"
    )
