"""Mermaid diagram module tests (fake tier): fence extraction, validation,
render HTML escaping. No Streamlit involved — diagrams.py is Streamlit-free
by design, like walkthrough.py."""

import os

os.environ["RAG_DEMO_FAKE"] = "1"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_demo.diagrams import (
    DIAGRAM_INSTRUCTION,
    DIAGRAM_PREFIX,
    extract_mermaid_blocks,
    is_diagram_request,
    render_html,
    split_message,
    validate_mermaid,
)

VALID = (
    'flowchart TD\n'
    '    A["Check status"] --> B["Deploy"]\n'
    '    B --> C["Verify"]\n'
)

PLAN_STYLE = (
    "flowchart TD\n"
    '    Q["❓ How do I roll back?"]\n'
    '    S1["📚 search_docs<br/>rollback query"]\n'
    "    Q --> S1\n"
    '    S1 --> SYN["📝 Synthesize answer"]\n'
)


# ------------------------------------------------------------ splitting ----

def test_single_block_splits_around_markdown():
    text = "Here is the flow:\n```mermaid\n" + VALID + "```\nDone.\n"
    kinds = split_message(text)
    assert kinds[0] == ("md", "Here is the flow:\n")
    assert kinds[1] == ("mermaid", VALID.rstrip("\n"))
    assert kinds[2] == ("md", "Done.\n")


def test_multiple_blocks_in_order():
    text = "```mermaid\n" + VALID + "```\nbetween\n```mermaid\nflowchart LR\n    A --> B\n```\n"
    blocks = extract_mermaid_blocks(text)
    assert len(blocks) == 2
    assert blocks[0].startswith("flowchart TD")
    assert blocks[1].startswith("flowchart LR")


def test_unclosed_fence_is_not_a_diagram():
    text = "Intro\n```mermaid\n" + VALID
    kinds = split_message(text)
    assert all(kind == "md" for kind, _ in kinds)
    assert extract_mermaid_blocks(text) == []
    # The fence and its content survive as markdown (shown as code by UI).
    assert "```mermaid" in "".join(chunk for _, chunk in kinds)


def test_non_mermaid_fences_left_alone():
    text = "```python\nprint(1)\n```\n```mermaid\n" + VALID + "```\n"
    kinds = split_message(text)
    # No trailing markdown after the final fence, so just two segments.
    assert [kind for kind, _ in kinds] == ["md", "mermaid"]
    assert "```python" in kinds[0][1]
    assert extract_mermaid_blocks(text) == [VALID.rstrip("\n")]


def test_mermaid_fence_with_trailing_spaces():
    text = "```mermaid   \n" + VALID + "```\n"
    assert extract_mermaid_blocks(text) == [VALID.rstrip("\n")]


def test_plain_text_has_no_diagrams():
    assert split_message("Just an answer.\n") == [("md", "Just an answer.\n")]
    assert extract_mermaid_blocks("") == []


# ------------------------------------------------------------- trigger -----

def test_diagram_request_keywords():
    assert is_diagram_request("Show this as a diagram")
    assert is_diagram_request("Draw a flowchart of the deploy")
    assert is_diagram_request("Give me the mermaid chart")
    assert is_diagram_request("What is the flow chart for rollback?")
    assert not is_diagram_request("How do I roll back a deploy?")
    assert not is_diagram_request("")


def test_diagram_instruction_constrains_syntax_and_grounding():
    assert "flowchart TD" in DIAGRAM_INSTRUCTION
    assert "double quotes" in DIAGRAM_INSTRUCTION
    assert "retrieved passages" in DIAGRAM_INSTRUCTION
    assert DIAGRAM_PREFIX.startswith("📊")


# ----------------------------------------------------------- validation ----

def test_valid_flowcharts_pass():
    assert validate_mermaid(VALID) == (True, "")
    # Agent-plan style output (quoted labels, <br/>) must also pass.
    assert validate_mermaid(PLAN_STYLE) == (True, "")


def test_invalid_patterns_rejected():
    cases = [
        ("", "empty"),
        ("graph TD\n    A --> B\n", None),  # graph header is fine, actually
        ("A --> B", "header"),
        ('flowchart TD\n    A["unclosed] --> B', "quotes"),
        ("flowchart TD\n    A --> <script>alert(1)</script>", "unsafe"),
        ("flowchart TD\n    A --> B\n    B <div>oops</div>", "HTML"),
        ('flowchart TD\n    A["x" --> B', "brackets"),
    ]
    for code, _label in cases:
        if code.startswith("graph TD"):
            assert validate_mermaid(code)[0] is True
        else:
            ok, reason = validate_mermaid(code)
            assert ok is False, f"expected rejection: {code!r}"
            assert reason, "rejection must carry a human-readable reason"


# ------------------------------------------------------------ rendering ----

def test_render_html_escapes_code():
    out = render_html(PLAN_STYLE)
    assert "mermaid.min.js" in out
    assert "class='mermaid'" in out
    # <br/> inside a label is escaped in the source (browser decodes it
    # back when building the DOM, so mermaid.js still sees a line break).
    assert "&lt;br/&gt;" in out
    assert '["📚 search_docs<br/>rollback query"]' not in out


def test_render_html_cannot_break_out_of_script():
    evil = 'flowchart TD\n    A["x</script><script>alert(1)"]\n'
    out = render_html(evil)
    assert "&lt;/script&gt;" in out
    # Only the template's own two closing script tags remain.
    assert out.count("</script>") == 2
