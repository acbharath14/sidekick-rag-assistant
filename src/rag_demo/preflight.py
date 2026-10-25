"""Runtime pre-flight checks (Phase 4, U3).

On startup the app checks Ollama reachability, model presence, and
index presence. When anything is missing, chat input is disabled and a
fix-it banner shows what is missing and how to fix it — instead of
letting the user prompt into a dead end and fail after waiting.

No Streamlit dependency: the checks are pure functions with injectable
backends so tests run without Ollama, a model, or an index. app.py
renders the result; scripts/setup.py reuses the tag parsing.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class PreflightResult:
    """Outcome of one pre-flight run. `problems` holds one human-
    readable line per failure, each with its fix; empty when all green."""

    ollama_ok: bool
    model_ok: bool
    index_ok: bool
    problems: list[str] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return self.ollama_ok and self.model_ok and self.index_ok


def normalize_model_name(name: str) -> str:
    """Ollama lists untagged models with a :latest tag, so `qwen3:8b`
    and `qwen3:8b` compare equal, and bare `qwen3` matches `qwen3:latest`."""
    return name if ":" in name else f"{name}:latest"


def model_in_tags(tags: dict, model: str) -> bool:
    """True if `model` appears in an Ollama /api/tags response payload."""
    wanted = normalize_model_name(model)
    for entry in tags.get("models", []):
        name = entry.get("name", "")
        if name and normalize_model_name(name) == wanted:
            return True
    return False


def fetch_tags(base_url: str, timeout: float = 2) -> dict | None:
    """Ollama /api/tags payload, or None when the server is unreachable
    or answers garbage. Unreachable is a status, not an exception."""
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 — any failure means "not reachable"
        return None


def ollama_reachable(base_url: str) -> bool:
    return fetch_tags(base_url) is not None


def model_present(base_url: str, model: str) -> bool:
    tags = fetch_tags(base_url)
    return tags is not None and model_in_tags(tags, model)


def index_present(index_dir: Path) -> bool:
    """A built index, not just a directory: FAISS save_local always
    writes index.faiss, so its presence is the built-index signal."""
    index_dir = Path(index_dir)
    return index_dir.is_dir() and (index_dir / "index.faiss").exists()


def run_preflight(
    settings,
    corpus: str,
    index_dir: Path,
    ollama_check: Callable[[str], bool] | None = None,
    model_check: Callable[[str, str], bool] | None = None,
    index_check: Callable[[Path], bool] | None = None,
) -> PreflightResult:
    """Check Ollama, model, and index for one question-ready state.

    `settings` needs only .fake, .ollama_base_url, .ollama_model (duck-
    typed so tests can pass a stub). Under RAG_DEMO_FAKE=1 the fake LLM
    needs neither Ollama nor a model, so those two checks are vacuously
    green and only the index is checked. Check callables are injectable
    for tests; defaults do the real HTTP/disk checks.
    """
    ollama_check = ollama_check or ollama_reachable
    model_check = model_check or model_present
    index_check = index_check or index_present

    problems: list[str] = []
    if settings.fake:
        ollama_ok = True
        model_ok = True
    else:
        ollama_ok = ollama_check(settings.ollama_base_url)
        if not ollama_ok:
            problems.append(
                f"Ollama is not reachable at {settings.ollama_base_url} — "
                "start it with `ollama serve`, or run `python scripts/setup.py`."
            )
        # A missing model is only meaningful when Ollama answers; when
        # Ollama is down the single Ollama problem above is enough.
        model_ok = (
            model_check(settings.ollama_base_url, settings.ollama_model)
            if ollama_ok
            else False
        )
        if ollama_ok and not model_ok:
            problems.append(
                f"Model {settings.ollama_model} is not pulled — run "
                f"`ollama pull {settings.ollama_model}`, or run "
                "`python scripts/setup.py`."
            )
    index_ok = index_check(Path(index_dir))
    if not index_ok:
        problems.append(
            f"No index for '{corpus}' at {index_dir} — build it from the "
            "Sources panel (Build demo index) or run "
            f"`python -m rag_demo.ingest --source {corpus}`."
        )
    return PreflightResult(
        ollama_ok=ollama_ok,
        model_ok=model_ok,
        index_ok=index_ok,
        problems=problems,
    )
