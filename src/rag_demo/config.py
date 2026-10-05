"""Settings, all overridable via environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent


@dataclass
class Settings:
    """Runtime configuration. `RAG_DEMO_FAKE=1` swaps in deterministic fakes
    (used by tests and CI — no model downloads, no API keys)."""

    docs_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("RAG_DEMO_DOCS_DIR", PROJECT_ROOT / "docs")
        )
    )
    index_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("RAG_DEMO_INDEX_DIR", PROJECT_ROOT / ".faiss_index")
        )
    )
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 4

    # Embeddings: "huggingface" (default) or "fake"
    embedding_provider: str = field(
        default_factory=lambda: os.environ.get("RAG_DEMO_EMBEDDINGS", "huggingface")
    )
    embedding_model: str = field(
        default_factory=lambda: os.environ.get(
            "RAG_DEMO_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )

    # LLM: "ollama" (default) or "fake". Ollama model defaults to qwen3:8b.
    llm_provider: str = field(default_factory=lambda: os.environ.get("RAG_DEMO_LLM", "ollama"))
    ollama_model: str = field(default_factory=lambda: os.environ.get("RAG_DEMO_MODEL", "qwen3:8b"))
    ollama_base_url: str = field(
        default_factory=lambda: os.environ.get("RAG_DEMO_BASE_URL", "http://localhost:11434")
    )

    @property
    def fake(self) -> bool:
        return os.environ.get("RAG_DEMO_FAKE") == "1"


def get_settings() -> Settings:
    s = Settings()
    if s.fake:
        s.embedding_provider = "fake"
        s.llm_provider = "fake"
    return s


def get_embeddings(settings: Settings):
    """Lazy imports keep `import rag_demo` light when deps are missing."""
    if settings.embedding_provider == "fake":
        from langchain_core.embeddings import FakeEmbeddings

        return FakeEmbeddings(size=384)
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def get_llm(settings: Settings):
    if settings.llm_provider == "fake":
        from langchain_core.language_models.fake import FakeListLLM

        return FakeListLLM(
            responses=[
                "Based on the indexed docs: see the cited sources below."
            ]
        )
    from langchain_ollama import OllamaLLM

    return OllamaLLM(model=settings.ollama_model, base_url=settings.ollama_base_url)
