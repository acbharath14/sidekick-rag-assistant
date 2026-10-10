"""Episodic memory: long-term conversational recall via LanceDB.

Stores past Q&A pairs with metadata (timestamp, question, answer) in a
persistent LanceDB table (`.lancedb_episodic/`). When answering a new
question, retrieves the most relevant past conversations — enabling recall
like "what did we discuss last week about Playwright?"

Why LanceDB over FAISS for this:
- Native metadata: question, answer, timestamp in one table (no separate
  JSONL file to keep in sync)
- SQL filtering: query by time range, e.g. recent conversations only
- Built-in persistence and versioning
- Embedded: no server, like SQLite

This complements the short-term `history` (last 6 turns). Episodic memory
persists across sessions and scales to hundreds of conversations.

Disable with RAG_DEMO_EPISODIC=0.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings, get_embeddings, get_settings


def _episodic_dir(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    base = settings.index_dir.parent if hasattr(settings.index_dir, "parent") else Path(".")
    return base / ".lancedb_episodic"


def _is_enabled() -> bool:
    return os.environ.get("RAG_DEMO_EPISODIC", "1") == "1"


class EpisodicMemory:
    """Persistent LanceDB table of past conversations."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.dir = _episodic_dir(self.settings)
        self._db = None
        self._table = None
        self._embeddings = None
        if _is_enabled():
            self._connect()

    def _connect(self) -> None:
        """Connect to (or create) the LanceDB table."""
        try:
            import lancedb
            import pyarrow as pa

            self.dir.mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(str(self.dir))
            self._embeddings = get_embeddings(self.settings)

            # Define schema: vector + metadata in one table.
            schema = pa.schema([
                pa.field("vector", pa.list_(pa.float32())),
                pa.field("question", pa.string()),
                pa.field("answer", pa.string()),
                pa.field("ts", pa.string()),
            ])
            try:
                self._table = self._db.open_table("episodes")
            except Exception:
                self._table = self._db.create_table(
                    "episodes", schema=schema, mode="create"
                )
        except Exception:
            self._db = None
            self._table = None

    def _embed(self, text: str) -> list[float]:
        """Embed text using the configured embedding model."""
        if self._embeddings is None:
            return []
        try:
            # get_embeddings returns a LangChain embeddings object.
            vec = self._embeddings.embed_query(text)
            return [float(x) for x in vec]
        except Exception:
            return []

    def add(self, question: str, answer: str) -> None:
        """Store a Q&A pair in episodic memory."""
        if not _is_enabled() or self._table is None:
            return
        if not question.strip() or not answer.strip():
            return
        if len(question.strip()) < 3 or len(answer.strip()) < 10:
            return

        ts = datetime.now(timezone.utc).isoformat()
        # Embed the combined Q&A for retrieval.
        episode_text = f"Q: {question}\nA: {answer}"
        vector = self._embed(episode_text)
        if not vector:
            return

        try:
            import pyarrow as pa

            self._table.add([{
                "vector": vector,
                "question": question[:500],
                "answer": answer[:2000],
                "ts": ts,
            }])
        except Exception:
            pass  # best-effort; memory is auxiliary

    def search(self, query: str, k: int = 3) -> list[dict]:
        """Retrieve relevant past conversations.

        Returns list of {"question": str, "answer": str, "ts": str}.
        """
        if not _is_enabled() or self._table is None:
            return []
        if not query.strip() or len(query.strip()) < 3:
            return []
        try:
            vector = self._embed(query)
            if not vector:
                return []
            results = (
                self._table.search(vector)
                .limit(k)
                .to_list()
            )
            return [
                {
                    "question": r.get("question", ""),
                    "answer": r.get("answer", ""),
                    "ts": r.get("ts", ""),
                }
                for r in results
            ]
        except Exception:
            return []

    def count(self) -> int:
        """Number of stored episodes."""
        if self._table is None:
            return 0
        try:
            return self._table.count_rows()
        except Exception:
            return 0

    def clear(self) -> None:
        """Wipe episodic memory."""
        try:
            import shutil
            if self.dir.exists():
                shutil.rmtree(self.dir)
            self._db = None
            self._table = None
        except OSError:
            pass
