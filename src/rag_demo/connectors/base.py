"""Shared connector interface: every enterprise source implements this.

A connector fetches documents from one external system (Confluence,
Outlook, GitHub, ...) and yields them as :class:`ConnectorDoc` records.
The sync layer (``sync.py``) merges those records into the local store
and rebuilds the FAISS index through the existing ingest pipeline, so a
new source only needs a client + a mock + tests — no indexing code.

Token rules (apply to every connector):

- Tokens come from environment variables only, never from a CLI flag
  (so they never land in shell history) and never from a file in the
  repo.
- A token is held in memory on the connector instance, sent only as an
  ``Authorization`` header, and never written to disk, logs, or output.
- Tests use literal ``FAKE-...`` tokens only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from langchain_core.documents import Document


@dataclass
class ConnectorDoc:
    """One fetchable unit from an external system.

    ``source_label`` is the human-readable citation shown in answers,
    e.g. ``confluence:OPS/Deploy checklist`` or
    ``github:acme/widgets#42``. ``doc_id`` is the stable identity used
    for dedup in the store; it defaults to the label.
    """

    title: str
    text: str
    source_label: str
    updated_at: str | None = None  # ISO 8601, when the source provides one
    url: str | None = None
    doc_id: str | None = None

    def __post_init__(self) -> None:
        if self.doc_id is None:
            self.doc_id = self.source_label


class Connector(ABC):
    """One external system. Implementations are small HTTP clients."""

    source_name: str = "unnamed"

    def __init__(self) -> None:
        # Working state, updated by fetch_since as it runs (e.g. per-repo
        # blob SHAs for incremental sync). The sync layer reads it back
        # and persists it; see sync.py.
        self.state: dict = {}
        # Doc ids the source currently holds (README + docs for GitHub).
        # The sync layer prunes stored docs for this source that are no
        # longer present. Only set for kinds the source can enumerate.
        self.present_doc_ids: set[str] = set()

    @abstractmethod
    def connect(self, config: dict) -> None:
        """Validate config and store credentials. Raises ValueError with
        a helpful message when something required is missing."""

    @abstractmethod
    def fetch_since(
        self, since: datetime | None, state: dict | None = None
    ):
        """Yield ConnectorDoc records changed since ``since``.

        ``since`` is None for a full sync. ``state`` is the previous
        per-source state dict loaded by the sync layer (may be empty);
        the connector updates ``self.state`` in place as it runs, and
        the sync layer persists it afterwards.
        """


def to_documents(docs: list[ConnectorDoc]) -> list[Document]:
    """Adapt connector records to langchain Documents for indexing."""
    return [
        Document(
            page_content=d.text,
            metadata={
                "source": d.source_label,
                "doc_id": d.doc_id,
                "allowed_groups": ["*"],
            },
        )
        for d in docs
    ]
