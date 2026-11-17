"""Connector framework: the interface every enterprise source implements.

A connector fetches documents from one external system (Confluence, Outlook,
...) and yields them as ConnectorDoc records. The sync module feeds those
records through the existing chunk/embed/index pipeline, so a connector never
touches indexing code itself.

Credentials come from environment variables only and are held in memory on
the connector instance. They are never logged, never written to disk, and
never appear in output. Tests use literal fake values and run against the
local mock servers in tests/mock_servers.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ConnectorDoc:
    """One document fetched from an external system.

    `source_label` is the human-readable citation shown to the user, e.g.
    ``confluence:OPS/TMS deploy checklist`` or ``outlook:TMS deploy window``.
    `updated_at` is ISO 8601 (UTC, ``...Z``) and drives incremental sync:
    a later sync only fetches documents newer than the stored timestamp.
    `url` is the link back to the source system when it has one.
    """

    title: str
    text: str
    source_label: str
    updated_at: str
    url: str | None = None
    doc_id: str | None = None


class Connector(ABC):
    """One external system. Subclasses implement three methods."""

    source_name: str = "unnamed"

    @abstractmethod
    def connect(self, config: dict | None = None) -> None:
        """Read credentials/config (env vars by default) and validate them.

        Raises ValueError when a required value is missing. The credential
        is kept on the instance only — never logged, never written to disk.
        """

    @abstractmethod
    def fetch_since(self, since: str | None) -> Iterable[ConnectorDoc]:
        """Yield documents updated after `since` (ISO 8601), or all of them
        when `since` is None (first sync)."""


def to_documents(docs: Iterable[ConnectorDoc]):
    """Convert connector records to langchain Documents for the pipeline.

    The chunker/indexer reads `source` for citations and `doc_id` for the
    manifest, so both are stamped here. `allowed_groups` is ["*"]: the
    connector fetches with the token owner's credentials, so the index
    reflects exactly what that user can see (permission-aware by
    construction).
    """
    from langchain_core.documents import Document

    out = []
    for d in docs:
        out.append(
            Document(
                page_content=d.text,
                metadata={
                    "source": d.source_label,
                    "doc_id": d.doc_id or d.source_label,
                    "allowed_groups": ["*"],
                    "last_modified": d.updated_at,
                },
            )
        )
    return out
