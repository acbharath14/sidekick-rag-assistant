"""Demo mode vs workspace mode (Phase 4, U1+U2).

Structural separation, not a label. Demo mode serves fictional corpora
for zero-config trial and the deterministic eval corpus; the workspace
path serves the user's own sources and never shows a demo corpus. The
two paths share the chat engine but never share a source list — that is
what makes the separation structural: a demo corpus cannot appear in
the workspace picker, and a workspace source cannot appear in Demo mode.

Groups:
- Demo: Meridian Logistics docs + Confluence mock fixtures. Both are
  fictional, and both are marked fictional everywhere they appear
  (mode label, banner, citations, export).
- Workspace / Documentation: Playwright docs, GitHub repo docs. These
  are real public docs; the GitHub label names the repo explicitly
  (sidekick-rag-assistant itself) so the picker never reads as a
  generic "GitHub docs".
- Workspace / Your workspace: Confluence (real), Jira, Outlook, AWS S3.
  Each shows a status: Not connected / Index building / Ready.

Entry point (approved 2026-10-24, revisitable): first open lands in
Workspace if any workspace source is connected, else Demo. "Connected"
here means a built index exists on disk for that source — credentials
alone do not count, because without an index there is nothing to ask.
Demo mode carries its fictional marking persistently, not just on
first open; switching mode is one visible control in the sidebar.

This module has no Streamlit dependency so the mode rules are unit-
testable; app.py renders them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

# Fictional demo corpora. The workspace picker must never contain these.
DEMO_CORPORA: tuple[str, ...] = ("meridian", "confluence-mock")

# Real public documentation sources, part of the workspace path.
DOCUMENTATION_SOURCES: tuple[str, ...] = ("playwright-docs", "github-docs")

# Real enterprise sources, part of the workspace path.
WORKSPACE_ENTERPRISE_SOURCES: tuple[str, ...] = (
    "confluence",
    "jira",
    "outlook",
    "aws-s3",
)

# Every non-demo corpus. The workspace picker is exactly this set.
WORKSPACE_CORPORA: tuple[str, ...] = (
    DOCUMENTATION_SOURCES + WORKSPACE_ENTERPRISE_SOURCES
)

# Picker labels. Demo labels carry the fictional marking; the GitHub
# label names the repo explicitly instead of reading as generic.
CORPUS_DISPLAY: dict[str, str] = {
    "meridian": "Meridian Logistics (fictional demo)",
    "confluence-mock": "Confluence mock (fictional demo)",
    "playwright-docs": "Playwright docs",
    "github-docs": "GitHub repo docs (sidekick-rag-assistant)",
    "confluence": "Confluence",
    "jira": "Jira issues",
    "outlook": "Outlook emails",
    "aws-s3": "AWS S3 documents",
}

DEMO_MODE = "demo"
WORKSPACE_MODE = "workspace"


def is_demo_corpus(corpus: str) -> bool:
    """True for fictional demo corpora (never shown on the workspace path)."""
    return corpus in DEMO_CORPORA


def is_workspace_corpus(corpus: str) -> bool:
    """True for real sources (never shown in Demo mode)."""
    return corpus in WORKSPACE_CORPORA


def default_mode(connected_workspace_sources) -> str:
    """First-open entry point: Workspace if any workspace source is
    connected, else Demo. `connected_workspace_sources` is the list of
    workspace sources with a built index (see connected_sources)."""
    return WORKSPACE_MODE if connected_workspace_sources else DEMO_MODE


def connected_sources(
    index_parent: Path, index_dirs: Mapping[str, str]
) -> list[str]:
    """Workspace sources with a built index on disk. `index_parent` is
    the directory holding the per-source index dirs (settings.index_dir
    .parent), and `index_dirs` maps source key -> index dir name (the
    INDEX_DIRS table from ingest). Demo corpora are never returned."""
    return [
        source
        for source in WORKSPACE_CORPORA
        if source in index_dirs and (index_parent / index_dirs[source]).exists()
    ]


def source_status(exists: bool, building: bool = False) -> str:
    """One of the three workspace status labels. `building` is set while
    a build runs (the UI shows a progress indicator during the build);
    at rest a source is Ready if its index exists, else Not connected."""
    if building:
        return "Index building"
    return "Ready" if exists else "Not connected"
