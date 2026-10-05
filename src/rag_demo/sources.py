"""Document sources: where the indexed corpus comes from.

A DocumentSource is anything that can produce LangChain Documents with a
`source` metadata key (used for citations). The default is a local directory
of markdown files (the fictional Meridian Logistics corpus). Additional
sources — real documentation, enterprise connectors — implement the same
interface, so swapping corpora never touches the indexing code.

Enterprise connectors (Confluence, SharePoint, Outlook, ...) follow this
interface too; see ARCHITECTURE.md for the pattern and why permission
filtering belongs at the source, not the retriever.
"""

from __future__ import annotations

import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document


class DocumentSource(ABC):
    """Produces documents for indexing."""

    name: str = "unnamed"

    @abstractmethod
    def load(self) -> list[Document]:
        """Return documents, each carrying a `source` metadata key."""
        ...


class MarkdownDirectorySource(DocumentSource):
    """Load `**/*.md` from a local directory. The default corpus lives here."""

    name = "meridian"

    def __init__(self, docs_dir: Path | str):
        self.docs_dir = Path(docs_dir)

    def load(self) -> list[Document]:
        loader = DirectoryLoader(
            str(self.docs_dir),
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=False,
        )
        docs = loader.load()
        # Keep just the filename for citations.
        for d in docs:
            d.metadata["source"] = d.metadata.get("source", "").split("/")[-1]
        return docs


class PlaywrightDocsSource(DocumentSource):
    """Fetch real Playwright documentation at a pinned version.

    Sparse-checkouts only `docs/src/*.md` from microsoft/playwright at the
    given tag — no full clone. Opt-in: needs git and network. The checkout is
    cached, so rebuilds don't re-download.

    Useful to demo retrieval on a real, messy corpus. The golden eval set in
    evals/ targets the default corpus; quality on this corpus is eyeballed,
    not asserted, because upstream docs drift over time.
    """

    name = "playwright-docs"
    REPO = "https://github.com/microsoft/playwright.git"

    def __init__(
        self,
        tag: str = "v1.63.0",
        language: str | None = None,
        cache_dir: Path | str | None = None,
    ):
        """
        Args:
            tag: pinned Playwright release tag.
            language: e.g. "python" keeps `*-python.md` plus language-agnostic
                pages; None keeps everything.
            cache_dir: where to keep the sparse checkout (default: a temp dir
                under the project).
        """
        self.tag = tag
        self.language = language
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir
            else Path(__file__).resolve().parent.parent.parent / ".playwright-docs"
        )

    def _ensure_checkout(self) -> Path:
        docs_src = self.cache_dir / "docs" / "src"
        if docs_src.exists() and any(docs_src.glob("*.md")):
            return docs_src
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git", "clone", "--depth", "1",
                "--filter=blob:none", "--sparse",
                "--branch", self.tag,
                self.REPO, str(self.cache_dir),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "sparse-checkout", "set", "docs/src"],
            cwd=str(self.cache_dir),
            check=True,
            capture_output=True,
            text=True,
        )
        return docs_src

    def load(self) -> list[Document]:
        docs_src = self._ensure_checkout()
        loader = DirectoryLoader(
            str(docs_src),
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=False,
        )
        docs = loader.load()
        if self.language:
            suffix = f"-{self.language}.md"
            docs = [
                d for d in docs
                if d.metadata.get("source", "").endswith(suffix)
                or not any(
                    d.metadata.get("source", "").endswith(f"-{lang}.md")
                    for lang in ("python", "java", "js", "csharp")
                )
            ]
        for d in docs:
            # Cite like [playwright-docs/auth.md], not a local temp path.
            filename = d.metadata.get("source", "").split("/")[-1]
            d.metadata["source"] = f"playwright-docs/{filename}"
        return docs


SOURCES: dict[str, type[DocumentSource]] = {
    MarkdownDirectorySource.name: MarkdownDirectorySource,
    PlaywrightDocsSource.name: PlaywrightDocsSource,
}
