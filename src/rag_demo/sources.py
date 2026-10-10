"""Document sources: where the indexed corpus comes from.

A DocumentSource is anything that can produce LangChain Documents with a
`source` metadata key (used for citations). The default is a local directory
of markdown files (the fictional Meridian Logistics corpus). Additional
sources — real documentation, enterprise connectors — implement the same
interface, so swapping corpora never touches the indexing code.

Enterprise connectors (Confluence, SharePoint, ...) follow this interface;
see ARCHITECTURE.md for the pattern. Permission filtering happens at the
retriever (metadata `allowed_groups`), never in the LLM prompt.
"""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
from abc import ABC, abstractmethod
from html.parser import HTMLParser
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
    "github-docs": None,  # filled below (defined after the registry for readability)
    "confluence-mock": None,
}


class _TextExtractor(HTMLParser):
    """Strip tags; block elements become paragraph breaks."""

    BLOCKS = {
        "h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "div", "br",
        "tr", "table", "ul", "ol",
    }

    def __init__(self):
        super().__init__()
        self.chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs):
        if tag in self.BLOCKS:
            self.chunks.append("\n\n")

    def handle_data(self, data: str):
        text = " ".join(data.split())
        if text:
            self.chunks.append(text + " ")

    def text(self) -> str:
        import re

        paras = [re.sub(r"\s+", " ", p).strip() for p in "".join(self.chunks).split("\n\n")]
        return "\n\n".join(p for p in paras if p)


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    return parser.text()


class GitHubDocsSource(DocumentSource):
    """Index markdown docs from any GitHub repo via the REST API — no clone.

    Reads the git tree recursively, fetches each `.md` blob, and cites like
    [github:owner/repo/docs/guide.md]. Auth is optional (public repos work
    anonymously); pass a token for private repos or higher rate limits.
    The token always comes from RAG_DEMO_GITHUB_TOKEN — never committed.
    """

    name = "github-docs"

    def __init__(
        self,
        repo: str,
        branch: str = "main",
        token: str | None = None,
        path_prefix: str = "",
    ):
        self.repo = repo
        self.branch = branch
        self.token = token
        self.path_prefix = path_prefix

    def _api(self, path: str):
        import requests

        headers = {"Accept": "application/vnd.github+json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        resp = requests.get(f"https://api.github.com{path}", headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def load(self) -> list[Document]:
        tree = self._api(f"/repos/{self.repo}/git/trees/{self.branch}?recursive=1")
        if tree.get("truncated"):
            raise RuntimeError(
                f"git tree for {self.repo}@{self.branch} was truncated — "
                "narrow path_prefix or use a smaller repo"
            )
        docs = []
        for node in tree.get("tree", []):
            if node.get("type") != "blob":
                continue
            path = node.get("path", "")
            if not path.endswith(".md") or not path.startswith(self.path_prefix):
                continue
            blob = self._api(f"/repos/{self.repo}/git/blobs/{node['sha']}")
            content = base64.b64decode(blob["content"]).decode("utf-8")
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": f"github:{self.repo}/{path}",
                        "doc_id": f"github:{self.repo}/{node['sha']}",
                        "sha": node["sha"],
                        "allowed_groups": ["*"],
                    },
                )
            )
        return docs


class MockConfluenceSource(DocumentSource):
    """Simulated Confluence Cloud source from JSON fixtures.

    The fixtures mirror the Confluence REST shape (results[].id/title/
    body.storage.value/space/restrictions/version), including pagination via
    `_links.next`. Swap the fixtures for real API calls and this becomes a
    real connector — the downstream code doesn't change.
    """

    name = "confluence-mock"

    def __init__(self, fixtures_dir: Path | str | None = None):
        self.fixtures_dir = (
            Path(fixtures_dir)
            if fixtures_dir
            else Path(__file__).resolve().parent.parent.parent
            / "fixtures"
            / "confluence"
        )

    def load(self) -> list[Document]:
        docs = []
        for fixture in sorted(self.fixtures_dir.glob("pages-*.json")):
            data = json.loads(fixture.read_text(encoding="utf-8"))
            for page in data.get("results", []):
                html = page.get("body", {}).get("storage", {}).get("value", "")
                read_groups = page.get("restrictions", {}).get("read", [])
                docs.append(
                    Document(
                        page_content=html_to_text(html),
                        metadata={
                            "source": f"confluence:{page['space']['key']}/{page['title']}",
                            "doc_id": f"confluence:{page['id']}",
                            "allowed_groups": read_groups or ["*"],
                            "last_modified": page.get("version", {}).get("when", ""),
                        },
                    )
                )
        return docs


SOURCES["github-docs"] = GitHubDocsSource
SOURCES["confluence-mock"] = MockConfluenceSource

# Enterprise connectors (Jira, Confluence, Outlook, AWS S3). These are
# optional — they require credentials via env vars and their dependencies
# (boto3 for AWS) may not be installed. Registered lazily so a missing
# dependency doesn't break the default sources.
try:
    from .enterprise import ENTERPRISE_SOURCES

    SOURCES.update(ENTERPRISE_SOURCES)
except ImportError:
    pass
