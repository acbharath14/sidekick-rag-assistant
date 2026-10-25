"""Build (or rebuild) the FAISS index from a document source."""

from __future__ import annotations

import argparse
import hashlib
import json

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import INDEX_FORMAT_VERSION, Settings, get_embeddings, get_settings
from .sources import SOURCES, DocumentSource, MarkdownDirectorySource

# Each non-default source gets its own index so corpora stay pristine.
INDEX_DIRS = {
    "meridian": ".faiss_index",
    "playwright-docs": ".faiss_index_playwright",
    "github-docs": ".faiss_index_github",
    "confluence-mock": ".faiss_index_confluence",
    "confluence": ".faiss_index_confluence_real",
    "jira": ".faiss_index_jira",
    "outlook": ".faiss_index_outlook",
    "aws-s3": ".faiss_index_s3",
}


def _doc_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _doc_id(doc) -> str:
    return doc.metadata.get("doc_id", doc.metadata.get("source", "?"))


def load_documents(settings: Settings):
    """Load documents from the default source (kept for backward compatibility)."""
    return MarkdownDirectorySource(settings.docs_dir).load()


def _extract_headings(text: str) -> list[tuple[int, int, str]]:
    """Markdown headings as (char_offset, level, title), in doc order."""
    import re

    headings = []
    offset = 0
    for line in text.splitlines(keepends=True):
        m = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if m:
            headings.append((offset, len(m.group(1)), m.group(2).strip()))
        offset += len(line)
    return headings


def _section_at(headings: list[tuple[int, int, str]], offset: int) -> str:
    """Heading hierarchy in effect at a char offset, 'A > B' style."""
    path: list[str] = []
    for h_offset, level, title in headings:
        if h_offset > offset:
            break
        while len(path) >= level:
            path.pop()
        path.append(title)
    return " > ".join(path)


def _slugify(text: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _chunk_section(headings, doc_text: str, chunk_text: str, cursor: int) -> tuple[str, int]:
    """Section path for a chunk + the cursor advanced past its location.

    Chunks are slices of the doc, so the chunk's opening text locates it;
    overlap means the next chunk may start before the previous one ended,
    so the cursor advances by one past the match, not past the chunk.
    When the text cannot be located (splitter normalization), the last
    known section is kept rather than guessed.
    """
    probe = chunk_text[:120] or chunk_text
    pos = doc_text.find(probe, cursor) if probe else -1
    if pos == -1 and len(chunk_text) > 20:
        pos = doc_text.find(chunk_text[:20], cursor)
    if pos == -1:
        return "", cursor
    return _section_at(headings, pos), pos + 1


def chunk_documents(docs, settings: Settings):
    """Split docs into chunks, stamping each with passage identity
    (Phase 5, A2+A3): chunk_id = document id + section + ordinal, and
    the section path from headings. Chunk texts and boundaries are
    unchanged — only metadata is added, so existing chunk-count
    expectations still hold. Docs without headings get section "" and
    chunk_id 'doc_id:ordinal'."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    out = []
    for doc in docs:
        doc_id = _doc_id(doc)
        headings = _extract_headings(doc.page_content)
        cursor = 0
        section = ""
        for ordinal, chunk in enumerate(splitter.split_documents([doc])):
            found, cursor = _chunk_section(
                headings, doc.page_content, chunk.page_content, cursor
            )
            if found:
                section = found
            chunk.metadata["section"] = section
            slug = _slugify(section)
            chunk.metadata["chunk_id"] = (
                f"{doc_id}#{slug}:{ordinal}" if slug else f"{doc_id}:{ordinal}"
            )
            out.append(chunk)
    return out


def build_index(
    settings: Settings | None = None,
    source: DocumentSource | None = None,
    incremental: bool = False,
) -> FAISS | None:
    """Build the index. With incremental=True, compare document hashes against
    the manifest and skip the rebuild entirely when nothing changed — the
    common case for a nightly refresh job."""
    settings = settings or get_settings()
    source = source or MarkdownDirectorySource(settings.docs_dir)
    docs = source.load()
    if not docs:
        raise RuntimeError(f"source {source.name!r} produced no documents")
    manifest_path = settings.index_dir / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {"documents": {}}
    )
    current = {_doc_id(d): _doc_sha(d.page_content) for d in docs}
    # Phase 5 migration: an incremental skip must not preserve a stale
    # format — an index without the current format marker is rebuilt even
    # when document hashes match, so old indexes can never linger.
    format_path = settings.index_dir / "index_format.json"
    format_current = False
    if format_path.exists():
        try:
            format_current = (
                json.loads(format_path.read_text(encoding="utf-8")).get("version")
                == INDEX_FORMAT_VERSION
            )
        except (OSError, ValueError):
            format_current = False
    if incremental and manifest.get("documents") and format_current:
        previous = {k: v["sha256"] for k, v in manifest["documents"].items()}
        if current == previous:
            print(
                f"index up to date ({len(current)} docs from {source.name!r}) — nothing to do"
            )
            return None
    chunks = chunk_documents(docs, settings)
    store = FAISS.from_documents(chunks, get_embeddings(settings))
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(settings.index_dir))
    # Persist the chunk texts alongside the vector index so the BM25 side of
    # hybrid retrieval can rebuild its corpus without re-reading the source.
    # allowed_groups powers permission-aware retrieval (PR #5).
    # chunk_id + section power Phase 5 passage-level citations.
    (settings.index_dir / "chunks.json").write_text(
        json.dumps(
            [
                {
                    "text": c.page_content,
                    "source": c.metadata.get("source", "?"),
                    "doc_id": c.metadata.get("doc_id", c.metadata.get("source", "?")),
                    "allowed_groups": c.metadata.get("allowed_groups", ["*"]),
                    "chunk_id": c.metadata.get("chunk_id", ""),
                    "section": c.metadata.get("section", ""),
                }
                for c in chunks
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    # Phase 5 format marker: loaders treat a missing or older marker as
    # a stale index (citations degrade to filename-only + rebuild banner).
    format_path.write_text(
        json.dumps(
            {
                "version": INDEX_FORMAT_VERSION,
                "features": ["chunk_id", "section"],
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "documents": {
                    doc_id: {"sha256": sha}
                    for doc_id, sha in current.items()
                }
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    return store


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the FAISS index.")
    parser.add_argument(
        "--source",
        choices=sorted(SOURCES),
        default="meridian",
        help="document source to index (default: meridian)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="with --source playwright-docs, filter to a language (e.g. python)",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="with --source github-docs, the repo as owner/name "
        "(or set RAG_DEMO_GITHUB_REPO)",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="skip the rebuild when no documents changed (nightly refresh)",
    )
    args = parser.parse_args()

    settings = get_settings()
    # A real corpus deserves its own index so the default stays pristine.
    settings.index_dir = (
        settings.index_dir.parent / INDEX_DIRS.get(args.source, ".faiss_index")
    )
    if args.source == "playwright-docs":
        from .sources import PlaywrightDocsSource

        source: DocumentSource = PlaywrightDocsSource(language=args.language)
    elif args.source == "github-docs":
        from .sources import GitHubDocsSource

        repo = args.repo or settings.github_repo
        if not repo:
            parser.error("--source github-docs needs --repo owner/name or RAG_DEMO_GITHUB_REPO")
        source = GitHubDocsSource(repo=repo, token=settings.github_token)
    elif args.source == "confluence-mock":
        from .sources import MockConfluenceSource

        source = MockConfluenceSource()
    elif args.source in ("confluence", "jira", "outlook", "aws-s3"):
        from .enterprise import ENTERPRISE_SOURCES

        source_cls = ENTERPRISE_SOURCES[args.source]
        source = source_cls()
    else:
        source = MarkdownDirectorySource(settings.docs_dir)

    store = build_index(settings, source, incremental=args.incremental)
    if store is None:
        return
    print(f"indexed {store.index.ntotal} chunks from {source.name!r} -> {settings.index_dir}")


if __name__ == "__main__":
    main()
