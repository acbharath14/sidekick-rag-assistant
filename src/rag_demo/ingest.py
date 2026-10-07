"""Build (or rebuild) the FAISS index from a document source."""

from __future__ import annotations

import argparse
import json

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings, get_embeddings, get_settings
from .sources import SOURCES, DocumentSource, MarkdownDirectorySource

# Each non-default source gets its own index so corpora stay pristine.
INDEX_DIRS = {
    "meridian": ".faiss_index",
    "playwright-docs": ".faiss_index_playwright",
    "github-docs": ".faiss_index_github",
    "confluence-mock": ".faiss_index_confluence",
}


def load_documents(settings: Settings):
    """Load documents from the default source (kept for backward compatibility)."""
    return MarkdownDirectorySource(settings.docs_dir).load()


def chunk_documents(docs, settings: Settings):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(docs)


def build_index(
    settings: Settings | None = None,
    source: DocumentSource | None = None,
) -> FAISS:
    settings = settings or get_settings()
    source = source or MarkdownDirectorySource(settings.docs_dir)
    docs = source.load()
    if not docs:
        raise RuntimeError(f"source {source.name!r} produced no documents")
    chunks = chunk_documents(docs, settings)
    store = FAISS.from_documents(chunks, get_embeddings(settings))
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(settings.index_dir))
    # Persist the chunk texts alongside the vector index so the BM25 side of
    # hybrid retrieval can rebuild its corpus without re-reading the source.
    # allowed_groups powers permission-aware retrieval (PR #5).
    (settings.index_dir / "chunks.json").write_text(
        json.dumps(
            [
                {
                    "text": c.page_content,
                    "source": c.metadata.get("source", "?"),
                    "doc_id": c.metadata.get("doc_id", c.metadata.get("source", "?")),
                    "allowed_groups": c.metadata.get("allowed_groups", ["*"]),
                }
                for c in chunks
            ],
            ensure_ascii=False,
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
    else:
        source = MarkdownDirectorySource(settings.docs_dir)

    store = build_index(settings, source)
    print(f"indexed {store.index.ntotal} chunks from {source.name!r} -> {settings.index_dir}")


if __name__ == "__main__":
    main()
