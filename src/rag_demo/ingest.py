"""Build (or rebuild) the FAISS index from a document source."""

from __future__ import annotations

import argparse

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings, get_embeddings, get_settings
from .sources import DocumentSource, MarkdownDirectorySource, PlaywrightDocsSource


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
    return store


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the FAISS index.")
    parser.add_argument(
        "--source",
        choices=["meridian", "playwright-docs"],
        default="meridian",
        help="document source to index (default: meridian)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="with --source playwright-docs, filter to a language (e.g. python)",
    )
    args = parser.parse_args()

    settings = get_settings()
    if args.source == "playwright-docs":
        # A real corpus deserves its own index so the default stays pristine.
        settings.index_dir = settings.index_dir.parent / ".faiss_index_playwright"
        source: DocumentSource = PlaywrightDocsSource(language=args.language)
    else:
        source = MarkdownDirectorySource(settings.docs_dir)

    store = build_index(settings, source)
    print(f"indexed {store.index.ntotal} chunks from {source.name!r} -> {settings.index_dir}")


if __name__ == "__main__":
    main()
