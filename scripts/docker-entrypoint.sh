#!/bin/bash
# Backend container entrypoint: build the FAISS index on first boot
# (downloads the embedding model once, ~90 MB), then run the API.
set -e

INDEX_DIR="${RAG_DEMO_INDEX_DIR:-/data/index}"

if [ ! -d "$INDEX_DIR" ] || [ -z "$(ls -A "$INDEX_DIR" 2>/dev/null)" ]; then
  echo "[entrypoint] No index at $INDEX_DIR — building (first boot)..."
  python -m rag_demo.ingest
else
  echo "[entrypoint] Index already present at $INDEX_DIR — skipping ingest."
fi

exec "$@"
