# STATUS — sidekick-rag-assistant
_Last updated: 2026-10-08_

## Where things stand
- All 9 PRs merged to main, CI green, zero open issues — code-complete.
  https://github.com/acbharath14/sidekick-rag-assistant
- Retrieval: hybrid BM25+dense with rank fusion, opt-in rerank, query rewriting.
- Enterprise: GitHub connector, mock-Confluence fixtures, ACL filtering, audit logging.
- UI: dark Streamlit, streaming, citations, file upload (PDF/DOCX/TXT/MD/CSV), Tesseract OCR fallback for scanned PDFs.
- Hybrid "ask anything" mode: corpus misses fall back to general knowledge, always marked.
- Regression suite: 9 cases (`PYTHONPATH=src python -m evals.regression`).
- Stale feature branches deleted via web UI after zero-diff verification.
- iMac environment: Fusion Drive at 1.8TB free (was ~106GB); Ollama models on NVMe SSD, server confirmed on new path; open-webui venv created (cryptography wheel build issue pending).

## Next actions (ordered)
1. iMac: conda-install tesseract + poppler, retest scanned-PDF upload.
2. Real-model retest: hybrid Paris answer, follow-up rewrite, 9-case regression.
3. UI enhancement pass (current Streamlit UI underwhelming — needs spec).
4. Confirm latest main is green in GitHub Actions.

## Blocked on
- (none)

## Decisions made
- 2026-10-07: Hybrid mode default-on (RAG_DEMO_HYBRID=1); strict corpus-only via sidebar toggle or env.
- 2026-10-07: GitHub workflow files go via web UI (App lacks `workflows` permission).
- 2026-10-07: Enterprise connectors take credentials from env vars only; repo never holds secrets.
- 2026-10-08: sidekick is the dev-harness pilot project (STATUS.md adopted here first).

## Don't
- Don't push .github/workflows/* via the API — silently rejected.
- Don't claim CI/Pages status in README until verified live.
