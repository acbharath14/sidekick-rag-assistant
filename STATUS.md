# STATUS — sidekick-rag-assistant
_Last updated: 2026-10-25 EDT_

## Where things stand
- Main @0012801. PRs #36–#46 merged: production-readiness phases 1–5, pipeline walkthrough, README trim, setup-prereqs (#44), per-source regression (#45), Mermaid diagrams (#46). No open PRs.
  https://github.com/acbharath14/sidekick-rag-assistant
- Retrieval: hybrid BM25+dense with RRF fusion by default, top_k 8, rerank on (fake-aware: stays off under FAKE; explicit env wins). Relevance floor 0.5 (fake-aware, never on RRF scores). Chunk ids + section paths; citations as [source › section]; weak-evidence notice + abstain-friendly prompt when nothing passes the floor.
- Answering: classic + agentic paths. Agentic loop: parallel search dispatch, single planning, streamed synthesis, iteration/wall-clock budgets with evidence + uncertainty on exhaustion.
- Caching: settings-keyed module dicts (not @st.cache_resource); one active retriever across paths.
- UI: sidebar Chat / Sources / Settings tabs. Demo mode structurally separate (Meridian + Confluence mock only, marked fictional everywhere); Workspace picker grouped Documentation + Your workspace; default entry is Workspace if any workspace source has a built index, else Demo. Pre-flight (Ollama/model/index) disables input with a fix-it banner when something is missing. Build demo index button in the UI. Pipeline walkthrough (chunking, embedding, retrieval — no generation step) via the sidebar "How it works" button.
- Setup script (`scripts/setup.py`, stdlib-only, idempotent): per-source prerequisite checks before any build, 8-line readiness report; default run builds demo indexes only; `--all` builds every ready source, `--source NAME` fails fast with the exact fix, `--check-only` reports without building, `--repo` for github-docs. Unconfigured enterprise sources are skipped with reason, never errors.
- Regression: per-source suites (`evals/per_source.py`) reporting accuracy / retrieval / rerank (before-vs-after) / precision (chunk+section) per source and overall. Prompt counts: Meridian 44, Playwright docs 12, GitHub docs 8, Confluence mock 7, enterprise 0 (skipped "not connected" — no mock enterprise data, by decision). Full suite 143 passed, fake tier only, at #46.
- Diagrams: Mermaid flowcharts in chat answers, explicit trigger only (ask naming a diagram, or the "Show as diagram" button); rendered via mermaid.js v10 from CDN with raw code always shown; invalid syntax or weak evidence shows code, not a broken render. Repair retry deferred.
- Enterprise: GitHub connector, mock-Confluence fixtures, ACL filtering (default-deny), audit logging. Credentials from env vars only; repo never holds secrets.
- iMac environment: 1.0TB NVMe Sequoia external boot (internal drive kept as fallback); Ollama models on the NVMe; qwen3:8b daily driver; Open WebUI at localhost:8080.

## Next actions (ordered)
1. iMac: pull main @0012801; Mermaid visual proof in Streamlit + qwen3:8b Mermaid validity rate on real flows.
2. iMac: Build demo index (committed Confluence-mock index is stale — old ENG fixtures, not current OPS); GitHub-docs ingest with `--repo acbharath14/sidekick-rag-assistant`.
3. iMac: first real per-source runs (retrieval-only, then `--answers`); Playwright prompts vs the real 7,009-chunk index.
4. Final acceptance: 3 golden questions + `eval_compare` + LanceDB restart check + citation-precision and weak-query spot checks.
5. Build (on his word): streaming perception; then multi-source search. Mermaid repair retry if the validity rate proves poor.

## Blocked on
- (none — remaining items are his iMac runs or his go-ahead)

## Decisions made
- 2026-10-07: Hybrid mode default-on (RAG_DEMO_HYBRID=1); strict corpus-only via sidebar toggle or env.
- 2026-10-07: GitHub workflow files go via web UI (App lacks `workflows` permission).
- 2026-10-07: Enterprise connectors take credentials from env vars only; repo never holds secrets.
- 2026-10-08: sidekick is the dev-harness pilot project (STATUS.md adopted here first).
- 2026-10-24: Production-readiness plan approved (5 phases: caching, agentic loop, default pipeline, UI split, grounding) + walkthrough follow-up; no-think mode NOT approved.
- 2026-10-25: Enterprise regression skips when unconnected — no mock enterprise data; Demo/Workspace separation is structural, fiction stays in Demo.
- 2026-10-25: Mermaid diagrams explicit-trigger only; raw code always shown; no auto-detection in v1.

## Don't
- Don't push .github/workflows/* via the API — silently rejected.
- Don't claim CI/Pages status in README until verified live.
- Don't claim Streamlit runtime proof from sandbox runs — sandbox proves fake-tier logic only; visual/real-model proof is an iMac job.
- Don't delete merged branches via the API — the GitHub App exposes no delete tool; delete via the GitHub web UI (Branches page).
