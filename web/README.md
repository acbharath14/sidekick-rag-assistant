# Sidekick Web UI

React + TypeScript + Vite chat frontend for the Sidekick RAG assistant.
Talks to the FastAPI backend (`src/rag_demo/api.py`). This is v1: the chat
experience only — no enterprise connectors, regression, or eval pages.

## Prerequisites

- Node 20+ and npm (the preflight script checks)
- The FastAPI backend running at `http://127.0.0.1:8000`

Start the backend from the repo root:

```
PYTHONPATH=src python -m rag_demo.api
```

## Run

```
cd web
chmod +x preflight.sh
./preflight.sh
npm run dev
```

Then open http://127.0.0.1:5173 in a browser.

To point at a different backend, set the environment variable before
starting Vite:

```
VITE_API_URL=http://192.168.1.10:8000 npm run dev
```

## Build

```
npm run build
npm run preview
```

`npm run build` typechecks (`tsc --noEmit`) and emits a static bundle to
`dist/`. `npm run preview` serves it on port 4173.

## What it does

- **ContextBanner** (top, always visible): shows the model, retrieval mode,
  and embedding model from `GET /api/config`, so it is always clear which
  context answers come from.
- **HealthIndicator**: `GET /api/health` — Ollama reachability and index
  presence, with a retry button.
- **Chat**: type a question, Enter sends. Answers stream token-by-token via
  `POST /api/chat/stream` (SSE over POST, read with `fetch` +
  `ReadableStream` since `EventSource` only supports GET).
- **ChatMessage**: markdown rendering, citation pills from `sources[]`,
  latency caption, and a "General knowledge" badge when `grounded` is
  false. A note shows the rewritten standalone question when the backend
  rewrote a follow-up.
- **PassageViewer**: expandable list of retrieved chunks with scores.
- **FeedbackButtons**: thumbs up/down per answer, persisted via
  `POST /api/feedback` to the same `feedback.jsonl` the Streamlit app
  writes.

## Styling

Plain CSS in `src/styles.css`, no component library. Dark mode follows
`prefers-color-scheme` automatically.

## Preflight

`preflight.sh` verifies node and npm exist, runs `npm install` when
`node_modules` is missing, and prints fix commands for anything it cannot
resolve. It is idempotent and safe to run on every checkout. Note: files
pushed through the GitHub API land as mode 100644, so the first clone
needs `chmod +x preflight.sh`.
