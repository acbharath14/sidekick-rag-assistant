#!/bin/bash
# One-shot launcher for the Sidekick React GUI + FastAPI backend.
# Kills stale servers, starts the backend in the background (log: /tmp/sidekick-api.log),
# then starts the Vite dev server in the foreground.
#
# Usage: ./scripts/dev.sh
#   Backend:  http://127.0.0.1:8000
#   Frontend: http://localhost:5173
set -u

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Stopping stale servers..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:5173 | xargs kill -9 2>/dev/null || true
sleep 1

echo "==> Starting backend (log: /tmp/sidekick-api.log)..."
if [ ! -d ".venv" ]; then
  echo "ERROR: .venv not found in $REPO_ROOT"
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate
PYTHONPATH=src nohup python -m rag_demo.api > /tmp/sidekick-api.log 2>&1 &
echo "    backend PID $!"

echo "==> Waiting for backend..."
for _ in $(seq 1 20); do
  if curl -sf http://127.0.0.1:8000/api/health > /dev/null 2>&1; then
    echo "    backend up."
    break
  fi
  sleep 1
done

echo "==> Starting frontend..."
cd web
exec npm run dev
