#!/bin/bash
# Prerequisite checks for the Sidekick FastAPI backend.
# Verifies Ollama is reachable and warns if OLLAMA_KEEP_ALIVE is not set
# (without it, Ollama unloads the model after 5 min idle, adding full
# reload latency to every prompt on CPU machines).
set -u

OLLAMA_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
FAIL=0

echo "Checking Ollama at $OLLAMA_URL ..."
if ! curl -sf "$OLLAMA_URL/api/tags" > /dev/null 2>&1; then
    echo "FAIL: Ollama is not reachable at $OLLAMA_URL"
    echo "  Start it first: open the Ollama app, or run 'ollama serve'."
    FAIL=1
else
    echo "OK: Ollama reachable."
fi

echo "Checking OLLAMA_KEEP_ALIVE ..."
# launchctl exists on macOS only. Under Git Bash on Windows there is no
# launchctl, so print the Windows fix (setx) instead of the macOS one.
# On any other system without launchctl, fall back to the plain env var.
IS_WINDOWS=0
case "${OSTYPE:-}" in
    msys*|cygwin*) IS_WINDOWS=1 ;;
esac
if [ -n "${WINDIR:-}" ]; then
    IS_WINDOWS=1
fi
if [ "$IS_WINDOWS" -eq 1 ]; then
    KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-}"
    KEEP_ALIVE_FIX="setx OLLAMA_KEEP_ALIVE 24h"
    KEEP_ALIVE_AFTER="Then quit and restart the Ollama app (or the Ollama service)."
elif command -v launchctl >/dev/null 2>&1; then
    KEEP_ALIVE="$(launchctl getenv OLLAMA_KEEP_ALIVE 2>/dev/null || true)"
    KEEP_ALIVE_FIX="launchctl setenv OLLAMA_KEEP_ALIVE 24h"
    KEEP_ALIVE_AFTER="Then quit and restart the Ollama app."
else
    KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-}"
    KEEP_ALIVE_FIX="export OLLAMA_KEEP_ALIVE=24h  (or set it in your shell profile)"
    KEEP_ALIVE_AFTER="Then quit and restart Ollama."
fi
if [ -z "$KEEP_ALIVE" ]; then
    echo "WARN: OLLAMA_KEEP_ALIVE is not set."
    echo "  Without it, Ollama unloads the model after 5 min idle,"
    echo "  and every new prompt pays a full model reload on CPU."
    echo "  Fix: $KEEP_ALIVE_FIX"
    echo "  $KEEP_ALIVE_AFTER"
else
    echo "OK: OLLAMA_KEEP_ALIVE=$KEEP_ALIVE"
fi

echo "Checking Langfuse tracing (optional) ..."
if [ -n "${LANGFUSE_PUBLIC_KEY:-}" ] && [ -n "${LANGFUSE_SECRET_KEY:-}" ]; then
    LANGFUSE_URL="${LANGFUSE_BASE_URL:-${LANGFUSE_HOST:-http://127.0.0.1:3000}}"
    if curl -sfL "$LANGFUSE_URL/" > /dev/null 2>&1; then
        echo "OK: Langfuse reachable at $LANGFUSE_URL (tracing on)."
    else
        echo "WARN: Langfuse keys are set but the server is not reachable at $LANGFUSE_URL."
        echo "  Tracing stays off until it is up; the app still works."
        echo "  Start it with: docker compose -f docker-compose.langfuse.yml up -d"
    fi
else
    echo "SKIP: Langfuse tracing is off (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set)."
fi

if [ "$FAIL" -ne 0 ]; then
    exit 1
fi
echo "Prerequisites OK."
