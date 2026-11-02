#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

fail() {
  echo "preflight: $1" >&2
  exit 1
}

command -v node >/dev/null 2>&1 || fail "node not found. Install Node 20+: https://nodejs.org/en/download"
command -v npm >/dev/null 2>&1 || fail "npm not found. It ships with Node: https://nodejs.org/en/download"

node_version="$(node --version)"
echo "preflight: node ${node_version}, npm $(npm --version)"

if [ ! -d "node_modules" ]; then
  echo "preflight: node_modules missing, running npm install (uses package.json lockfile)..."
  npm install || fail "npm install failed. Check network access to the npm registry, then retry."
else
  echo "preflight: node_modules present, skipping install."
fi

echo "preflight: OK. Run the dev server with: npm run dev"
echo "preflight: The API backend must be running at http://127.0.0.1:8000 (override with VITE_API_URL)."
