#!/usr/bin/env python3
"""Idempotent bootstrap for a fresh Sidekick clone (Phase 4, U1+U2).

Usage:  python scripts/setup.py [--skip-test-question]

Steps, each skipped when already done (so re-running is safe):
  1. Dependencies importable (else: `pip install -r requirements.txt`).
  2. Ollama installed (install if absent, see install_command) and
     running (start `ollama serve` if the server is down).
  3. Default model `qwen3:8b` pulled (skip if already present).
  4. Demo indexes built — Meridian + Confluence mock (skip each index
     that already exists).
  5. Verify with one test question against the Meridian demo corpus
     (skip with --skip-test-question; takes ~1 minute on CPU).

Honest limits (also in the README):
- Model weights are gigabytes and cannot live in the git clone. The
  repo stays light; this script fetches the heavy parts (model pull,
  embedding model download on first ingest).
- The macOS/Windows install branches below are written carefully but
  have NOT been verified on real macOS/Windows machines. If an
  automatic install fails, the script prints the manual steps and
  exits non-zero instead of pretending success.
- Docker Compose is an optional self-contained alternative, not the
  default path; this script is the default.

Stdlib only. Safe to re-run: every step checks current state first.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "qwen3:8b"
OLLAMA_URL = "http://localhost:11434"
DEMO_SOURCES = ("meridian", "confluence-mock")
TEST_QUESTION = "How long are API tokens valid, and what happens when one expires?"

sys.path.insert(0, str(REPO_ROOT / "src"))


def install_command(platform: str, has_brew: bool, has_winget: bool) -> list[str] | None:
    """Install command for Ollama on this platform, or None when no
    automatic install is available and manual steps are needed.

    Pure function (no side effects) so it is unit-testable. Linux uses
    Ollama's official install script; macOS uses Homebrew when present;
    Windows uses winget when present.
    """
    if platform.startswith("linux"):
        return ["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"]
    if platform == "darwin":
        return ["brew", "install", "ollama"] if has_brew else None
    if platform == "win32":
        return ["winget", "install", "Ollama.Ollama"] if has_winget else None
    return None


def manual_install_steps(platform: str) -> str:
    if platform == "darwin":
        return (
            "Install Ollama from https://ollama.com/download (or "
            "`brew install ollama`), then re-run this script."
        )
    if platform == "win32":
        return (
            "Install Ollama from https://ollama.com/download (or "
            "`winget install Ollama.Ollama`), then re-run this script."
        )
    return "Install Ollama from https://ollama.com/download, then re-run this script."


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=capture)


def server_reachable(url: str = OLLAMA_URL, timeout: float = 2) -> bool:
    try:
        urllib.request.urlopen(f"{url}/api/tags", timeout=timeout)
        return True
    except Exception:  # noqa: BLE001 — unreachable is a status, not an error
        return False


def fail(message: str) -> int:
    print(f"\n❌ SETUP FAILED: {message}")
    return 1


def step_deps() -> int:
    print("== 1. Dependencies ==")
    result = run(
        [sys.executable, "-c", "import sys; sys.path.insert(0, 'src'); import rag_demo"],
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        return fail("Python dependencies are not installed. Run `pip install -r requirements.txt` first.")
    print("   Dependencies OK.")
    return 0


def step_ollama() -> int:
    print("== 2. Ollama ==")
    if shutil.which("ollama") is None:
        cmd = install_command(sys.platform, shutil.which("brew") is not None, shutil.which("winget") is not None)
        if cmd is None:
            return fail(manual_install_steps(sys.platform))
        print(f"   Ollama not found — installing via: {' '.join(cmd)}")
        result = run(cmd)
        if result.returncode != 0 or shutil.which("ollama") is None:
            return fail(
                "Automatic Ollama install failed. " + manual_install_steps(sys.platform)
            )
        print("   Ollama installed.")
    else:
        print("   Ollama binary found.")
    if server_reachable():
        print("   Ollama server is running.")
        return 0
    print("   Ollama server is down — starting `ollama serve` in the background...")
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    for _ in range(30):
        if server_reachable():
            print("   Ollama server is running.")
            return 0
        time.sleep(1)
    return fail("Could not start Ollama automatically. Run `ollama serve` in another terminal, then re-run this script.")


def step_model() -> int:
    print(f"== 3. Model ({DEFAULT_MODEL}) ==")
    # Reuse the app's tag parsing so setup and pre-flight agree.
    from rag_demo.preflight import fetch_tags, model_in_tags

    tags = fetch_tags(OLLAMA_URL)
    if tags is not None and model_in_tags(tags, DEFAULT_MODEL):
        print(f"   Model {DEFAULT_MODEL} already present — skipping pull.")
        return 0
    print(f"   Pulling {DEFAULT_MODEL} (several GB — this takes a while)...")
    result = run(["ollama", "pull", DEFAULT_MODEL])
    if result.returncode != 0:
        return fail(f"`ollama pull {DEFAULT_MODEL}` failed. Check your network and re-run this script.")
    print(f"   Model {DEFAULT_MODEL} ready.")
    return 0


def step_demo_indexes() -> int:
    print("== 4. Demo indexes ==")
    from rag_demo.ingest import INDEX_DIRS
    from rag_demo.preflight import index_present

    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))
    for source in DEMO_SOURCES:
        index_dir = REPO_ROOT / INDEX_DIRS[source]
        if index_present(index_dir):
            print(f"   {source}: index already exists — skipping.")
            continue
        print(f"   Building {source} index...")
        result = run(
            [sys.executable, "-m", "rag_demo.ingest", "--source", source],
            cwd=REPO_ROOT,
            env=env,
        )
        if result.returncode != 0 or not index_present(index_dir):
            return fail(
                f"Could not build the {source} index. Try manually: "
                f"`PYTHONPATH=src python -m rag_demo.ingest --source {source}`."
            )
        print(f"   {source}: indexed.")
    return 0


def step_test_question() -> int:
    print("== 5. Test question (takes ~1 minute on CPU) ==")
    snippet = (
        "import sys; sys.path.insert(0, 'src')\n"
        "from rag_demo.chain import ask\n"
        "from rag_demo.config import get_settings\n"
        f"ans = ask({TEST_QUESTION!r}, get_settings())\n"
        "print(ans.text)\n"
        "print('Sources:', ', '.join(ans.sources))\n"
    )
    result = run([sys.executable, "-c", snippet], cwd=REPO_ROOT, capture=True)
    if result.returncode != 0 or not (result.stdout or "").strip():
        return fail("The test question failed. Check that Ollama is running and the demo index exists.")
    print("   Test question answered:")
    for line in result.stdout.strip().splitlines():
        print(f"   | {line}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--skip-test-question",
        action="store_true",
        help="Skip the final test-question verification step.",
    )
    args = parser.parse_args(argv)

    print("Sidekick setup — idempotent bootstrap for a fresh clone.")
    print("Note: model weights are gigabytes and are fetched, not cloned;")
    print("the repo itself stays light.\n")
    for step in (step_deps, step_ollama, step_model, step_demo_indexes):
        code = step()
        if code != 0:
            return code
    if not args.skip_test_question:
        code = step_test_question()
        if code != 0:
            return code
    print("\n✅ SETUP COMPLETE. Start the app with: PYTHONPATH=src streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
