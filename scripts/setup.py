#!/usr/bin/env python3
"""Idempotent bootstrap for a fresh Sidekick clone (Phase 4, U1+U2).

Usage:  python scripts/setup.py [--skip-test-question] [--all]
        [--source NAME] [--repo owner/name] [--check-only]

Steps, each skipped when already done (so re-running is safe):
  1. Dependencies importable (else: `pip install -r requirements.txt`).
  2. Ollama installed (install if absent, see install_command) and
     running (start `ollama serve` if the server is down).
  3. Default model `qwen3:8b` pulled (skip if already present).
  4. Prerequisite check: a per-source readiness report for all 8
     sources is printed BEFORE any index build, so a missing repo,
     credential, or package is reported with its fix instead of
     failing mid-run with a traceback. Unconfigured enterprise
     sources are skipped with a reason, never errors.
  5. Index build (default: demo indexes — Meridian + Confluence
     mock; --all: every ready source; --source NAME: one source,
     failing fast with the exact fix when its prerequisites are
     missing). Indexes that already exist with the current format
     are skipped; stale ones are rebuilt.
  6. Verify with one test question against the Meridian demo corpus
     (skip with --skip-test-question; takes ~1 minute on CPU; skipped
     automatically when this run did not build Meridian).

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
import importlib.util
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "qwen3:8b"
OLLAMA_URL = "http://localhost:11434"
DEMO_SOURCES = ("meridian", "confluence-mock")
TEST_QUESTION = "How long are API tokens valid, and what happens when one expires?"

# All 8 ingest sources, in the order the README lists them for rebuilds.
SOURCE_ORDER = (
    "meridian",
    "confluence-mock",
    "playwright-docs",
    "github-docs",
    "confluence",
    "jira",
    "outlook",
    "aws-s3",
)

# Required env vars per enterprise source (verified against
# rag_demo/enterprise.py, which raises on the first missing one).
REQUIRED_ENV = {
    "confluence": ("RAG_DEMO_CONFLUENCE_URL", "RAG_DEMO_CONFLUENCE_TOKEN"),
    "jira": ("RAG_DEMO_JIRA_URL", "RAG_DEMO_JIRA_TOKEN"),
    "outlook": ("RAG_DEMO_OUTLOOK_TOKEN",),
    "aws-s3": ("RAG_DEMO_AWS_S3_BUCKET",),
}


@dataclass
class SourceReadiness:
    """Prerequisite state for one source.

    status is "ready" (ingest can run), "skipped" (an optional source
    whose prerequisites are absent — reported, never an error), or
    "blocked" (intent is signaled, e.g. an S3 bucket is set, but a
    dependency is missing — setup tries to fix it, then fails fast
    with the fix if it cannot). `reason` states the missing piece and
    its fix, and is empty when ready.
    """

    source: str
    status: str
    reason: str = ""


def _boto3_present() -> bool:
    return importlib.util.find_spec("boto3") is not None


def check_source_readiness(
    source: str,
    env: dict | None = None,
    repo: str | None = None,
    has_git: bool | None = None,
    boto3_ok: bool | None = None,
) -> SourceReadiness:
    """Prerequisite check for one source. Pure function (no side
    effects) so it is unit-testable: env defaults to os.environ,
    has_git defaults to shutil.which("git"), boto3_ok defaults to a
    real import check — tests inject all three.

    Rules verified against rag_demo/sources.py, enterprise.py, ingest.py:
    meridian / confluence-mock need nothing; playwright-docs shells out
    to git for its sparse checkout; github-docs needs a repo (flag or
    RAG_DEMO_GITHUB_REPO); enterprise sources need their env vars;
    aws-s3 needs boto3 (checked first, matching load() order) plus the
    bucket.
    """
    env = os.environ if env is None else env
    if source in ("meridian", "confluence-mock"):
        return SourceReadiness(source, "ready")
    if source == "playwright-docs":
        git = shutil.which("git") is not None if has_git is None else has_git
        if not git:
            return SourceReadiness(
                source, "skipped",
                "git not found — install git to fetch the Playwright docs",
            )
        return SourceReadiness(source, "ready")
    if source == "github-docs":
        chosen = repo or env.get("RAG_DEMO_GITHUB_REPO", "")
        if not chosen:
            return SourceReadiness(
                source, "skipped",
                "pass --repo owner/name or set RAG_DEMO_GITHUB_REPO",
            )
        return SourceReadiness(source, "ready")
    if source in REQUIRED_ENV:
        missing = [f"{v} not set" for v in REQUIRED_ENV[source] if not env.get(v)]
        if source == "aws-s3":
            boto3 = _boto3_present() if boto3_ok is None else boto3_ok
            if not boto3 and not missing:
                return SourceReadiness(
                    source, "blocked",
                    "boto3 not installed — setup will try `pip install boto3` "
                    "when building aws-s3",
                )
            if not boto3:
                missing.append("boto3 not installed (pip install boto3)")
        if missing:
            return SourceReadiness(source, "skipped", "; ".join(missing))
        return SourceReadiness(source, "ready")
    raise ValueError(f"unknown source: {source!r}")


def check_all_sources(
    repo: str | None = None,
    env: dict | None = None,
    has_git: bool | None = None,
    boto3_ok: bool | None = None,
) -> list[SourceReadiness]:
    """Readiness for all 8 sources, in README rebuild order."""
    return [
        check_source_readiness(
            s, env=env, repo=repo, has_git=has_git, boto3_ok=boto3_ok
        )
        for s in SOURCE_ORDER
    ]


def format_readiness_report(readiness: list[SourceReadiness]) -> str:
    """One line per source: 'name: ready' or 'name: skipped — reason'."""
    lines = []
    for r in readiness:
        if r.status == "ready":
            lines.append(f"   {r.source}: ready")
        else:
            lines.append(f"   {r.source}: {r.status} — {r.reason}")
    return "\n".join(lines)


def index_action(index_dir, present=None, current=None) -> str:
    """'skip' when the index exists with the current format, 'rebuild'
    when it exists but is stale, 'build' when absent. Check callables
    are injectable for tests; defaults are the app's pre-flight checks."""
    from rag_demo.preflight import index_format_current, index_present

    present = index_present if present is None else present
    current = index_format_current if current is None else current
    if not present(Path(index_dir)):
        return "build"
    return "skip" if current(Path(index_dir)) else "rebuild"


def ensure_boto3() -> bool:
    """Install boto3 via pip. Only called when S3 is actually being
    built with a bucket set — never unconditionally."""
    print("   boto3 not installed — installing with `pip install boto3`...")
    result = run([sys.executable, "-m", "pip", "install", "boto3"])
    ok = result.returncode == 0 and _boto3_present()
    print("   boto3 installed." if ok else "   boto3 install failed.")
    return ok


def resolve_build_set(
    *,
    all_flag: bool,
    source: str | None,
    readiness: list[SourceReadiness],
    boto3_installer=None,
) -> tuple[list[str], str | None]:
    """Decide what a run builds. Returns (to_build, error).

    Default: demo sources only. --all: every ready source; blocked or
    skipped sources are left out, never errors. --source: exactly that
    source, or an error naming the missing piece and its fix (fail fast,
    before ingest runs). boto3_installer, when given, is a () -> bool
    called only for an S3 build blocked on boto3 (bucket set, boto3
    missing); it is never called for users with no S3 intent.
    """
    by_source = {r.source: r for r in readiness}
    if source:
        target = by_source[source]
        if target.status == "blocked" and source == "aws-s3" and boto3_installer is not None:
            if boto3_installer():
                return [source], None
            return [], (
                "Cannot build aws-s3: boto3 is not installed and "
                "`pip install boto3` failed. Install it manually, then re-run."
            )
        if target.status != "ready":
            return [], f"Cannot build {source}: {target.reason}."
        return [source], None
    if all_flag:
        to_build = []
        for r in readiness:
            if r.status == "ready":
                to_build.append(r.source)
            elif r.status == "blocked" and r.source == "aws-s3" and boto3_installer is not None:
                if boto3_installer():
                    to_build.append(r.source)
        return to_build, None
    return [s for s in DEMO_SOURCES if by_source[s].status == "ready"], None

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


def step_prereqs(repo: str | None) -> list[SourceReadiness]:
    """Prerequisite-check phase: print the per-source readiness report
    BEFORE any index build. Never fails: unconfigured sources are
    skipped with a reason, not errors."""
    print("== 4. Prerequisites (before any index build) ==")
    readiness = check_all_sources(repo=repo)
    print(format_readiness_report(readiness))
    return readiness


def step_indexes(sources: list[str], repo: str | None) -> int:
    """Build each listed source's index. Existing indexes with the
    current format are skipped; stale ones are rebuilt; missing ones
    are built. A build failure fails fast with the manual command."""
    print("== 5. Index build ==")
    from rag_demo.ingest import INDEX_DIRS
    from rag_demo.preflight import index_present

    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))
    for source in sources:
        index_dir = REPO_ROOT / INDEX_DIRS[source]
        action = index_action(index_dir)
        if action == "skip":
            print(f"   {source}: index already exists and is current — skipping.")
            continue
        print(f"   {'Rebuilding' if action == 'rebuild' else 'Building'} {source} index...")
        cmd = [sys.executable, "-m", "rag_demo.ingest", "--source", source]
        if source == "github-docs" and repo:
            cmd += ["--repo", repo]
        result = run(cmd, cwd=REPO_ROOT, env=env)
        if result.returncode != 0 or not index_present(index_dir):
            return fail(
                f"Could not build the {source} index. Try manually: "
                f"`PYTHONPATH=src python -m rag_demo.ingest --source {source}`."
            )
        print(f"   {source}: indexed.")
    return 0


def step_test_question() -> int:
    print("== 6. Test question (takes ~1 minute on CPU) ==")
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
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Print the per-source readiness report and exit without "
        "building anything or touching Ollama.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--all",
        action="store_true",
        help="Build every ready source (demo + docs + any configured "
        "enterprise source). Unconfigured sources are skipped, not errors.",
    )
    group.add_argument(
        "--source",
        choices=list(SOURCE_ORDER),
        default=None,
        help="Build one source only. Fails fast with the exact fix when "
        "its prerequisites are missing.",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="With --source github-docs (or --all), the repo as "
        "owner/name (or set RAG_DEMO_GITHUB_REPO).",
    )
    args = parser.parse_args(argv)

    print("Sidekick setup — idempotent bootstrap for a fresh clone.")
    print("Note: model weights are gigabytes and are fetched, not cloned;")
    print("the repo itself stays light.\n")

    if args.check_only:
        print("Prerequisite check only — nothing will be built or installed.\n")
        print(format_readiness_report(check_all_sources(repo=args.repo)))
        return 0

    for step in (step_deps, step_ollama, step_model):
        code = step()
        if code != 0:
            return code
    readiness = step_prereqs(args.repo)
    to_build, error = resolve_build_set(
        all_flag=args.all,
        source=args.source,
        readiness=readiness,
        boto3_installer=ensure_boto3,
    )
    if error is not None:
        return fail(error)
    code = step_indexes(to_build, args.repo)
    if code != 0:
        return code

    # The test question needs the Meridian index; skip it when this run
    # did not build Meridian (e.g. --source jira) instead of failing a
    # run whose requested build succeeded.
    from rag_demo.ingest import INDEX_DIRS
    from rag_demo.preflight import index_present

    if not args.skip_test_question and index_present(REPO_ROOT / INDEX_DIRS["meridian"]):
        code = step_test_question()
        if code != 0:
            return code
    elif not index_present(REPO_ROOT / INDEX_DIRS["meridian"]):
        print("Skipping test question (this run did not build the Meridian index).")
    print("\n✅ SETUP COMPLETE. Start the app with: PYTHONPATH=src streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
