"""Demo: run the real sync CLI against the mock servers (no live tokens).

    RAG_DEMO_EMBEDDINGS=fake python scripts/mock_sync_demo.py

Starts mock Confluence, Graph, and GitHub servers on localhost, runs the
sync CLI against each with fake tokens, and prints the summaries. The CLI
uses fake embeddings so nothing is downloaded.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

from mock_servers import (  # noqa: E402
    MockConfluenceHandler,
    MockGitHubHandler,
    MockGraphHandler,
    mock_server,
)


def run_cli(env_extra, args):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["RAG_DEMO_EMBEDDINGS"] = "fake"
    env.update(env_extra)
    out = subprocess.run(
        [sys.executable, "-m", "rag_demo.connectors.sync", *args],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    print(out.stdout.strip())
    if out.returncode != 0:
        print(out.stderr[-2000:], file=sys.stderr)
        sys.exit(1)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        with mock_server(MockConfluenceHandler) as (conf_url, _):
            run_cli(
                {"CONFLUENCE_URL": conf_url, "CONFLUENCE_PAT": "FAKE-DEMO-PAT", "CONFLUENCE_SPACE": "OPS"},
                ["--source", "confluence", "--index-dir", f"{tmp}/conf", "--full"],
            )
            run_cli(
                {"CONFLUENCE_URL": conf_url, "CONFLUENCE_PAT": "FAKE-DEMO-PAT", "CONFLUENCE_SPACE": "OPS"},
                ["--source", "confluence", "--index-dir", f"{tmp}/conf"],
            )
        with mock_server(MockGraphHandler) as (graph_url, _):
            run_cli(
                {"OUTLOOK_BEARER_TOKEN": "FAKE-DEMO-TOKEN", "OUTLOOK_BASE_URL": graph_url},
                ["--source", "outlook", "--index-dir", f"{tmp}/out", "--full"],
            )
        MockGitHubHandler.reset()
        with mock_server(MockGitHubHandler) as (github_url, _):
            run_cli(
                {
                    "GITHUB_TOKEN": "FAKE-DEMO-TOKEN",
                    "GITHUB_REPO": "acme/widgets",
                    "GITHUB_BASE_URL": github_url,
                },
                ["--source", "github", "--index-dir", f"{tmp}/gh", "--full"],
            )
            run_cli(
                {
                    "GITHUB_TOKEN": "FAKE-DEMO-TOKEN",
                    "GITHUB_REPO": "acme/widgets",
                    "GITHUB_BASE_URL": github_url,
                },
                ["--source", "github", "--index-dir", f"{tmp}/gh"],
            )
    print("mock sync demo OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
