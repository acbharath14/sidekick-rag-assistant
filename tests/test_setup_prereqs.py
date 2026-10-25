"""Setup-script prerequisite checks: readiness matrix, boto3/repo
detection, skip-vs-fail behavior, index actions.

Run with RAG_DEMO_FAKE=1 like the rest of the suite; nothing here
touches Ollama, a model, or Streamlit. The readiness functions are
pure (injectable env / git / boto3 probes), following the
install_command pattern from Phase 4.
"""

import importlib.util
import os
import sys
from pathlib import Path

os.environ["RAG_DEMO_FAKE"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _load_setup_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "setup.py"
    spec = importlib.util.spec_from_file_location("sidekick_setup_prereqs", path)
    module = importlib.util.module_from_spec(spec)
    # setup.py uses @dataclass with `from __future__ import annotations`,
    # so its module must be registered before exec or dataclass
    # resolution fails on sys.modules[__name__] being None. (Phase 4's
    # loader omits this and gets away with it only because that module
    # had no dataclasses.)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


setup = _load_setup_module()

EMPTY_ENV = {}
FULL_ENV = {
    "RAG_DEMO_GITHUB_REPO": "acbharath14/sidekick-rag-assistant",
    "RAG_DEMO_CONFLUENCE_URL": "https://example.atlassian.net/wiki",
    "RAG_DEMO_CONFLUENCE_TOKEN": "token",
    "RAG_DEMO_JIRA_URL": "https://example.atlassian.net",
    "RAG_DEMO_JIRA_TOKEN": "token",
    "RAG_DEMO_OUTLOOK_TOKEN": "token",
    "RAG_DEMO_AWS_S3_BUCKET": "bucket",
}


# ------------------------------------------------------------ readiness ---

def test_demo_sources_always_ready():
    for source in ("meridian", "confluence-mock"):
        r = setup.check_source_readiness(source, env=EMPTY_ENV)
        assert r.status == "ready" and r.reason == ""


def test_playwright_docs_needs_git():
    blocked = setup.check_source_readiness(
        "playwright-docs", env=EMPTY_ENV, has_git=False
    )
    assert blocked.status == "skipped"
    assert "git" in blocked.reason
    ready = setup.check_source_readiness(
        "playwright-docs", env=EMPTY_ENV, has_git=True
    )
    assert ready.status == "ready"


def test_github_docs_repo_flag_and_env_forms():
    assert setup.check_source_readiness(
        "github-docs", env=EMPTY_ENV, repo="owner/name"
    ).status == "ready"
    assert setup.check_source_readiness(
        "github-docs", env=FULL_ENV
    ).status == "ready"
    blocked = setup.check_source_readiness("github-docs", env=EMPTY_ENV)
    assert blocked.status == "skipped"
    assert "--repo" in blocked.reason and "RAG_DEMO_GITHUB_REPO" in blocked.reason


def test_confluence_missing_vars_named():
    both_missing = setup.check_source_readiness("confluence", env=EMPTY_ENV)
    assert both_missing.status == "skipped"
    assert "RAG_DEMO_CONFLUENCE_URL" in both_missing.reason
    assert "RAG_DEMO_CONFLUENCE_TOKEN" in both_missing.reason
    one_missing = setup.check_source_readiness(
        "confluence",
        env={"RAG_DEMO_CONFLUENCE_URL": "https://example.atlassian.net/wiki"},
    )
    assert one_missing.status == "skipped"
    assert "RAG_DEMO_CONFLUENCE_URL" not in one_missing.reason
    assert "RAG_DEMO_CONFLUENCE_TOKEN" in one_missing.reason
    assert setup.check_source_readiness("confluence", env=FULL_ENV).status == "ready"


def test_jira_missing_vars_named():
    missing = setup.check_source_readiness("jira", env=EMPTY_ENV)
    assert missing.status == "skipped"
    assert "RAG_DEMO_JIRA_URL" in missing.reason
    assert "RAG_DEMO_JIRA_TOKEN" in missing.reason
    assert setup.check_source_readiness("jira", env=FULL_ENV).status == "ready"


def test_outlook_token():
    missing = setup.check_source_readiness("outlook", env=EMPTY_ENV)
    assert missing.status == "skipped"
    assert "RAG_DEMO_OUTLOOK_TOKEN" in missing.reason
    assert setup.check_source_readiness("outlook", env=FULL_ENV).status == "ready"


def test_aws_s3_matrix():
    # Bucket set + boto3 present: ready.
    assert setup.check_source_readiness(
        "aws-s3", env=FULL_ENV, boto3_ok=True
    ).status == "ready"
    # Bucket set + boto3 missing: blocked (intent signaled; setup installs).
    blocked = setup.check_source_readiness(
        "aws-s3", env=FULL_ENV, boto3_ok=False
    )
    assert blocked.status == "blocked"
    assert "boto3" in blocked.reason
    # Bucket missing + boto3 present: skipped, bucket named.
    skipped = setup.check_source_readiness(
        "aws-s3", env=EMPTY_ENV, boto3_ok=True
    )
    assert skipped.status == "skipped"
    assert "RAG_DEMO_AWS_S3_BUCKET" in skipped.reason
    assert "boto3" not in skipped.reason
    # Both missing: skipped, both named (not blocked — no S3 intent).
    both = setup.check_source_readiness("aws-s3", env=EMPTY_ENV, boto3_ok=False)
    assert both.status == "skipped"
    assert "RAG_DEMO_AWS_S3_BUCKET" in both.reason
    assert "boto3" in both.reason


def test_unknown_source_raises():
    try:
        setup.check_source_readiness("not-a-source", env=EMPTY_ENV)
    except ValueError as e:
        assert "not-a-source" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown source")


def test_check_all_covers_eight_sources():
    readiness = setup.check_all_sources(
        env=EMPTY_ENV, has_git=True, boto3_ok=True
    )
    assert [r.source for r in readiness] == list(setup.SOURCE_ORDER)
    assert len(readiness) == 8
    by_source = {r.source: r.status for r in readiness}
    assert by_source["meridian"] == "ready"
    assert by_source["confluence-mock"] == "ready"
    assert by_source["playwright-docs"] == "ready"
    assert by_source["github-docs"] == "skipped"
    assert by_source["confluence"] == "skipped"
    assert by_source["jira"] == "skipped"
    assert by_source["outlook"] == "skipped"
    assert by_source["aws-s3"] == "skipped"


def test_format_report_lines():
    readiness = setup.check_all_sources(
        env=EMPTY_ENV, has_git=True, boto3_ok=True
    )
    report = setup.format_readiness_report(readiness)
    assert "   meridian: ready" in report
    assert "   confluence: skipped — RAG_DEMO_CONFLUENCE_URL" in report


# ------------------------------------------------------- index actions ---

def test_index_action_states():
    assert setup.index_action(
        "/tmp/x", present=lambda p: False, current=lambda p: False
    ) == "build"
    assert setup.index_action(
        "/tmp/x", present=lambda p: True, current=lambda p: False
    ) == "rebuild"
    assert setup.index_action(
        "/tmp/x", present=lambda p: True, current=lambda p: True
    ) == "skip"


# ------------------------------------------------- build-set resolution ---

def _readiness_with(statuses):
    return [
        setup.SourceReadiness(source, statuses.get(source, "skipped"))
        for source in setup.SOURCE_ORDER
    ]


def test_default_build_set_is_demo_only():
    readiness = _readiness_with({s: "ready" for s in setup.SOURCE_ORDER})
    to_build, error = setup.resolve_build_set(
        all_flag=False, source=None, readiness=readiness
    )
    assert to_build == ["meridian", "confluence-mock"]
    assert error is None


def test_all_build_set_skips_unconfigured_without_error():
    readiness = _readiness_with({
        "meridian": "ready",
        "confluence-mock": "ready",
        "playwright-docs": "ready",
        "github-docs": "ready",
    })
    to_build, error = setup.resolve_build_set(
        all_flag=True, source=None, readiness=readiness
    )
    assert to_build == ["meridian", "confluence-mock", "playwright-docs", "github-docs"]
    assert error is None


def test_all_build_set_installs_boto3_only_for_blocked_s3():
    calls = []

    def installer():
        calls.append(1)
        return True

    readiness = _readiness_with({
        "meridian": "ready",
        "confluence-mock": "ready",
        "aws-s3": "blocked",
    })
    to_build, error = setup.resolve_build_set(
        all_flag=True, source=None, readiness=readiness,
        boto3_installer=installer,
    )
    assert "aws-s3" in to_build
    assert error is None
    assert len(calls) == 1

    # Installer failure: S3 left out, still no error (skipped, not fatal).
    readiness = _readiness_with({
        "meridian": "ready",
        "confluence-mock": "ready",
        "aws-s3": "blocked",
    })
    to_build, error = setup.resolve_build_set(
        all_flag=True, source=None, readiness=readiness,
        boto3_installer=lambda: False,
    )
    assert "aws-s3" not in to_build
    assert error is None


def test_explicit_source_fails_fast_with_fix():
    readiness = _readiness_with({
        "meridian": "ready",
        "confluence-mock": "ready",
        "github-docs": "skipped",
    })
    # Give the skipped entry its real reason for the message check.
    for r in readiness:
        if r.source == "github-docs":
            r.reason = "pass --repo owner/name or set RAG_DEMO_GITHUB_REPO"
    to_build, error = setup.resolve_build_set(
        all_flag=False, source="github-docs", readiness=readiness
    )
    assert to_build == []
    assert error is not None and "--repo" in error and "RAG_DEMO_GITHUB_REPO" in error


def test_explicit_source_ready_builds():
    readiness = _readiness_with({"jira": "ready"})
    to_build, error = setup.resolve_build_set(
        all_flag=False, source="jira", readiness=readiness
    )
    assert to_build == ["jira"]
    assert error is None


def test_explicit_blocked_s3_installer_paths():
    readiness = _readiness_with({"aws-s3": "blocked"})
    to_build, error = setup.resolve_build_set(
        all_flag=False, source="aws-s3", readiness=readiness,
        boto3_installer=lambda: True,
    )
    assert to_build == ["aws-s3"] and error is None
    to_build, error = setup.resolve_build_set(
        all_flag=False, source="aws-s3", readiness=readiness,
        boto3_installer=lambda: False,
    )
    assert to_build == [] and error is not None and "boto3" in error


def test_check_only_exits_zero_without_building(capsys):
    code = setup.main(["--check-only"])
    assert code == 0
    out = capsys.readouterr().out
    assert "meridian: ready" in out
    assert "nothing will be built" in out
