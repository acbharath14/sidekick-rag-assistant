"""Phase 4 tests: Demo/workspace modes, pre-flight checks, setup script
detection logic. Run with RAG_DEMO_FAKE=1 like the rest of the suite;
nothing here touches Ollama, a model, or Streamlit."""

import os

os.environ["RAG_DEMO_FAKE"] = "1"

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_demo.ingest import INDEX_DIRS
from rag_demo.modes import (
    CORPUS_DISPLAY,
    DEMO_CORPORA,
    WORKSPACE_CORPORA,
    WORKSPACE_ENTERPRISE_SOURCES,
    connected_sources,
    default_mode,
    is_demo_corpus,
    is_workspace_corpus,
    source_status,
)
from rag_demo.preflight import (
    index_present,
    model_in_tags,
    normalize_model_name,
    run_preflight,
)


def _fake_settings(fake=True):
    return SimpleNamespace(
        fake=fake,
        ollama_base_url="http://localhost:11434",
        ollama_model="qwen3:8b",
    )


# ------------------------------------------------------------------ modes ---


def test_default_mode_is_demo_with_nothing_connected():
    assert default_mode([]) == "demo"


def test_default_mode_is_workspace_when_any_workspace_source_connected():
    assert default_mode(["playwright-docs"]) == "workspace"
    assert default_mode(["jira"]) == "workspace"


def test_demo_and_workspace_corpora_do_not_overlap():
    # Structural separation: a corpus cannot be both demo and workspace.
    assert not set(DEMO_CORPORA) & set(WORKSPACE_CORPORA)
    assert "meridian" not in WORKSPACE_CORPORA
    assert "confluence-mock" not in WORKSPACE_CORPORA


def test_corpus_classification():
    assert is_demo_corpus("meridian")
    assert is_demo_corpus("confluence-mock")
    assert not is_demo_corpus("jira")
    assert is_workspace_corpus("playwright-docs")
    assert is_workspace_corpus("confluence")
    assert not is_workspace_corpus("meridian")


def test_demo_labels_carry_fictional_marking():
    for corpus in DEMO_CORPORA:
        assert "fictional" in CORPUS_DISPLAY[corpus].lower()


def test_github_label_names_the_repo_explicitly():
    assert "sidekick-rag-assistant" in CORPUS_DISPLAY["github-docs"]


def test_source_status_three_states():
    assert source_status(False) == "Not connected"
    assert source_status(True) == "Ready"
    assert source_status(False, building=True) == "Index building"
    assert source_status(True, building=True) == "Index building"


def test_connected_sources_reads_disk(tmp_path):
    parent = tmp_path
    (parent / INDEX_DIRS["jira"]).mkdir()
    (parent / INDEX_DIRS["playwright-docs"]).mkdir()
    connected = connected_sources(parent, INDEX_DIRS)
    assert "jira" in connected
    assert "playwright-docs" in connected
    assert "confluence" not in connected
    # Demo corpora are never reported as connected workspace sources.
    (parent / INDEX_DIRS["meridian"]).mkdir()
    assert "meridian" not in connected_sources(parent, INDEX_DIRS)


# --------------------------------------------------------------- preflight ---


def test_normalize_model_name_adds_latest():
    assert normalize_model_name("qwen3") == "qwen3:latest"
    assert normalize_model_name("qwen3:8b") == "qwen3:8b"


def test_model_in_tags_matches_exact_and_latest_equivalence():
    tags = {"models": [{"name": "qwen3:8b"}, {"name": "llama3:latest"}]}
    assert model_in_tags(tags, "qwen3:8b")
    assert model_in_tags(tags, "llama3")
    assert not model_in_tags(tags, "qwen3:30b")
    assert not model_in_tags({"models": []}, "qwen3:8b")


def test_index_present_needs_index_file(tmp_path):
    assert not index_present(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    assert not index_present(empty)
    (empty / "index.faiss").write_bytes(b"fake")
    assert index_present(empty)


def test_preflight_all_ok_with_stubs(tmp_path):
    index_dir = tmp_path / "idx"
    index_dir.mkdir()
    (index_dir / "index.faiss").write_bytes(b"fake")
    result = run_preflight(
        _fake_settings(fake=False),
        "meridian",
        index_dir,
        ollama_check=lambda url: True,
        model_check=lambda url, model: True,
        index_check=lambda path: True,
    )
    assert result.all_ok
    assert result.problems == []


def test_preflight_ollama_down_reports_one_fixable_problem(tmp_path):
    result = run_preflight(
        _fake_settings(fake=False),
        "meridian",
        tmp_path,
        ollama_check=lambda url: False,
        model_check=lambda url, model: True,
        index_check=lambda path: True,
    )
    assert not result.all_ok
    assert not result.ollama_ok
    assert any("ollama serve" in p for p in result.problems)
    # No spurious model problem when Ollama itself is down.
    assert not any("ollama pull" in p for p in result.problems)


def test_preflight_model_missing_reports_pull_fix(tmp_path):
    result = run_preflight(
        _fake_settings(fake=False),
        "meridian",
        tmp_path,
        ollama_check=lambda url: True,
        model_check=lambda url, model: False,
        index_check=lambda path: True,
    )
    assert not result.all_ok
    assert not result.model_ok
    assert any("ollama pull qwen3:8b" in p for p in result.problems)


def test_preflight_index_missing_reports_ingest_fix(tmp_path):
    result = run_preflight(
        _fake_settings(fake=False),
        "meridian",
        tmp_path / "nope",
        ollama_check=lambda url: True,
        model_check=lambda url, model: True,
        index_check=lambda path: False,
    )
    assert not result.all_ok
    assert not result.index_ok
    assert any("rag_demo.ingest --source meridian" in p for p in result.problems)


def test_preflight_fake_mode_skips_ollama_and_model(tmp_path):
    def _must_not_run(*args):
        raise AssertionError("must not be called under RAG_DEMO_FAKE=1")

    result = run_preflight(
        _fake_settings(fake=True),
        "meridian",
        tmp_path,
        ollama_check=_must_not_run,
        model_check=_must_not_run,
        index_check=lambda path: True,
    )
    assert result.ollama_ok and result.model_ok
    assert result.all_ok


# ------------------------------------------------------------------- setup ---


def _load_setup_module():
    path = (
        Path(__file__).resolve().parent.parent / "scripts" / "setup.py"
    )
    spec = importlib.util.spec_from_file_location("sidekick_setup", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_setup_install_command_matrix():
    setup = _load_setup_module()
    assert setup.install_command("linux", False, False) == [
        "sh",
        "-c",
        "curl -fsSL https://ollama.com/install.sh | sh",
    ]
    assert setup.install_command("darwin", True, False) == [
        "brew",
        "install",
        "ollama",
    ]
    assert setup.install_command("darwin", False, False) is None
    assert setup.install_command("win32", False, True) == [
        "winget",
        "install",
        "Ollama.Ollama",
    ]
    assert setup.install_command("win32", False, False) is None
    assert setup.install_command("sunos", False, False) is None


def test_setup_manual_steps_name_the_download():
    setup = _load_setup_module()
    assert "ollama.com/download" in setup.manual_install_steps("darwin")
    assert "ollama.com/download" in setup.manual_install_steps("win32")
