"""Tests run with RAG_DEMO_FAKE=1: deterministic fakes, no model downloads."""

import os

os.environ["RAG_DEMO_FAKE"] = "1"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json

import pytest

from rag_demo import ingest
from rag_demo.chain import ask, build_chain, format_hits
from rag_demo.config import get_settings
from rag_demo.retriever import Retriever


@pytest.fixture(scope="module")
def settings(tmp_path_factory):
    s = get_settings()
    s.index_dir = tmp_path_factory.mktemp("index")
    return s


@pytest.fixture(scope="module")
def _indexed(settings):
    ingest.build_index(settings)
    return settings


def test_ingest_indexes_all_docs(settings):
    store = ingest.build_index(settings)
    assert store.index.ntotal > 0


def test_chunks_carry_source_metadata(settings):
    docs = ingest.load_documents(settings)
    chunks = ingest.chunk_documents(docs, settings)
    assert chunks, "expected at least one chunk"
    assert all(c.metadata.get("source", "").endswith(".md") for c in chunks)


def test_retriever_returns_hits_with_sources(_indexed):
    r = Retriever(_indexed)
    hits = r.search("API token expiry", k=3)
    assert len(hits) == 3
    assert all(h.source.endswith(".md") and h.text for h in hits)


def test_retriever_returns_k_hits(_indexed):
    # With FakeEmbeddings the vector space is random, so this asserts
    # structure, not which doc wins. Real-embedding quality is measured
    # by evals/eval_retrieval.py.
    r = Retriever(_indexed)
    hits = r.search("anything at all", k=4)
    assert len(hits) == 4
    assert all(h.source.endswith(".md") and h.text for h in hits)


def test_chain_answers_with_citations(_indexed):
    answer = ask("What is the max parcel weight?", _indexed)
    assert answer.text
    assert answer.sources, "expected source citations"


def test_format_hits_marks_sources():
    from rag_demo.retriever import Hit

    out = format_hits([Hit(text="hello", source="a.md", score=0.1)])
    assert "[a.md]" in out and "hello" in out


def test_golden_eval_runs_end_to_end(_indexed, monkeypatch):
    # The fake embedding space is random; this asserts the eval harness runs
    # end-to-end and reports a rate between 0 and 1.
    import evals.eval_retrieval as ev

    monkeypatch.setattr(ev, "THRESHOLD", 0.0)
    assert ev.main(settings=_indexed) == 0


def test_golden_file_is_well_formed():
    golden = json.loads(
        Path(__file__).resolve().parent.parent.joinpath("evals/golden.json").read_text()
    )
    assert len(golden) >= 10
    for item in golden:
        assert item["question"] and item["expected_sources"]


def test_markdown_source_loads_docs_with_citation_names():
    from rag_demo.sources import MarkdownDirectorySource

    docs = MarkdownDirectorySource(
        Path(__file__).resolve().parent.parent / "docs"
    ).load()
    assert len(docs) == 5
    assert all(d.metadata["source"].endswith(".md") for d in docs)
    assert {d.metadata["source"] for d in docs} == {
        "api-reference.md",
        "deployment-runbook.md",
        "onboarding.md",
        "security-policy.md",
        "troubleshooting.md",
    }


def test_ingest_accepts_explicit_source(settings):
    from rag_demo.sources import MarkdownDirectorySource

    source = MarkdownDirectorySource(
        Path(__file__).resolve().parent.parent / "docs"
    )
    store = ingest.build_index(settings, source)
    assert store.index.ntotal > 0


def test_playwright_source_is_registered():
    from rag_demo.sources import SOURCES, PlaywrightDocsSource

    assert SOURCES["playwright-docs"] is PlaywrightDocsSource
    assert SOURCES["meridian"].__name__ == "MarkdownDirectorySource"


def test_rrf_fuse_orders_by_rank():
    from rag_demo.retriever import Hit, rrf_fuse

    a = Hit(text="doc-a", source="a.md", score=0.0)
    b = Hit(text="doc-b", source="b.md", score=0.0)
    fused = rrf_fuse([[a, b], [a]], k=2)
    assert [h.source for h in fused] == ["a.md", "b.md"]
    assert fused[0].score > fused[1].score


def test_rrf_fuse_dedupes_by_text_and_source():
    from rag_demo.retriever import Hit, rrf_fuse

    a1 = Hit(text="same", source="a.md", score=0.0)
    a2 = Hit(text="same", source="a.md", score=0.0)
    fused = rrf_fuse([[a1], [a2]], k=2)
    assert len(fused) == 1
    # ranked 1st in both lists: 2 * 1/(60+1)
    assert fused[0].score == pytest.approx(2 / 61)


def test_chunks_json_written_on_ingest(_indexed):
    p = _indexed.index_dir / "chunks.json"
    assert p.exists(), "ingest should persist chunks.json for hybrid retrieval"
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data and all({"text", "source", "doc_id"} <= set(c) for c in data)


def test_hybrid_bm25_finds_keyword_doc(_indexed, monkeypatch):
    # BM25 is deterministic; assert on it directly instead of the fused
    # ranking (the dense side uses random FakeEmbeddings in tests).
    monkeypatch.setenv("RAG_DEMO_RETRIEVAL", "hybrid")
    s = get_settings()
    s.index_dir = _indexed.index_dir
    r = Retriever(s)
    assert r.bm25 is not None
    hits = r._bm25_search("weight_kg", k=3)
    assert any(h.source == "api-reference.md" for h in hits)


def test_hybrid_search_returns_k_hits(_indexed, monkeypatch):
    monkeypatch.setenv("RAG_DEMO_RETRIEVAL", "hybrid")
    s = get_settings()
    s.index_dir = _indexed.index_dir
    r = Retriever(s)
    hits = r.search("weight_kg", k=3)
    assert len(hits) == 3


def test_rewrite_no_history_returns_question_unchanged():
    from rag_demo.rewrite import rewrite_query

    assert rewrite_query("hello?", [], llm=None) == "hello?"


def test_rewrite_uses_history_with_fake_llm():
    from langchain_core.language_models.fake import FakeListLLM

    from rag_demo.rewrite import rewrite_query

    llm = FakeListLLM(responses=["What is the max parcel weight for express?"])
    out = rewrite_query(
        "what about express?",
        [("What is the max parcel weight?", "1200 kg per parcel")],
        llm,
    )
    assert out == "What is the max parcel weight for express?"


def test_reranker_import_is_lazy():
    # Importing the module must not pull in sentence_transformers/torch (CI).
    import sys

    assert "sentence_transformers" not in sys.modules
    import rag_demo.rerank  # noqa: F401

    assert "sentence_transformers" not in sys.modules


def test_ask_accepts_history(_indexed):
    answer = ask(
        "what about express?",
        _indexed,
        history=[("What is the max parcel weight?", "1200 kg per parcel")],
    )
    assert answer.text


def test_confluence_mock_loads_pages_with_acls():
    from rag_demo.sources import MockConfluenceSource

    docs = MockConfluenceSource(
        Path(__file__).resolve().parent.parent / "fixtures" / "confluence"
    ).load()
    assert len(docs) == 4
    by_source = {d.metadata["source"]: d for d in docs}
    assert "confluence:ENG/Salary bands FY27" in by_source
    restricted = by_source["confluence:ENG/Salary bands FY27"]
    assert restricted.metadata["allowed_groups"] == ["eng-leads"]
    public = by_source["confluence:ENG/Deploy checklist"]
    assert public.metadata["allowed_groups"] == ["*"]
    assert "Deploy checklist" in public.page_content  # HTML was stripped


def test_html_to_text_strips_tags():
    from rag_demo.sources import html_to_text

    assert html_to_text("<h1>Title</h1><p>Some <b>bold</b> text.</p>") == "Title\n\nSome bold text."


def test_github_source_parses_tree_and_blobs(monkeypatch):
    import base64

    from rag_demo.sources import GitHubDocsSource

    tree = {"tree": [
        {"type": "blob", "path": "docs/guide.md", "sha": "abc123"},
        {"type": "blob", "path": "src/main.py", "sha": "def456"},
    ], "truncated": False}
    blob = {"content": base64.b64encode(b"# Guide\n\nHello docs.").decode()}

    def fake_api(self, path):
        return blob if "/git/blobs/" in path else tree

    monkeypatch.setattr(GitHubDocsSource, "_api", fake_api)
    docs = GitHubDocsSource(repo="owner/repo").load()
    assert len(docs) == 1  # only the .md file
    assert docs[0].metadata["source"] == "github:owner/repo/docs/guide.md"
    assert "Hello docs" in docs[0].page_content
    assert docs[0].metadata["allowed_groups"] == ["*"]


def test_github_source_truncated_tree_raises(monkeypatch):
    from rag_demo.sources import GitHubDocsSource

    monkeypatch.setattr(
        GitHubDocsSource, "_api", lambda self, path: {"tree": [], "truncated": True}
    )
    with pytest.raises(RuntimeError, match="truncated"):
        GitHubDocsSource(repo="owner/repo").load()


def test_acl_filtering_hides_restricted_hits(_indexed_confluence):
    settings, _ = _indexed_confluence
    r = Retriever(settings)
    # eng-leads sees the restricted page; eng-all does not.
    leads_hits = r.search("salary bands", k=10, user_groups=["eng-leads"])
    assert any("Salary bands" in h.source for h in leads_hits)
    all_hits = r.search("salary bands", k=10, user_groups=["eng-all"])
    assert not any("Salary bands" in h.source for h in all_hits)
    # Default-deny: no/empty groups see public docs only, never restricted.
    for groups in (None, []):
        hits = r.search("salary bands", k=10, user_groups=groups)
        assert not any("Salary bands" in h.source for h in hits)
        assert hits  # public docs still visible


def test_chain_prompt_never_sees_restricted_text(_indexed_confluence, monkeypatch):
    # The retrieval feeding the LLM prompt must be ACL-filtered, not just
    # the displayed sources. Echo-LLM returns the prompt it received.
    from langchain_core.runnables import RunnableLambda

    import rag_demo.chain as chain_mod

    settings, _ = _indexed_confluence
    monkeypatch.setattr(
        chain_mod,
        "get_llm",
        lambda s: RunnableLambda(lambda prompt: prompt.to_string()),
    )
    chain, _, groups = build_chain(settings, user_groups=["eng-all"])
    assert groups == ["eng-all"]
    assert "Salary bands" not in chain.invoke("salary bands")

    chain, _, _ = build_chain(settings, user_groups=["eng-leads"])
    assert "Salary bands" in chain.invoke("salary bands")

    # Empty groups default-deny at the chain level too.
    chain, _, groups = build_chain(settings, user_groups=[])
    assert groups == []
    assert "Salary bands" not in chain.invoke("salary bands")


def test_audit_log_writes_one_line_per_ask(_indexed, tmp_path, monkeypatch):
    from rag_demo.audit import log_question
    from rag_demo.chain import Answer

    log_path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("RAG_DEMO_AUDIT_LOG", str(log_path))
    s = get_settings()
    answer = Answer(text="hi", sources=["a.md"], hits=[])
    log_question(s, "hello?", answer, 12.5)
    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["question"] == "hello?"
    assert entry["sources"] == ["a.md"]
    assert entry["latency_ms"] == 12.5
    assert "answer" not in entry  # answers not logged by default


def test_audit_log_disabled_by_default(_indexed, tmp_path, monkeypatch):
    from rag_demo.audit import log_question
    from rag_demo.chain import Answer

    monkeypatch.delenv("RAG_DEMO_AUDIT_LOG", raising=False)
    s = get_settings()
    log_question(s, "hello?", Answer(text="hi", sources=[], hits=[]), 1.0)
    # No file created anywhere we can see; just assert no crash.
    assert True


@pytest.fixture(scope="module")
def _indexed_confluence(tmp_path_factory):
    from rag_demo.sources import MockConfluenceSource

    s = get_settings()
    s.index_dir = tmp_path_factory.mktemp("index-confluence")
    ingest.build_index(
        s,
        MockConfluenceSource(
            Path(__file__).resolve().parent.parent / "fixtures" / "confluence"
        ),
    )
    return s, True


def test_incremental_ingest_skips_when_unchanged(_indexed, capsys):
    store = ingest.build_index(_indexed, incremental=True)
    assert store is None
    out = capsys.readouterr().out
    assert "up to date" in out


def test_incremental_ingest_rebuilds_when_changed(_indexed, tmp_path):
    from rag_demo.sources import MarkdownDirectorySource

    # Point a fresh docs dir at the corpus plus one new file.
    import shutil

    docs_dir = tmp_path / "docs2"
    shutil.copytree(
        Path(__file__).resolve().parent.parent / "docs", docs_dir
    )
    (docs_dir / "extra.md").write_text("# Extra\n\nBrand new content here.")
    s = get_settings()
    s.index_dir = tmp_path / "index-incr"
    source = MarkdownDirectorySource(docs_dir)
    store1 = ingest.build_index(s, source)
    n1 = store1.index.ntotal
    store2 = ingest.build_index(s, source, incremental=True)
    assert store2 is None  # unchanged since the just-finished build
    (docs_dir / "extra.md").write_text("# Extra\n\nChanged content now.")
    store3 = ingest.build_index(s, source, incremental=True)
    assert store3 is not None
    assert store3.index.ntotal == n1  # same file count, rebuilt


def test_manifest_written_on_ingest(_indexed):
    manifest = json.loads(
        (_indexed.index_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["documents"]
    assert all("sha256" in v for v in manifest["documents"].values())


def test_stream_ask_yields_tokens_then_answer(_indexed):
    from rag_demo.chain import stream_ask

    events = list(stream_ask("What is the max parcel weight?", _indexed))
    tokens = [e["token"] for e in events if "token" in e]
    answers = [e["answer"] for e in events if "answer" in e]
    assert tokens and "".join(tokens)
    assert len(answers) == 1
    assert answers[0].text == "".join(tokens)
    assert answers[0].sources


def test_system_prompt_handles_small_talk():
    from rag_demo.chain import SYSTEM

    assert "greetings" in SYSTEM and "briefly and naturally" in SYSTEM


def test_cli_keeps_history(monkeypatch):
    import rag_demo.cli as cli

    inputs = iter(["first question", "/quit"])
    outputs = []
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr("builtins.print", lambda *a, **k: outputs.append(" ".join(map(str, a))))
    monkeypatch.setenv("RAG_DEMO_FAKE", "1")
    # Run against a fake settings with no index needed — ask() is monkeypatched.
    from rag_demo.chain import Answer

    monkeypatch.setattr(
        cli, "ask", lambda q, settings, history=None: Answer(text="ok", sources=[], hits=[])
    )
    cli.main()
    assert any("ok" in o for o in outputs)


def _fixture(name):
    return Path(__file__).resolve().parent / "fixtures" / name


def test_extract_txt():
    from rag_demo.extract import extract_text

    text = extract_text("sample.txt", b"# Sample\n\nHello from the text fixture.")
    assert "Hello from the text fixture" in text


def _make_pdf_bytes(text: str) -> bytes:
    """Minimal valid PDF with extractable text (proper xref table)."""
    stream_body = b"BT /F1 12 Tf 20 100 Td (%s) Tj ET\n" % text.encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream_body) + stream_body + b"endstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = [b"%PDF-1.4"]
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(sum(len(x) + 1 for x in out))
        out += [b"%d 0 obj" % i, body, b"endobj"]
    xref_pos = sum(len(x) + 1 for x in out)
    out.append(b"xref")
    out.append(b"0 %d" % (len(objs) + 1))
    out.append(b"0000000000 65535 f ")
    out += [b"%010d 00000 n " % off for off in offsets]
    out.append(b"trailer << /Size %d /Root 1 0 R >>" % (len(objs) + 1))
    out += [b"startxref", str(xref_pos).encode(), b"%%EOF"]
    return b"\n".join(out)


def test_extract_pdf():
    from rag_demo.extract import extract_text

    text = extract_text("sample.pdf", _make_pdf_bytes("Hello from the pdf fixture."))
    assert "Hello from the pdf fixture" in text


def test_extract_docx():
    import io

    from docx import Document

    from rag_demo.extract import extract_text

    buf = io.BytesIO()
    doc = Document()
    doc.add_paragraph("Hello from the docx fixture.")
    doc.save(buf)
    text = extract_text("notes.docx", buf.getvalue())
    assert "Hello from the docx fixture" in text


def test_extract_csv_renders_table():
    from rag_demo.extract import extract_text

    text = extract_text("data.csv", b"name,role\nada,engineer\ngrace,lead\n")
    assert "| name | role |" in text
    assert "ada" in text


def test_extract_unsupported_format_raises():
    from rag_demo.extract import UnsupportedFormat, extract_text

    with pytest.raises(UnsupportedFormat):
        extract_text("evil.exe", b"MZ...")


def test_summarize_short_doc_single_call():
    from langchain_core.language_models.fake import FakeListLLM

    from rag_demo.summarize import summarize

    s = get_settings()
    llm = FakeListLLM(responses=["Short summary."])
    import rag_demo.summarize as summod

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(summod, "get_llm", lambda settings=None: llm)
    try:
        assert summarize("A short document.", s) == "Short summary."
    finally:
        monkeypatch.undo()


def test_summarize_long_doc_map_reduce():
    from langchain_core.language_models.fake import FakeListLLM
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    import rag_demo.summarize as summod
    from rag_demo.summarize import CHUNK_CHARS, summarize

    long_text = ("This is sentence number {}. " * 400).format(*range(400))
    assert len(long_text) > 6000
    n_chunks = len(
        RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_CHARS, chunk_overlap=200
        ).split_text(long_text)
    )
    llm = FakeListLLM(responses=["- bullet"] * n_chunks + ["FINAL SUMMARY"])
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(summod, "get_llm", lambda settings=None: llm)
    try:
        out = summarize(long_text, get_settings())
        assert out == "FINAL SUMMARY"
    finally:
        monkeypatch.undo()


def test_retriever_accepts_injected_store(_indexed):
    from langchain_community.vectorstores import FAISS

    from rag_demo.config import get_embeddings
    from rag_demo.retriever import Retriever

    store = FAISS.from_texts(
        ["injected document about wombats"],
        get_embeddings(_indexed),
        metadatas=[{"source": "upload:test.txt", "allowed_groups": ["*"]}],
    )
    r = Retriever(_indexed, store=store, chunks=[
        {"text": "injected document about wombats", "source": "upload:test.txt",
         "allowed_groups": ["*"]}
    ])
    hits = r.search("wombats", k=1)
    assert hits and hits[0].source == "upload:test.txt"


def test_regression_runner_passes_end_to_end(_indexed, monkeypatch):
    # Fake tier: proves the full ask() pipeline runs for every standard
    # prompt (retrieval + chain + citations). Deterministic by design.
    import evals.regression as reg

    assert reg.main(settings=_indexed) == 0


def test_regression_cases_are_well_formed():
    cases = json.loads(
        Path(__file__).resolve().parent.parent.joinpath(
            "evals/regression_cases.json"
        ).read_text()
    )
    assert len(cases) >= 8
    for case in cases:
        assert case["id"] and case["question"]


def test_feedback_log_appends_vote(tmp_path, monkeypatch):
    from rag_demo.audit import log_feedback

    monkeypatch.setenv("RAG_DEMO_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    s = get_settings()
    log_feedback(s, "How do I create a shipment?", "up")
    log_feedback(s, "What is the CEO's color?", "down")
    lines = (tmp_path / "feedback.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    first, second = (json.loads(line) for line in lines)
    assert first["vote"] == "up" and "shipment" in first["question"]
    assert second["vote"] == "down" and first["ts"]


def test_fallback_marker_detection():
    from rag_demo.chain import is_fallback_answer

    assert is_fallback_answer("🌐 General knowledge (not from your corpus). Paris is…")
    assert is_fallback_answer("General knowledge: Paris is the capital of France.")
    assert not is_fallback_answer("Based on the indexed docs: see the cited sources below.")
    assert not is_fallback_answer("Tokens are valid for 90 days [api-reference.md].")


def test_hybrid_system_prompt_selected_by_setting(_indexed, monkeypatch):
    from langchain_core.runnables import RunnableLambda

    import rag_demo.chain as chain_mod

    monkeypatch.setattr(
        chain_mod, "get_llm",
        lambda s: RunnableLambda(lambda prompt: prompt.to_string()),
    )
    _indexed.hybrid_fallback = True
    chain, _, _ = build_chain(_indexed)
    assert "general knowledge" in chain.invoke("What is the capital of France?").lower()

    _indexed.hybrid_fallback = False
    chain, _, _ = build_chain(_indexed)
    assert "ONLY the context" in chain.invoke("What is the capital of France?")


def test_ask_marks_fallback_answer_ungrounded(_indexed, monkeypatch):
    from langchain_core.runnables import RunnableLambda

    import rag_demo.chain as chain_mod

    monkeypatch.setattr(
        chain_mod, "get_llm",
        lambda s: RunnableLambda(
            lambda prompt: "🌐 General knowledge (not from your corpus). Paris."
        ),
    )
    _indexed.hybrid_fallback = True
    ans = ask("What is the capital of France?", _indexed)
    assert ans.grounded is False
    assert ans.text.startswith("🌐")


def test_rewrite_strips_think_blocks():
    from rag_demo.rewrite import rewrite_query

    class FakeLLM:
        def invoke(self, prompt):
            return "<think>Let me think about this.</think>\nWhat is the max parcel weight for express?"

    out = rewrite_query(
        "what about express?",
        [("What is the max parcel weight?", "1200 kg per parcel")],
        FakeLLM(),
    )
    assert out == "What is the max parcel weight for express?"
    assert "<think>" not in out


def test_rewrite_no_history_is_passthrough():
    from rag_demo.rewrite import rewrite_query

    assert rewrite_query("hello?", [], llm=None) == "hello?"
