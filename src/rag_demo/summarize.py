"""LLM summarization for uploaded documents.

Short docs: one call. Long docs: map-reduce — summarize each chunk, then
summarize the summaries. Works with any langchain LLM (fake LLM in tests).
"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings, get_llm, get_settings

SINGLE_CALL_CHARS = 6000
CHUNK_CHARS = 4000

SUMMARY_PROMPT = """Summarize the document below. Lead with the key points, then list any dates, deadlines, owners, or numbers named. Be concise.

Document:
{text}

Summary:"""

MAP_PROMPT = """Summarize this section of a longer document in 3-5 bullet points. Reply with ONLY the bullets.

Section:
{chunk}"""

REDUCE_PROMPT = """Combine these section summaries into one coherent summary. Lead with the key points, then any dates, deadlines, owners, or numbers. Be concise.

Section summaries:
{summaries}

Summary:"""


def _invoke(llm, prompt: str, timeout: float | None = None) -> str:
    """Invoke the LLM with a timeout; returns '' on timeout/failure."""
    import concurrent.futures
    import os

    from .chain import strip_think

    if timeout is None:
        timeout = float(os.environ.get("RAG_DEMO_SUMMARY_TIMEOUT", "600"))

    def _do():
        out = llm.invoke(prompt)
        text = out.content if hasattr(out, "content") else str(out)
        return strip_think(text.strip())

    # NOTE: not using a context manager — on timeout we abandon the stuck
    # thread via shutdown(wait=False) instead of blocking until Ollama
    # finishes (which is what `with` does and why the UI hung forever).
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = ex.submit(_do)
    try:
        return future.result(timeout=timeout)
    except (concurrent.futures.TimeoutError, Exception):
        return ""
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def summarize(text: str, settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    llm = get_llm(settings)
    if len(text) <= SINGLE_CALL_CHARS:
        result = _invoke(llm, SUMMARY_PROMPT.format(text=text))
        return result or "Summary didn't complete — the model may be overloaded. Try again."
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_CHARS,
        chunk_overlap=200,
    )
    chunks = splitter.split_text(text)
    partials = []
    for i, c in enumerate(chunks):
        p = _invoke(llm, MAP_PROMPT.format(chunk=c))
        if p:
            partials.append(p)
        # else: skip timed-out chunks rather than failing the whole summary
    if not partials:
        return "Summary didn't complete — the model may be overloaded. Try again."
    if len(partials) < len(chunks):
        partials.append(
            f"_Note: {len(chunks) - len(partials)} of {len(chunks)} sections "
            "couldn't be summarized in time and were skipped._"
        )
    return _invoke(llm, REDUCE_PROMPT.format(summaries="\n\n".join(partials))) or "Summary didn't complete — the model may be overloaded. Try again."
