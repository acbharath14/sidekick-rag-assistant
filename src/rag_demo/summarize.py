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


def _invoke(llm, prompt: str, timeout: float = 120.0) -> str:
    """Invoke the LLM with a timeout; returns '' on timeout/failure."""
    import concurrent.futures

    from .chain import strip_think

    def _do():
        out = llm.invoke(prompt)
        text = out.content if hasattr(out, "content") else str(out)
        return strip_think(text.strip())

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(_do)
        try:
            return future.result(timeout=timeout)
        except (concurrent.futures.TimeoutError, Exception):
            future.cancel()
            return ""


def summarize(text: str, settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    llm = get_llm(settings)
    if len(text) <= SINGLE_CALL_CHARS:
        result = _invoke(llm, SUMMARY_PROMPT.format(text=text))
        return result or "Summary timed out — the document may be too long for this model."
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_CHARS,
        chunk_overlap=200,
    )
    chunks = splitter.split_text(text)
    partials = []
    for c in chunks:
        p = _invoke(llm, MAP_PROMPT.format(chunk=c))
        if p:  # skip timed-out chunks rather than failing the whole summary
            partials.append(p)
    if not partials:
        return "Summary timed out — the document may be too long for this model."
    return _invoke(llm, REDUCE_PROMPT.format(summaries="\n\n".join(partials))) or "Summary timed out."
