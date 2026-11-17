"""Local mock servers for the enterprise connectors (mock-first testing).

A tiny stdlib HTTP server pretends to be Confluence Cloud or Microsoft
Graph, serving the JSON fixtures on disk. The connector clients point at
it via env vars, so the same code path runs against the mock and the real
service — only the base URL changes.

Security note: requests are recorded as (method, path, auth-scheme) only.
The Authorization header value is NEVER stored or logged, so a token can
never leak into test output through this module.
"""

from __future__ import annotations

import json
import re
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CONFLUENCE_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "confluence"


def _load_confluence_pages() -> list[dict]:
    pages: list[dict] = []
    for fixture in sorted(CONFLUENCE_FIXTURES.glob("pages-*.json")):
        pages.extend(json.loads(fixture.read_text(encoding="utf-8")).get("results", []))
    return pages


def _load_outlook_messages() -> list[dict]:
    messages: list[dict] = []
    for fixture in sorted((FIXTURES / "outlook").glob("messages-*.json")):
        messages.extend(json.loads(fixture.read_text(encoding="utf-8")).get("value", []))
    # Graph returns newest first (the client orders by receivedDateTime desc).
    messages.sort(key=lambda m: m.get("receivedDateTime", ""), reverse=True)
    return messages


class _BaseHandler(BaseHTTPRequestHandler):
    """Shared plumbing: JSON replies, silent logging, request recording."""

    requests_log: list[dict] = []  # per-server log, reset on start

    def log_message(self, *args):  # silence default stderr logging
        pass

    def _record(self, path: str) -> None:
        auth = self.headers.get("Authorization", "")
        scheme = auth.split(" ", 1)[0] if auth else "missing"
        self.requests_log.append({"method": self.command, "path": path, "auth": scheme})

    def _reply(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class MockConfluenceHandler(_BaseHandler):
    """Confluence Cloud search: /rest/api/content/search?cql&start&limit.

    Understands the CQL the connector emits: ``space=<KEY>`` and
    ``lastModified >= <yyyy-mm-dd>``. Paginates with start/limit.
    """

    pages = _load_confluence_pages()

    def do_GET(self):
        parsed = urlparse(self.path)
        self._record(parsed.path)
        if parsed.path != "/rest/api/content/search":
            self._reply({"message": "not found"}, status=404)
            return
        qs = parse_qs(parsed.query)
        cql = qs.get("cql", [""])[0]
        space = re.search(r"space=(\w+)", cql)
        date = re.search(r"lastModified >= (\d{4}-\d{2}-\d{2})", cql)
        start = int(qs.get("start", ["0"])[0])
        limit = int(qs.get("limit", ["25"])[0])
        results = list(self.pages)
        if space:
            results = [p for p in results if (p.get("space") or {}).get("key") == space.group(1)]
        if date:
            results = [p for p in results if (p.get("version") or {}).get("when", "")[:10] >= date.group(1)]
        page = results[start:start + limit]
        self._reply({"results": page, "start": start, "limit": limit, "size": len(page)})


class MockGraphHandler(_BaseHandler):
    """Microsoft Graph messages: /v1.0/me/mailFolders/{folder}/messages.

    Understands ``$filter=receivedDateTime ge <ts>``, ``$top``, and
    ``$skiptoken`` pagination via @odata.nextLink.
    """

    messages = _load_outlook_messages()

    def do_GET(self):
        parsed = urlparse(self.path)
        self._record(parsed.path)
        m = re.match(r"^/v1\.0/me/mailFolders/[^/]+/messages$", parsed.path)
        if not m:
            self._reply({"message": "not found"}, status=404)
            return
        qs = parse_qs(parsed.query, keep_blank_values=True)
        flt = qs.get("$filter", [""])[0]
        date = re.search(r"receivedDateTime ge (\S+)", flt)
        top = int(qs.get("$top", ["50"])[0])
        skip = int(qs.get("$skiptoken", ["0"])[0])
        results = list(self.messages)
        if date:
            results = [msg for msg in results if msg.get("receivedDateTime", "") >= date.group(1)]
        page = results[skip:skip + top]
        payload: dict = {"value": page}
        if skip + top < len(results):
            host = f"http://127.0.0.1:{self.server.server_port}"
            payload["@odata.nextLink"] = (
                f"{host}{parsed.path}?$skiptoken={skip + top}&$top={top}"
            )
        self._reply(payload)


@contextmanager
def mock_server(handler_cls):
    """Start a mock server on a free localhost port. Yields (base_url, log).

    `log` is the per-server request record: (method, path, auth-scheme).
    """
    handler_cls.requests_log = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", handler_cls.requests_log
    finally:
        server.shutdown()
        server.server_close()
