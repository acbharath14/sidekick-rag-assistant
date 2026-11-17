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

import base64
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


def _load_gmail_messages() -> list[dict]:
    messages: list[dict] = []
    for fixture in sorted((FIXTURES / "gmail").glob("messages-*.json")):
        messages.extend(json.loads(fixture.read_text(encoding="utf-8")).get("messages", []))
    # Gmail lists newest first (internalDate descending).
    messages.sort(key=lambda m: int(m.get("internalDate", "0")), reverse=True)
    return messages


class MockGmailHandler(_BaseHandler):
    """Gmail REST mock: /gmail/v1/users/me/messages (+/{id}).

    The list returns message ids newest first, paginated with
    ``maxResults``/``pageToken`` (the token is a plain offset). The
    full-message endpoint serves the fixture payload. Requires an
    Authorization header (401 without one).
    """

    messages = _load_gmail_messages()

    def do_GET(self):
        self._record(self.path)
        if not self.headers.get("Authorization"):
            self._reply({"message": "Login Required"}, status=401)
            return
        parsed = urlparse(self.path)
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) < 4 or parts[:3] != ["gmail", "v1", "users"]:
            self._reply({"message": "not found"}, status=404)
            return
        if len(parts) == 5 and parts[3] == "me" and parts[4] == "messages":
            qs = parse_qs(parsed.query)
            max_results = int(qs.get("maxResults", ["50"])[0])
            offset = int(qs.get("pageToken", ["0"])[0])
            page = self.messages[offset:offset + max_results]
            payload: dict = {
                "messages": [
                    {"id": m["id"], "threadId": m.get("threadId", "")}
                    for m in page
                ],
                "resultSizeEstimate": len(self.messages),
            }
            if offset + max_results < len(self.messages):
                payload["nextPageToken"] = str(offset + max_results)
            self._reply(payload)
            return
        if len(parts) == 6 and parts[3] == "me" and parts[4] == "messages":
            found = next(
                (m for m in self.messages if m["id"] == parts[5]), None
            )
            if found is None:
                self._reply({"message": "not found"}, status=404)
                return
            self._reply(found)
            return
        self._reply({"message": "not found"}, status=404)


def _load_github_fixtures() -> dict:
    """Fake GitHub data, keyed by repo (``owner/name``).

    Tests mutate the returned dict between syncs (e.g. swap in an
    updated tree) to exercise incremental fetching.
    """
    github = FIXTURES / "github"
    data: dict = {"repos": {}}
    repo = "acme/widgets"
    data["repos"][repo] = {
        "readme": json.loads((github / "readme.json").read_text(encoding="utf-8")),
        "tree": json.loads((github / "tree.json").read_text(encoding="utf-8")),
        "blobs": json.loads((github / "blobs.json").read_text(encoding="utf-8")),
        "issues": json.loads((github / "issues.json").read_text(encoding="utf-8")),
        "pulls": json.loads((github / "pulls.json").read_text(encoding="utf-8")),
    }
    return data


class MockGitHubHandler(_BaseHandler):
    """GitHub REST mock: readme, git tree/blobs, issues, pulls.

    Requires an Authorization header (401 without one). Issues support
    a server-side ``since`` filter; issues and pulls paginate with
    ``per_page``/``page`` and a ``Link: rel="next"`` header.
    """

    data: dict = _load_github_fixtures()

    @classmethod
    def reset(cls, data: dict | None = None) -> None:
        cls.data = data if data is not None else _load_github_fixtures()

    def _send(self, payload, status: int = 200, link: str | None = None) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if link:
            self.send_header("Link", link)
        self.end_headers()
        self.wfile.write(body)

    def _paginate(self, items: list, query: dict):
        per_page = int(query.get("per_page", ["100"])[0])
        page = int(query.get("page", ["1"])[0])
        start = (page - 1) * per_page
        chunk = items[start:start + per_page]
        link = None
        if start + per_page < len(items):
            host = f"http://127.0.0.1:{self.server.server_port}"
            path = urlparse(self.path).path
            params = {k: v[0] for k, v in query.items()}
            params["page"] = str(page + 1)
            params.setdefault("per_page", str(per_page))
            qs = "&".join(f"{k}={v}" for k, v in params.items())
            link = f'<{host}{path}?{qs}>; rel="next"'
        return chunk, link

    def do_GET(self):
        # Record the full path (query included): the tests assert on
        # the `since` filter and the `page=2` pagination link.
        self._record(self.path)
        if not self.headers.get("Authorization"):
            self._reply({"message": "Requires authentication"}, status=401)
            return
        parsed = urlparse(self.path)
        parts = [p for p in parsed.path.split("/") if p]
        query = parse_qs(parsed.query)
        if len(parts) < 3 or parts[0] != "repos":
            self._reply({"message": "Not Found"}, status=404)
            return
        repo_data = self.data.get("repos", {}).get(f"{parts[1]}/{parts[2]}")
        if repo_data is None:
            self._reply({"message": "Not Found"}, status=404)
            return
        rest = parts[3:]
        if rest == ["readme"]:
            self._reply(repo_data["readme"])
            return
        if len(rest) == 3 and rest[0] == "git" and rest[1] == "trees":
            self._reply(repo_data["tree"])
            return
        if len(rest) == 3 and rest[0] == "git" and rest[1] == "blobs":
            text = repo_data["blobs"].get(rest[2])
            if text is None:
                self._reply({"message": "Not Found"}, status=404)
                return
            self._reply({
                "sha": rest[2],
                "content": base64.b64encode(text.encode("utf-8")).decode("ascii"),
                "encoding": "base64",
                "size": len(text),
            })
            return
        if rest == ["issues"]:
            items = list(repo_data["issues"])
            since = query.get("since", [None])[0]
            if since:
                items = [i for i in items if (i.get("updated_at") or "") >= since]
            chunk, link = self._paginate(items, query)
            self._send(chunk, link=link)
            return
        if rest == ["pulls"]:
            chunk, link = self._paginate(list(repo_data["pulls"]), query)
            self._send(chunk, link=link)
            return
        self._reply({"message": "Not Found"}, status=404)


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
