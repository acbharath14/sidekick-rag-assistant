"""Stdlib mock servers for connector tests.

Each mock serves canned fixture payloads over real HTTP on localhost, so
connector clients run their real code path with only the base URL
changed. No live credentials are needed and no network is touched.

Request logs record (method, path + query, auth scheme) only — the token
value is never stored or logged.
"""

from __future__ import annotations

import base64
import json
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from urllib.parse import parse_qs, urlparse


@contextmanager
def mock_server(handler_cls):
    """Run a mock server on a free localhost port; yield its base URL."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def _auth_scheme(headers) -> str:
    value = headers.get("Authorization") or ""
    if value.startswith("Bearer "):
        return "Bearer"
    if value.startswith("Basic "):
        return "Basic"
    return "none"


class GitHubMockHandler(BaseHTTPRequestHandler):
    """Minimal GitHub REST mock: readme, git tree/blobs, issues, pulls.

    Fixture data, keyed by repo (``owner/name``)::

        data = {"repos": {"acme/widgets": {
            "readme": {"path": ..., "sha": ..., "content": <base64>},
            "tree": {"sha": ..., "tree": [{"path", "sha", "type"}]},
            "blobs": {sha: text},
            "issues": [issue dicts],
            "pulls": [pr dicts],
        }}}

    Tests mutate ``data`` between syncs (e.g. swap in an updated tree)
    to exercise incremental fetching. ``requests`` records every call
    as (method, path + query, auth scheme).
    """

    data: dict = {"repos": {}}
    requests: list = []

    @classmethod
    def reset(cls, data=None):
        cls.data = data if data is not None else {"repos": {}}
        cls.requests = []

    def log_message(self, *args):  # keep test output clean
        pass

    def _record(self):
        self.__class__.requests.append(
            (self.command, self.path, _auth_scheme(self.headers))
        )

    def _send_json(self, payload, status=200, link=None):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if link:
            self.send_header("Link", link)
        self.end_headers()
        self.wfile.write(body)

    def _repo_data(self, owner: str, repo: str):
        return self.data.get("repos", {}).get(f"{owner}/{repo}")

    def _paginate(self, items: list, query: dict):
        per_page = int(query.get("per_page", ["100"])[0])
        page = int(query.get("page", ["1"])[0])
        start = (page - 1) * per_page
        chunk = items[start : start + per_page]
        link = None
        if start + per_page < len(items):
            host = self.headers.get("Host", "127.0.0.1")
            path = urlparse(self.path).path
            params = {k: v[0] for k, v in query.items()}
            params["page"] = str(page + 1)
            params.setdefault("per_page", str(per_page))
            qs = "&".join(f"{k}={v}" for k, v in params.items())
            link = f'<http://{host}{path}?{qs}>; rel="next"'
        return chunk, link

    def do_GET(self):
        self._record()
        if not self.headers.get("Authorization"):
            self._send_json({"message": "Requires authentication"}, status=401)
            return
        parsed = urlparse(self.path)
        parts = [p for p in parsed.path.split("/") if p]
        query = parse_qs(parsed.query)
        if len(parts) < 3 or parts[0] != "repos":
            self._send_json({"message": "Not Found"}, status=404)
            return
        owner, repo = parts[1], parts[2]
        repo_data = self._repo_data(owner, repo)
        if repo_data is None:
            self._send_json({"message": "Not Found"}, status=404)
            return
        rest = parts[3:]

        if rest == ["readme"]:
            self._send_json(repo_data["readme"])
            return
        if len(rest) == 3 and rest[0] == "git" and rest[1] == "trees":
            self._send_json(repo_data["tree"])
            return
        if len(rest) == 3 and rest[0] == "git" and rest[1] == "blobs":
            sha = rest[2]
            text = repo_data["blobs"].get(sha)
            if text is None:
                self._send_json({"message": "Not Found"}, status=404)
                return
            content = base64.b64encode(text.encode("utf-8")).decode("ascii")
            self._send_json(
                {"sha": sha, "content": content, "encoding": "base64", "size": len(text)}
            )
            return
        if rest == ["issues"]:
            items = list(repo_data["issues"])
            since = query.get("since", [None])[0]
            if since:
                items = [i for i in items if (i.get("updated_at") or "") >= since]
            chunk, link = self._paginate(items, query)
            self._send_json(chunk, link=link)
            return
        if rest == ["pulls"]:
            chunk, link = self._paginate(list(repo_data["pulls"]), query)
            self._send_json(chunk, link=link)
            return
        self._send_json({"message": "Not Found"}, status=404)
