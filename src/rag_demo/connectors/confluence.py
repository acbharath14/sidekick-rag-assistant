"""Confluence connector: pages via the Confluence Cloud REST API.

Env (all from the environment, never hardcoded, never committed):
    CONFLUENCE_URL    e.g. https://your-domain.atlassian.net/wiki
    CONFLUENCE_PAT    Personal Access Token (Server/DC) or API token (Cloud)
    CONFLUENCE_SPACE  Space key to index (optional; omit for all spaces)

Incremental sync uses CQL ``lastModified >= <date>`` so a sync only fetches
pages changed since the stored timestamp. Page content comes from the
storage-format body, stripped to plain text.

The PAT is held on the instance only. Requests are logged with method, URL
path, and status — never headers, never the token.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable

import requests

from .base import Connector, ConnectorDoc
from ..sources import html_to_text

logger = logging.getLogger(__name__)

_PAGE_SIZE = 25


class ConfluenceConnector(Connector):
    source_name = "confluence"

    def __init__(self, session: requests.Session | None = None):
        self._session = session or requests.Session()
        self.base_url: str = ""
        self.space: str | None = None
        self._token: str | None = None

    def connect(self, config: dict | None = None) -> None:
        cfg = config or {}
        self.base_url = (cfg.get("url") or os.environ.get("CONFLUENCE_URL") or "").rstrip("/")
        self._token = cfg.get("pat") or os.environ.get("CONFLUENCE_PAT")
        self.space = cfg.get("space") or os.environ.get("CONFLUENCE_SPACE")
        if not self.base_url:
            raise ValueError("Confluence connector needs CONFLUENCE_URL set.")
        if not self._token:
            raise ValueError("Confluence connector needs CONFLUENCE_PAT set.")

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}", "Accept": "application/json"}

    def _search(self, cql: str, start: int) -> dict:
        url = f"{self.base_url}/rest/api/content/search"
        resp = self._session.get(
            url,
            headers=self._headers(),
            params={
                "cql": cql,
                "start": start,
                "limit": _PAGE_SIZE,
                "expand": "body.storage,version",
            },
            timeout=30,
        )
        logger.debug("GET %s -> %s", url, resp.status_code)
        resp.raise_for_status()
        return resp.json()

    def fetch_since(self, since: str | None) -> Iterable[ConnectorDoc]:
        if not self.base_url:
            self.connect()
        cql = "type=page"
        if self.space:
            cql = f"space={self.space} AND {cql}"
        if since:
            # CQL lastModified takes a date; the stored timestamp is ISO 8601.
            cql = f"{cql} AND lastModified >= {since[:10]}"
        start = 0
        while True:
            data = self._search(cql, start)
            results = data.get("results", [])
            if not results:
                return
            for page in results:
                title = page.get("title", "Untitled")
                space_key = (page.get("space") or {}).get("key") or self.space or "?"
                body_html = page.get("body", {}).get("storage", {}).get("value", "")
                text = html_to_text(body_html)
                updated = (page.get("version") or {}).get("when", "")
                page_id = page.get("id", "")
                yield ConnectorDoc(
                    title=title,
                    text=f"# {title}\n\n{text}" if text else f"# {title}",
                    source_label=f"confluence:{space_key}/{title}",
                    updated_at=updated,
                    url=f"{self.base_url}/pages/viewpage.action?pageId={page_id}" if page_id else None,
                    doc_id=f"confluence:{page_id}" if page_id else None,
                )
            if len(results) < _PAGE_SIZE:
                return
            start += len(results)
