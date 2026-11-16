"""Outlook connector: emails via the Microsoft Graph API.

Env (all from the environment, never hardcoded, never committed):
    OUTLOOK_BEARER_TOKEN  An already-acquired OAuth2 bearer token for Graph
    OUTLOOK_FOLDER        Mail folder (default: inbox)
    OUTLOOK_MAX           Max messages per sync (default: 100)

Auth is a bearer token supplied by the caller. Acquiring that token is the
OAuth follow-up (Azure AD app registration + authorization-code flow) and
is explicitly NOT in this module — see docs/connectors.md. Pass a token you
already hold; the client never stores it anywhere but memory, and requests
are logged with method, URL path, and status only — never headers, never
the token.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable

import requests

from .base import Connector, ConnectorDoc
from ..sources import html_to_text

logger = logging.getLogger(__name__)

_PAGE_SIZE = 50


class OutlookConnector(Connector):
    source_name = "outlook"

    def __init__(self, session: requests.Session | None = None):
        self._session = session or requests.Session()
        self._token: str | None = None
        self.folder: str = "inbox"
        self.max_messages: int = 100
        # Base host only — the client appends /v1.0. Tests point this at a
        # local mock server via OUTLOOK_BASE_URL.
        self.base_url: str = "https://graph.microsoft.com"

    def connect(self, config: dict | None = None) -> None:
        cfg = config or {}
        self._token = cfg.get("bearer_token") or os.environ.get("OUTLOOK_BEARER_TOKEN")
        self.folder = cfg.get("folder") or os.environ.get("OUTLOOK_FOLDER", "inbox")
        self.max_messages = int(cfg.get("max_messages") or os.environ.get("OUTLOOK_MAX", "100"))
        self.base_url = (cfg.get("base_url") or os.environ.get("OUTLOOK_BASE_URL", "https://graph.microsoft.com")).rstrip("/")
        if not self._token:
            raise ValueError("Outlook connector needs OUTLOOK_BEARER_TOKEN set.")

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}", "Accept": "application/json"}

    def _get(self, url: str, params: dict | None = None) -> dict:
        resp = self._session.get(url, headers=self._headers(), params=params or {}, timeout=30)
        logger.debug("GET %s -> %s", url, resp.status_code)
        resp.raise_for_status()
        return resp.json()

    def fetch_since(self, since: str | None) -> Iterable[ConnectorDoc]:
        if not self._token:
            self.connect()
        url = f"{self.base_url}/v1.0/me/mailFolders/{self.folder}/messages"
        params: dict[str, str | int] = {
            "$top": min(_PAGE_SIZE, self.max_messages),
            "$select": "subject,body,from,receivedDateTime,webLink",
            "$orderby": "receivedDateTime desc",
        }
        if since:
            params["$filter"] = f"receivedDateTime ge {since}"
        fetched = 0
        while url and fetched < self.max_messages:
            data = self._get(url, params)
            params = {}  # @odata.nextLink already carries its own query
            for msg in data.get("value", []):
                subject = msg.get("subject", "(no subject)")
                body = msg.get("body", {}) or {}
                content = body.get("content", "") or ""
                if body.get("contentType") == "html":
                    content = html_to_text(content)
                sender = ((msg.get("from") or {}).get("emailAddress") or {}).get("address", "")
                received = msg.get("receivedDateTime", "")
                text = f"# {subject}\n\nFrom: {sender}\nDate: {received}\n\n{content}".strip()
                yield ConnectorDoc(
                    title=subject,
                    text=text,
                    source_label=f"outlook:{subject}",
                    updated_at=received,
                    url=msg.get("webLink"),
                    doc_id=f"outlook:{msg.get('id', subject)}",
                )
                fetched += 1
                if fetched >= self.max_messages:
                    break
            url = data.get("@odata.nextLink")
