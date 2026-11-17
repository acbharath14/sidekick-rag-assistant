"""Gmail connector: messages via the Gmail REST API.

Env (all from the environment, never hardcoded, never committed):
    GMAIL_ACCESS_TOKEN  An already-acquired OAuth2 access token
    GMAIL_QUERY         Search query (default: in:inbox)
    GMAIL_MAX           Max messages per sync (default: 100)

Auth is a bearer token supplied by the caller — the same shape as the
Outlook connector (an OAuth access token, NOT a password or PAT).
Acquiring that token is the OAuth follow-up (Google Cloud project +
OAuth consent screen + authorization-code flow) and is explicitly NOT
in this module — see docs/connectors.md. Pass a token you already
hold; the client never stores it anywhere but memory, and requests
are logged with method, URL path, and status only — never headers,
never the token.

Incremental sync: the Gmail list endpoint returns messages newest
first, so ``fetch_since`` walks the list and stops at the first
message at or before the stored cutoff (``internalDate``, milliseconds
since epoch, converted to ISO 8601 for ``updated_at``). The sync
layer stores the newest message date as ``last_sync``. Messages are
append-only here: ``present_doc_ids`` is left empty, so a message
that leaves the inbox (archived or deleted) stays in the index rather
than being pruned — a mailbox is a stream, not a directory listing.

A ``historyId``-based sync (the Gmail-native incremental mechanism)
is the documented follow-up; it would catch label changes and
deletions the date cutoff cannot see.
"""

from __future__ import annotations

import base64
import binascii
import logging
import os
from datetime import datetime, timezone
from typing import Iterable

import requests

from .base import Connector, ConnectorDoc
from ..sources import html_to_text

logger = logging.getLogger(__name__)

_PAGE_SIZE = 50
_DEFAULT_BASE_URL = "https://gmail.googleapis.com"


def _iso_from_ms(internal_date: str | None) -> str | None:
    """Gmail internalDate (ms since epoch, as a string) to ISO 8601 UTC."""
    if not internal_date:
        return None
    try:
        ms = int(internal_date)
    except (TypeError, ValueError):
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _ms_from_iso(value: str | None) -> int | None:
    """ISO 8601 timestamp to milliseconds since epoch, or None."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _decode_b64url(data: str | None) -> str:
    """Base64url body data to text (Gmail omits padding)."""
    if not data:
        return ""
    try:
        padded = data + "=" * (-len(data) % 4)
        return base64.urlsafe_b64decode(padded).decode("utf-8", errors="replace")
    except (binascii.Error, ValueError):
        return ""


def _header(payload: dict, name: str) -> str:
    for header in payload.get("headers", []) or []:
        if str(header.get("name", "")).lower() == name.lower():
            return str(header.get("value", ""))
    return ""


def _body_text(payload: dict) -> str:
    """Plain text of a message payload, preferring text/plain parts.

    Walks multipart payloads depth-first. A text/plain part wins over
    text/html; HTML is stripped to text. Falls back to the snippet
    when the payload carries no usable body.
    """
    plain: list[str] = []
    html: list[str] = []

    def walk(part: dict) -> None:
        mime = part.get("mimeType", "")
        data = (part.get("body") or {}).get("data")
        if mime == "text/plain" and data:
            plain.append(_decode_b64url(data))
        elif mime == "text/html" and data:
            html.append(_decode_b64url(data))
        for sub in part.get("parts", []) or []:
            walk(sub)

    walk(payload)
    if plain:
        return "\n\n".join(p.strip() for p in plain if p.strip())
    if html:
        return "\n\n".join(
            html_to_text(h).strip() for h in html if h.strip()
        )
    return ""


class GmailConnector(Connector):
    source_name = "gmail"

    def __init__(self, session: requests.Session | None = None):
        super().__init__()
        self._session = session or requests.Session()
        self._token: str | None = None
        self.query: str = "in:inbox"
        self.max_messages: int = 100
        # Base host only — the client appends /gmail/v1. Tests point
        # this at a local mock server via GMAIL_BASE_URL.
        self.base_url: str = _DEFAULT_BASE_URL

    def connect(self, config: dict | None = None) -> None:
        cfg = config or {}
        self._token = cfg.get("access_token") or os.environ.get("GMAIL_ACCESS_TOKEN")
        self.query = cfg.get("query") or os.environ.get("GMAIL_QUERY", "in:inbox")
        self.max_messages = int(cfg.get("max_messages") or os.environ.get("GMAIL_MAX", "100"))
        self.base_url = (cfg.get("base_url") or os.environ.get("GMAIL_BASE_URL", _DEFAULT_BASE_URL)).rstrip("/")
        if not self._token:
            raise ValueError("Gmail connector needs GMAIL_ACCESS_TOKEN set.")

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
        cutoff_ms = _ms_from_iso(since)
        list_url = f"{self.base_url}/gmail/v1/users/me/messages"
        params: dict[str, str | int] = {
            "q": self.query,
            "maxResults": min(_PAGE_SIZE, self.max_messages),
        }
        page_token: str | None = None
        fetched = 0
        newest_ms: int | None = None
        while fetched < self.max_messages:
            if page_token:
                params["pageToken"] = page_token
            data = self._get(list_url, params)
            messages = data.get("messages", []) or []
            if not messages:
                break
            stop = False
            for entry in messages:
                msg_id = entry.get("id")
                if not msg_id:
                    continue
                full = self._get(
                    f"{self.base_url}/gmail/v1/users/me/messages/{msg_id}",
                    {"format": "full"},
                )
                internal_ms = None
                try:
                    internal_ms = int(full.get("internalDate", ""))
                except (TypeError, ValueError):
                    internal_ms = None
                # The list is newest first: at or before the cutoff,
                # everything after is older too — stop here.
                if cutoff_ms is not None and internal_ms is not None and internal_ms <= cutoff_ms:
                    stop = True
                    break
                payload = full.get("payload", {}) or {}
                subject = _header(payload, "Subject") or "(no subject)"
                sender = _header(payload, "From")
                date = _header(payload, "Date")
                body = _body_text(payload) or full.get("snippet", "")
                updated_at = _iso_from_ms(full.get("internalDate"))
                text = f"# {subject}\n\nFrom: {sender}\nDate: {date}\n\n{body}".strip()
                if internal_ms is not None and (newest_ms is None or internal_ms > newest_ms):
                    newest_ms = internal_ms
                    self.state["newest_internal_date_ms"] = newest_ms
                yield ConnectorDoc(
                    title=subject,
                    text=text,
                    source_label=f"gmail:{subject}",
                    updated_at=updated_at,
                    url=f"https://mail.google.com/mail/u/0/#inbox/{msg_id}",
                    doc_id=f"gmail:{msg_id}",
                )
                fetched += 1
                if fetched >= self.max_messages:
                    break
            if stop:
                break
            page_token = data.get("nextPageToken")
            if not page_token:
                break
