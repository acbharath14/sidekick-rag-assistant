"""Enterprise connectors: Jira, Confluence, Outlook, AWS S3.

Each implements the DocumentSource interface (load() -> list[Document]),
so they plug into the existing ingest pipeline without changes.

All credentials come from environment variables — never hardcoded, never
committed. See README for the full list.

These are written against public API documentation and follow the same
patterns as the existing GitHubDocsSource. They have not been tested
against live instances; contributions with real-world fixes are welcome.
"""

from __future__ import annotations

import os
from abc import ABC

from langchain_core.documents import Document

from .sources import DocumentSource


def _env(name: str, required: bool = True) -> str | None:
    val = os.environ.get(name)
    if required and not val:
        raise ValueError(
            f"Enterprise connector needs {name} set. "
            f"See README for setup instructions."
        )
    return val


class ConfluenceSource(DocumentSource):
    """Real Confluence pages via the REST API.

    Env:
        RAG_DEMO_CONFLUENCE_URL   e.g. https://your-domain.atlassian.net/wiki
        RAG_DEMO_CONFLUENCE_TOKEN  Personal Access Token (or API token)
        RAG_DEMO_CONFLUENCE_SPACE  Space key to index (optional; omit for all)
        RAG_DEMO_CONFLUENCE_USER   Email for basic auth (Cloud only, optional)

    Uses CQL to list pages, then fetches each page's storage-format body
    and strips HTML to text.
    """

    name = "confluence"

    def load(self) -> list[Document]:
        import requests
        from html.parser import HTMLParser

        base_url = _env("RAG_DEMO_CONFLUENCE_URL").rstrip("/")
        token = _env("RAG_DEMO_CONFLUENCE_TOKEN")
        space = os.environ.get("RAG_DEMO_CONFLUENCE_SPACE")
        user = os.environ.get("RAG_DEMO_CONFLUENCE_USER", "")

        # Cloud uses basic auth (email + API token); Server/DC uses Bearer PAT.
        headers = {}
        auth = None
        if user:
            import base64
            creds = base64.b64encode(f"{user}:{token}".encode()).decode()
            headers["Authorization"] = f"Basic {creds}"
        else:
            headers["Authorization"] = f"Bearer {token}"
        headers["Accept"] = "application/json"

        # CQL: all pages in space, or all pages if no space given.
        cql = f"space={space} AND type=page" if space else "type=page"
        docs: list[Document] = []
        start = 0
        limit = 25

        class _Stripper(HTMLParser):
            def __init__(self):
                super().__init__()
                self.parts: list[str] = []
            def handle_data(self, data: str):
                self.parts.append(data)
            def get_text(self) -> str:
                return " ".join(" ".join(self.parts).split())

        while True:
            resp = requests.get(
                f"{base_url}/rest/api/content/search",
                headers=headers,
                params={"cql": cql, "start": start, "limit": limit,
                        "expand": "body.storage,version"},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if not results:
                break
            for page in results:
                title = page.get("title", "Untitled")
                body_html = page.get("body", {}).get("storage", {}).get("value", "")
                stripper = _Stripper()
                stripper.feed(body_html)
                text = f"# {title}\n\n{stripper.get_text()}"
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source": f"confluence:{title}",
                        "allowed_groups": ["*"],
                    },
                ))
            if len(results) < limit:
                break
            start += limit

        return docs


class JiraSource(DocumentSource):
    """Jira issues via the REST API.

    Env:
        RAG_DEMO_JIRA_URL    e.g. https://your-domain.atlassian.net
        RAG_DEMO_JIRA_TOKEN  Personal Access Token (or API token)
        RAG_DEMO_JIRA_JQL    JQL query (default: recently updated issues)
        RAG_DEMO_JIRA_USER   Email for basic auth (Cloud only, optional)
        RAG_DEMO_JIRA_MAX    Max issues to fetch (default: 100)
    """

    name = "jira"

    def load(self) -> list[Document]:
        import requests

        base_url = _env("RAG_DEMO_JIRA_URL").rstrip("/")
        token = _env("RAG_DEMO_JIRA_TOKEN")
        jql = os.environ.get(
            "RAG_DEMO_JIRA_JQL", "updated >= -30d ORDER BY updated DESC"
        )
        user = os.environ.get("RAG_DEMO_JIRA_USER", "")
        max_results = int(os.environ.get("RAG_DEMO_JIRA_MAX", "100"))

        headers = {"Accept": "application/json"}
        auth = None
        if user:
            import base64
            creds = base64.b64encode(f"{user}:{token}".encode()).decode()
            headers["Authorization"] = f"Basic {creds}"
        else:
            headers["Authorization"] = f"Bearer {token}"

        docs: list[Document] = []
        start_at = 0
        while len(docs) < max_results:
            resp = requests.get(
                f"{base_url}/rest/api/3/search/jql",
                headers=headers,
                params={
                    "jql": jql,
                    "startAt": start_at,
                    "maxResults": min(50, max_results - len(docs)),
                    "fields": "summary,description,status,assignee,comment",
                },
                timeout=30,
            )
            # Fall back to the older /search endpoint if /search/jql 404s.
            if resp.status_code == 404:
                resp = requests.get(
                    f"{base_url}/rest/api/3/search",
                    headers=headers,
                    params={
                        "jql": jql,
                        "startAt": start_at,
                        "maxResults": min(50, max_results - len(docs)),
                        "fields": "summary,description,status,assignee,comment",
                    },
                    timeout=30,
                )
            resp.raise_for_status()
            data = resp.json()
            issues = data.get("issues", [])
            if not issues:
                break
            for issue in issues:
                key = issue.get("key", "UNKNOWN")
                fields = issue.get("fields", {})
                summary = fields.get("summary", "")
                # Description may be Atlassian Document Format (dict) or plain text.
                desc = fields.get("description", "")
                if isinstance(desc, dict):
                    desc = _adf_to_text(desc)
                elif not isinstance(desc, str):
                    desc = str(desc or "")
                status = (fields.get("status") or {}).get("name", "")
                text = (
                    f"# {key}: {summary}\n\n"
                    f"Status: {status}\n\n"
                    f"{desc}"
                )
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source": f"jira:{key}",
                        "allowed_groups": ["*"],
                    },
                ))
            total = data.get("total", 0)
            start_at += len(issues)
            if start_at >= total:
                break

        return docs


def _adf_to_text(adf: dict) -> str:
    """Convert Atlassian Document Format to plain text (best-effort)."""
    parts: list[str] = []

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "text":
                parts.append(node.get("text", ""))
            for child in node.get("content", []):
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(adf)
    return " ".join(" ".join(parts).split())


class OutlookSource(DocumentSource):
    """Outlook emails via Microsoft Graph API.

    Env:
        RAG_DEMO_OUTLOOK_TOKEN   OAuth2 bearer token for Microsoft Graph
        RAG_DEMO_OUTLOOK_FOLDER  Mail folder (default: inbox)
        RAG_DEMO_OUTLOOK_MAX     Max messages (default: 100)
        RAG_DEMO_OUTLOOK_QUERY   $search query (optional)

    Note: obtaining a Graph token requires Azure AD app registration.
    See README for the OAuth flow overview.
    """

    name = "outlook"

    def load(self) -> list[Document]:
        import requests

        token = _env("RAG_DEMO_OUTLOOK_TOKEN")
        folder = os.environ.get("RAG_DEMO_OUTLOOK_FOLDER", "inbox")
        max_msgs = int(os.environ.get("RAG_DEMO_OUTLOOK_MAX", "100"))
        query = os.environ.get("RAG_DEMO_OUTLOOK_QUERY", "")

        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }
        docs: list[Document] = []
        url = f"https://graph.microsoft.com/v1.0/me/mailFolders/{folder}/messages"
        params = {
            "$top": min(50, max_msgs),
            "$select": "subject,body,from,receivedDateTime",
        }
        if query:
            params["$search"] = f'"{query}"'

        while url and len(docs) < max_msgs:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            for msg in data.get("value", []):
                subject = msg.get("subject", "(no subject)")
                body = msg.get("body", {})
                content = body.get("content", "")
                # Strip HTML if needed.
                if body.get("contentType") == "html":
                    from html.parser import HTMLParser

                    class _Stripper(HTMLParser):
                        def __init__(self):
                            super().__init__()
                            self.parts: list[str] = []
                        def handle_data(self, d: str):
                            self.parts.append(d)

                    s = _Stripper()
                    s.feed(content)
                    content = " ".join(" ".join(s.parts).split())
                sender = (msg.get("from") or {}).get("emailAddress", {}).get("address", "")
                date = msg.get("receivedDateTime", "")
                text = f"# {subject}\n\nFrom: {sender}\nDate: {date}\n\n{content}"
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source": f"outlook:{subject[:50]}",
                        "allowed_groups": ["*"],
                    },
                ))
                if len(docs) >= max_msgs:
                    break
            url = data.get("@odata.nextLink")
            params = {}  # nextLink already includes params

        return docs


class AWSS3Source(DocumentSource):
    """Documents from an S3 bucket (PDF, TXT, MD).

    Uses standard boto3 credential resolution: env vars (AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY), ~/.aws/credentials, IAM roles, etc.

    Env:
        RAG_DEMO_AWS_S3_BUCKET  Bucket name (required)
        RAG_DEMO_AWS_S3_PREFIX  Key prefix filter (optional)
        RAG_DEMO_AWS_REGION     Region (optional, defaults to boto3 default)
    """

    name = "aws-s3"

    def load(self) -> list[Document]:
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "AWSS3Source needs boto3: pip install boto3"
            )

        bucket = _env("RAG_DEMO_AWS_S3_BUCKET")
        prefix = os.environ.get("RAG_DEMO_AWS_S3_PREFIX", "")
        region = os.environ.get("RAG_DEMO_AWS_REGION")

        kwargs = {"region_name": region} if region else {}
        s3 = boto3.client("s3", **kwargs)

        docs: list[Document] = []
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                # Only process text-like files.
                if not key.lower().endswith((".txt", ".md", ".pdf")):
                    continue
                try:
                    resp = s3.get_object(Bucket=bucket, Key=key)
                    data = resp["Body"].read()
                    if key.lower().endswith(".pdf"):
                        # Lazy import to avoid hard dependency.
                        from pypdf import PdfReader
                        import io
                        reader = PdfReader(io.BytesIO(data))
                        text = "\n".join(p.extract_text() or "" for p in reader.pages)
                    else:
                        text = data.decode("utf-8", errors="replace")
                    if text.strip():
                        docs.append(Document(
                            page_content=text,
                            metadata={
                                "source": f"s3:{key}",
                                "allowed_groups": ["*"],
                            },
                        ))
                except Exception:
                    continue  # skip unreadable objects

        return docs


# Registry for the ingest CLI.
ENTERPRISE_SOURCES: dict[str, type[DocumentSource]] = {
    "confluence": ConfluenceSource,
    "jira": JiraSource,
    "outlook": OutlookSource,
    "aws-s3": AWSS3Source,
}
