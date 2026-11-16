"""GitHub connector: README + docs/*.md + issues + pull requests.

Uses the GitHub REST API with a fine-grained personal access token
(read-only Contents, Issues, and Pull requests scopes). Config comes
from environment variables only — see ``sync.py`` and ``docs/connectors.md``.

Incremental sync, per repo:

- Docs: the git tree is fetched and blob SHAs are diffed against the
  stored state, so only new or changed files are downloaded. Deleted
  files are reported via ``present_doc_ids`` so the sync layer can
  prune them from the store.
- Issues: the issues endpoint supports a server-side ``since`` filter.
- Pull requests: the pulls endpoint has no ``since`` parameter, so the
  client lists (newest first) and filters by ``updated_at`` itself.

Raw code files are deliberately NOT fetched in this pass (docs +
issues + PRs only); that is the documented follow-up.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone

import requests

from .base import Connector, ConnectorDoc

API_VERSION = "2022-11-28"


def _next_link(link_header: str) -> str | None:
    """Next-page URL from a GitHub ``Link`` header, or None."""
    for part in link_header.split(","):
        segments = [s.strip() for s in part.split(";")]
        if len(segments) == 2 and segments[1] == 'rel="next"':
            url = segments[0].strip()
            if url.startswith("<") and url.endswith(">"):
                return url[1:-1]
    return None


def _since_param(since: datetime) -> str:
    if since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)
    return since.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class GitHubConnector(Connector):
    """Read-only GitHub REST client behind the Connector interface."""

    source_name = "github"

    def __init__(self, base_url: str = "https://api.github.com", per_page: int = 100):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.per_page = per_page
        self.token: str | None = None
        self.repos: list[str] = []
        self.branch = "main"

    def connect(self, config: dict) -> None:
        token = config.get("token")
        if not token:
            raise ValueError(
                "GitHub connector needs a token: set GITHUB_TOKEN to a "
                "fine-grained personal access token with read-only "
                "Contents, Issues, and Pull requests scopes "
                "(see docs/connectors.md)."
            )
        repos = list(config.get("repos") or [])
        if not repos and config.get("repo"):
            repos = [config["repo"]]
        if not repos:
            raise ValueError(
                "GitHub connector needs a repo: set GITHUB_REPO to "
                "owner/name, or GITHUB_REPOS for a comma-separated list."
            )
        self.token = token
        self.repos = repos
        if config.get("base_url"):
            self.base_url = str(config["base_url"]).rstrip("/")
        if config.get("per_page"):
            self.per_page = int(config["per_page"])
        if config.get("branch"):
            self.branch = str(config["branch"])
        if config.get("state") is not None:
            self.state = dict(config["state"])

    # -- HTTP ----------------------------------------------------------

    def _get(self, path_or_url: str, params: dict | None = None) -> requests.Response:
        url = (
            path_or_url
            if path_or_url.startswith("http")
            else f"{self.base_url}{path_or_url}"
        )
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": API_VERSION,
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code == 401:
            raise RuntimeError(
                "GitHub rejected the token (401). Check that GITHUB_TOKEN "
                "is a valid fine-grained PAT that has not expired."
            )
        if resp.status_code == 404:
            raise RuntimeError(
                f"GitHub resource not found (404): {url}. Check the repo "
                "name and that the token can access it."
            )
        if resp.status_code == 403:
            raise RuntimeError(
                "GitHub refused the request (403). This is usually the "
                "rate limit or a token without the needed scopes."
            )
        resp.raise_for_status()
        return resp

    def _get_paged(self, path: str, params: dict | None = None):
        """Yield items across pages, following ``Link: rel="next"``."""
        query = dict(params or {})
        query.setdefault("per_page", self.per_page)
        url: str | None = f"{self.base_url}{path}"
        while url:
            resp = self._get(url, params=query)
            yield from resp.json()
            url = _next_link(resp.headers.get("Link", ""))
            query = None  # the next URL already carries its query string

    def _blob_text(self, repo: str, sha: str) -> str:
        blob = self._get(f"/repos/{repo}/git/blobs/{sha}").json()
        return base64.b64decode(blob["content"]).decode("utf-8")

    # -- fetch ---------------------------------------------------------

    def fetch_since(self, since: datetime | None, state: dict | None = None):
        if state is not None:
            self.state = state
        self.present_doc_ids = set()
        repos_state = self.state.setdefault("repos", {})
        for repo in self.repos:
            repo_state = repos_state.setdefault(
                repo, {"docs": {}, "readme_sha": None}
            )
            yield from self._fetch_repo_docs(repo, repo_state)
            yield from self._fetch_repo_issues(repo, since)
            yield from self._fetch_repo_pulls(repo, since)

    def _fetch_repo_docs(self, repo: str, repo_state: dict):
        # README via its own endpoint (works whatever the filename is);
        # the response carries the base64 content inline.
        readme = self._get(f"/repos/{repo}/readme").json()
        readme_path = readme.get("path", "README.md")
        if readme.get("sha") != repo_state.get("readme_sha"):
            text = base64.b64decode(readme["content"]).decode("utf-8")
            repo_state["readme_sha"] = readme.get("sha")
            doc_id = f"github:{repo}/{readme_path}"
            self.present_doc_ids.add(doc_id)
            yield ConnectorDoc(
                title=readme_path,
                text=text,
                source_label=doc_id,
                url=readme.get("html_url"),
                doc_id=doc_id,
            )
        else:
            self.present_doc_ids.add(f"github:{repo}/{readme_path}")

        tree = self._get(
            f"/repos/{repo}/git/trees/{self.branch}?recursive=1"
        ).json()
        if tree.get("truncated"):
            raise RuntimeError(
                f"git tree for {repo}@{self.branch} was truncated — "
                "use a smaller repo or narrow the scope"
            )
        docs_state: dict = repo_state.setdefault("docs", {})
        for node in tree.get("tree", []):
            if node.get("type") != "blob":
                continue
            path = node.get("path", "")
            if not path.startswith("docs/") or not path.endswith(".md"):
                continue
            sha = node.get("sha", "")
            doc_id = f"github:{repo}/{path}"
            self.present_doc_ids.add(doc_id)
            if docs_state.get(path) == sha:
                continue  # unchanged since last sync
            text = self._blob_text(repo, sha)
            docs_state[path] = sha
            yield ConnectorDoc(
                title=path,
                text=text,
                source_label=doc_id,
                url=f"https://github.com/{repo}/blob/{self.branch}/{path}",
                doc_id=doc_id,
            )

    def _fetch_repo_issues(self, repo: str, since: datetime | None):
        params: dict = {"state": "all", "sort": "updated", "direction": "desc"}
        if since is not None:
            params["since"] = _since_param(since)
        for item in self._get_paged(f"/repos/{repo}/issues", params):
            if "pull_request" in item:
                continue  # the issues endpoint also returns PRs
            number = item["number"]
            title = item.get("title", f"Issue #{number}")
            body = item.get("body") or ""
            text = f"{title}\n\n{body}\n\n(State: {item.get('state')}, updated {item.get('updated_at')})"
            yield ConnectorDoc(
                title=title,
                text=text,
                source_label=f"github:{repo}#{number}",
                updated_at=item.get("updated_at"),
                url=item.get("html_url"),
                doc_id=f"github:{repo}#issue-{number}",
            )

    def _fetch_repo_pulls(self, repo: str, since: datetime | None):
        # The pulls endpoint has no `since` parameter: list newest first
        # and filter client-side by updated_at.
        since_iso = _since_param(since) if since is not None else None
        params: dict = {"state": "all", "sort": "updated", "direction": "desc"}
        for pr in self._get_paged(f"/repos/{repo}/pulls", params):
            updated = pr.get("updated_at") or ""
            if since_iso is not None and updated <= since_iso:
                continue
            number = pr["number"]
            title = pr.get("title", f"PR #{number}")
            body = pr.get("body") or ""
            text = f"{title}\n\n{body}\n\n(State: {pr.get('state')}, updated {updated})"
            yield ConnectorDoc(
                title=title,
                text=text,
                source_label=f"github:{repo}#{number}",
                updated_at=updated,
                url=pr.get("html_url"),
                doc_id=f"github:{repo}#pr-{number}",
            )
