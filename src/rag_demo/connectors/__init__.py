"""Enterprise connectors (mock-first): Confluence, Outlook, and GitHub.

See docs/connectors.md. Credentials come from env vars only; tests use
literal fake values against the local mock servers in tests/mock_servers.py.
"""

from .base import Connector, ConnectorDoc, to_documents
from .confluence import ConfluenceConnector
from .github import GitHubConnector
from .outlook import OutlookConnector
from .sync import CONNECTORS, SOURCE_INFO, index_dir_for, is_configured, read_source_status, sync_source

__all__ = [
    "Connector",
    "ConnectorDoc",
    "to_documents",
    "ConfluenceConnector",
    "OutlookConnector",
    "GitHubConnector",
    "CONNECTORS",
    "SOURCE_INFO",
    "index_dir_for",
    "is_configured",
    "read_source_status",
    "sync_source",
]
