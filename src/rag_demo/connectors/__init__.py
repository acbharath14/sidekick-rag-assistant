"""Enterprise connectors (mock-first): Confluence and Outlook.

See docs/connectors.md. Credentials come from env vars only; tests use
literal fake values against the local mock servers in tests/mock_servers.py.
"""

from .base import Connector, ConnectorDoc, to_documents
from .confluence import ConfluenceConnector
from .outlook import OutlookConnector
from .sync import CONNECTORS, sync_source

__all__ = [
    "Connector",
    "ConnectorDoc",
    "to_documents",
    "ConfluenceConnector",
    "OutlookConnector",
    "CONNECTORS",
    "sync_source",
]
