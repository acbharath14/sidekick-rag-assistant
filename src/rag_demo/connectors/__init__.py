"""Enterprise connectors for Sidekick (mock-first, env-only tokens).

Available sources register in ``sync.REGISTRY``. Each source is a small
client implementing the :class:`Connector` interface in ``base.py``.
"""

from .base import Connector, ConnectorDoc, to_documents

__all__ = ["Connector", "ConnectorDoc", "to_documents"]
