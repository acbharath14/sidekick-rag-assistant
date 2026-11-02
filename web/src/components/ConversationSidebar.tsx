import { useState } from "react";
import type { Conversation } from "../types";

interface Props {
  conversations: Conversation[];
  activeId: string | null;
  collapsed: boolean;
  onToggle: () => void;
  onSelect: (id: string) => void;
  onNew: () => void;
  onRename: (id: string, title: string) => Promise<void>;
  onDelete: (id: string) => Promise<void>;
}

function relativeTime(ts: number): string {
  const diff = Date.now() / 1000 - ts;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  const d = Math.floor(diff / 86400);
  return d === 1 ? "yesterday" : `${d}d ago`;
}

export default function ConversationSidebar({
  conversations,
  activeId,
  collapsed,
  onToggle,
  onSelect,
  onNew,
  onRename,
  onDelete,
}: Props) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");
  const [busyId, setBusyId] = useState<string | null>(null);

  const startRename = (c: Conversation): void => {
    setEditingId(c.id);
    setEditTitle(c.title);
  };

  const commitRename = async (id: string): Promise<void> => {
    const title = editTitle.trim();
    setEditingId(null);
    if (!title) return;
    setBusyId(id);
    try {
      await onRename(id, title);
    } finally {
      setBusyId(null);
    }
  };

  const handleDelete = async (c: Conversation): Promise<void> => {
    if (!window.confirm(`Delete "${c.title}"? This cannot be undone.`)) return;
    setBusyId(c.id);
    try {
      await onDelete(c.id);
    } finally {
      setBusyId(null);
    }
  };

  if (collapsed) {
    return (
      <aside className="sidebar sidebar-collapsed" aria-label="Conversations">
        <button
          type="button"
          className="sidebar-toggle"
          onClick={onToggle}
          aria-label="Show conversations"
          title="Show conversations"
        >
          ☰
        </button>
      </aside>
    );
  }

  return (
    <aside className="sidebar" aria-label="Conversations">
      <div className="sidebar-header">
        <button
          type="button"
          className="sidebar-toggle"
          onClick={onToggle}
          aria-label="Hide conversations"
          title="Hide conversations"
        >
          ☰
        </button>
        <button
          type="button"
          className="sidebar-new"
          onClick={onNew}
          title="Start a new chat"
        >
          ＋ New chat
        </button>
      </div>
      <ul className="sidebar-list">
        {conversations.map((c) => (
          <li
            key={c.id}
            className={
              "sidebar-item" + (c.id === activeId ? " sidebar-item-active" : "")
            }
          >
            {editingId === c.id ? (
              <input
                className="sidebar-rename-input"
                value={editTitle}
                autoFocus
                onChange={(e) => setEditTitle(e.target.value)}
                onBlur={() => void commitRename(c.id)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") void commitRename(c.id);
                  if (e.key === "Escape") setEditingId(null);
                }}
                aria-label="Rename conversation"
              />
            ) : (
              <button
                type="button"
                className="sidebar-item-btn"
                onClick={() => onSelect(c.id)}
                title={c.title}
                disabled={busyId === c.id}
              >
                <span className="sidebar-item-title">{c.title}</span>
                {c.last_message && (
                  <span className="sidebar-item-preview">
                    {c.last_message.length > 60
                      ? c.last_message.slice(0, 60).trimEnd() + "…"
                      : c.last_message}
                  </span>
                )}
                <span className="sidebar-item-meta">
                  {relativeTime(c.updated_at)}
                </span>
              </button>
            )}
            {editingId !== c.id && (
              <span className="sidebar-item-actions">
                <button
                  type="button"
                  className="sidebar-action"
                  onClick={() => startRename(c)}
                  aria-label={`Rename ${c.title}`}
                  title="Rename"
                  disabled={busyId === c.id}
                >
                  ✎
                </button>
                <button
                  type="button"
                  className="sidebar-action sidebar-action-danger"
                  onClick={() => void handleDelete(c)}
                  aria-label={`Delete ${c.title}`}
                  title="Delete"
                  disabled={busyId === c.id}
                >
                  🗑
                </button>
              </span>
            )}
          </li>
        ))}
        {conversations.length === 0 && (
          <li className="sidebar-empty">No conversations yet.</li>
        )}
      </ul>
    </aside>
  );
}
