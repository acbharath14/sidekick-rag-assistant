import { useState } from "react";
import type { DocumentInfo } from "../types";

interface Props {
  documents: DocumentInfo[];
  loading: boolean;
  error: string | null;
  onSelect: (id: string) => void;
  onDelete: (id: string) => Promise<void>;
  onChatAbout: (id: string, filename: string) => void;
  onUpload: (file: File) => void;
  uploading: boolean;
  uploadProgress: number | null;
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(ts: number): string {
  return new Date(ts * 1000).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

export default function DocumentsView({
  documents,
  loading,
  error,
  onSelect,
  onDelete,
  onChatAbout,
  onUpload,
  uploading,
  uploadProgress,
}: Props) {
  const [query, setQuery] = useState("");
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const filtered = documents.filter((d) =>
    d.filename.toLowerCase().includes(query.toLowerCase()),
  );

  const handleDelete = async (d: DocumentInfo): Promise<void> => {
    if (
      !window.confirm(
        `Delete "${d.filename}" from the knowledge base? Its ${d.chunk_count} chunks will be removed from the index.`,
      )
    )
      return;
    setDeletingId(d.id);
    try {
      await onDelete(d.id);
    } finally {
      setDeletingId(null);
    }
  };

  const pickFile = (e: React.ChangeEvent<HTMLInputElement>): void => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (file) onUpload(file);
  };

  return (
    <div className="cabinet">
      <div className="cabinet-header">
        <div className="cabinet-search">
          <input
            type="search"
            placeholder="Search documents…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            aria-label="Search documents"
          />
        </div>
        <label className="cabinet-upload-btn">
          <input
            type="file"
            accept=".pdf,.docx,.txt,.md,.csv"
            onChange={pickFile}
            hidden
            disabled={uploading}
          />
          {uploadProgress !== null
            ? `Uploading… ${uploadProgress}%`
            : "＋ Upload document"}
        </label>
      </div>
      {error && (
        <div className="upload-error" role="alert">
          {error}
        </div>
      )}
      {loading ? (
        <div className="empty-state">
          <p>Loading documents…</p>
        </div>
      ) : filtered.length === 0 ? (
        <div className="empty-state">
          <p>
            {documents.length === 0
              ? "No documents yet. Upload one to add it to the knowledge base."
              : "No documents match your search."}
          </p>
        </div>
      ) : (
        <ul className="cabinet-grid">
          {filtered.map((d) => (
            <li key={d.id} className="doc-card">
              <button
                type="button"
                className="doc-card-main"
                onClick={() => onSelect(d.id)}
                title={`Open ${d.filename}`}
              >
                <span className="doc-card-icon" aria-hidden="true">
                  📄
                </span>
                <span className="doc-card-name">{d.filename}</span>
                <span className="doc-card-meta">
                  {d.chunk_count} chunks · {formatSize(d.file_size)} ·{" "}
                  {formatDate(d.uploaded_at)}
                </span>
                <span className="doc-card-meta">
                  via {d.extractor}
                  {d.has_summary ? " · summarized" : ""}
                </span>
              </button>
              <button
                type="button"
                className="doc-card-chat"
                onClick={() => onChatAbout(d.id, d.filename)}
                aria-label={`Chat about ${d.filename}`}
                title="Chat about this document"
              >
                💬
              </button>
              <button
                type="button"
                className="doc-card-delete"
                onClick={() => void handleDelete(d)}
                disabled={deletingId === d.id}
                aria-label={`Delete ${d.filename}`}
                title="Delete from knowledge base"
              >
                {deletingId === d.id ? "…" : "🗑"}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
