import ReactMarkdown from "react-markdown";
import type { UploadResponse } from "../types";

interface Props {
  upload: UploadResponse;
  onChat: () => void;
  onDismiss: () => void;
}

export default function UploadCard({ upload, onChat, onDismiss }: Props) {
  return (
    <section className="upload-card" aria-label="Uploaded document">
      <div className="upload-card-header">
        <span className="upload-card-icon" aria-hidden="true">
          📄
        </span>
        <div className="upload-card-meta">
          <strong className="upload-card-name">{upload.filename}</strong>
          <span className="upload-card-stats">
            {upload.char_count.toLocaleString()} characters ·{" "}
            {upload.chunk_count} chunks indexed
          </span>
        </div>
        <button
          type="button"
          className="upload-card-dismiss"
          onClick={onDismiss}
          aria-label="Dismiss uploaded document"
          title="Dismiss"
        >
          ✕
        </button>
      </div>
      <div className="upload-card-summary">
        <ReactMarkdown>{upload.summary}</ReactMarkdown>
      </div>
      <div className="upload-card-actions">
        <button
          type="button"
          className="upload-card-chat"
          onClick={onChat}
        >
          💬 Chat with this file
        </button>
      </div>
    </section>
  );
}
