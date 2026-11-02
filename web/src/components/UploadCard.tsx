import { useState } from "react";
import ReactMarkdown from "react-markdown";
import { summarizeUpload } from "../api";
import type { UploadResponse } from "../types";

interface Props {
  upload: UploadResponse;
  onChat: () => void;
  onDismiss: () => void;
}

export default function UploadCard({ upload, onChat, onDismiss }: Props) {
  const [summary, setSummary] = useState<string | null>(null);
  const [summarizing, setSummarizing] = useState(false);
  const [summarizeError, setSummarizeError] = useState<string | null>(null);

  const handleSummarize = async (): Promise<void> => {
    if (summarizing) return;
    setSummarizing(true);
    setSummarizeError(null);
    try {
      setSummary(await summarizeUpload(upload.upload_id));
    } catch (err) {
      setSummarizeError((err as Error).message);
    } finally {
      setSummarizing(false);
    }
  };

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
      {summary && (
        <div className="upload-card-summary">
          <ReactMarkdown>{summary}</ReactMarkdown>
        </div>
      )}
      {summarizeError && (
        <div className="upload-card-summary-error" role="alert">
          {summarizeError}
        </div>
      )}
      <div className="upload-card-actions">
        <button
          type="button"
          className="upload-card-summarize"
          onClick={() => void handleSummarize()}
          disabled={summarizing}
        >
          {summarizing ? (
            <>
              <span className="summarize-spinner" aria-hidden="true" />
              Summarizing…
            </>
          ) : (
            "📝 Summarize"
          )}
        </button>
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
