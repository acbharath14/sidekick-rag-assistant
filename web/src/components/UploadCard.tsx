import { useState } from "react";
import ReactMarkdown from "react-markdown";
import { summarizeDocument } from "../api";
import type { UploadResponse } from "../types";

interface Props {
  upload: UploadResponse;
  onDismiss: () => void;
  onViewDocuments: () => void;
}

export default function UploadCard({ upload, onDismiss, onViewDocuments }: Props) {
  const [summary, setSummary] = useState<string | null>(null);
  const [summarizing, setSummarizing] = useState(false);
  const [summarizeError, setSummarizeError] = useState<string | null>(null);

  const handleSummarize = async (): Promise<void> => {
    if (summarizing) return;
    setSummarizing(true);
    setSummarizeError(null);
    try {
      setSummary(await summarizeDocument(upload.id));
    } catch (err) {
      setSummarizeError((err as Error).message);
    } finally {
      setSummarizing(false);
    }
  };

  return (
    <section className="upload-card" aria-label="Document added">
      <div className="upload-card-header">
        <span className="upload-card-icon" aria-hidden="true">
          ✅
        </span>
        <div className="upload-card-meta">
          <strong className="upload-card-name">{upload.filename}</strong>
          <span className="upload-card-stats">
            Added to knowledge base · {upload.chunk_count} chunks ·{" "}
            {upload.char_count.toLocaleString()} characters · via{" "}
            {upload.extractor}
          </span>
        </div>
        <button
          type="button"
          className="upload-card-dismiss"
          onClick={onDismiss}
          aria-label="Dismiss"
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
          onClick={onViewDocuments}
        >
          📁 Open document cabinet
        </button>
      </div>
    </section>
  );
}
