import { useState } from "react";
import ReactMarkdown from "react-markdown";
import { streamSummarizeDocument } from "../api";
import type { SummarizeEvent } from "../api";
import type { UploadResponse } from "../types";

interface Props {
  upload: UploadResponse;
  onDismiss: () => void;
  onViewDocuments: () => void;
  onChatAbout: (id: string, filename: string) => void;
}

export default function UploadCard({
  upload,
  onDismiss,
  onViewDocuments,
  onChatAbout,
}: Props) {
  const [summary, setSummary] = useState<string | null>(null);
  const [summarizing, setSummarizing] = useState(false);
  const [summarizeProgress, setSummarizeProgress] = useState<string | null>(null);
  const [summarizeError, setSummarizeError] = useState<string | null>(null);

  const handleSummarize = (): void => {
    if (summarizing) return;
    setSummarizing(true);
    setSummarizeError(null);
    setSummarizeProgress("Starting…");

    const finish = (): void => {
      setSummarizing(false);
      setSummarizeProgress(null);
    };

    const onEvent = (event: SummarizeEvent): void => {
      if (event.type === "start") {
        setSummarizeProgress(
          event.mode === "single"
            ? "Summarizing…"
            : `Summarizing 0/${event.total} sections…`,
        );
      } else if (event.type === "progress") {
        if (event.stage === "reduce") {
          setSummarizeProgress("Combining section summaries…");
        } else {
          setSummarizeProgress(
            `Summarizing section ${event.done}/${event.total}…`,
          );
        }
      } else if (event.type === "done") {
        setSummary(event.summary);
        finish();
      }
    };

    streamSummarizeDocument(upload.id, {
      onEvent,
      onError: (err) => {
        setSummarizeError(err.message);
        finish();
      },
    });
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
              {summarizeProgress ?? "Summarizing…"}
            </>
          ) : (
            "📝 Summarize"
          )}
        </button>
        <button
          type="button"
          className="upload-card-chat-about"
          onClick={() => onChatAbout(upload.id, upload.filename)}
        >
          💬 Chat about this document
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
