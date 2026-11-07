import { useState } from "react";
import ReactMarkdown from "react-markdown";
import { streamSummarizeDocument } from "../api";
import type { SummarizeEvent } from "../api";
import type { DocumentDetail as DocumentDetailType } from "../types";

interface Props {
  doc: DocumentDetailType;
  onBack: () => void;
  onChatAbout: (docId: string, filename: string) => void;
  onDelete: (id: string) => Promise<void>;
  onSummaryUpdated: (summary: string) => void;
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(ts: number): string {
  return new Date(ts * 1000).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function formatDuration(ms: number): string {
  const totalSeconds = Math.round(ms / 1000);
  if (totalSeconds < 60) return `${totalSeconds}s`;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  if (minutes < 60) return `${minutes}m ${seconds}s`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${minutes % 60}m`;
}

export default function DocumentDetail({
  doc,
  onBack,
  onChatAbout,
  onDelete,
  onSummaryUpdated,
}: Props) {
  const [summary, setSummary] = useState<string | null>(doc.summary);
  const [summarizing, setSummarizing] = useState(false);
  const [summarizeProgress, setSummarizeProgress] = useState<string | null>(null);
  // Percentage shown next to the spinner, map-reduce only. Null means no
  // percentage: single-call docs have no chunks to count, so they get the
  // spinner alone. The reduce stage keeps the last map percentage.
  const [summarizePercent, setSummarizePercent] = useState<number | null>(null);
  // Wall time of the last successful summarize (click to done event),
  // with the section count for map-reduce runs, so speed is visible in
  // the UI and comparable across changes. Client-side timing: it covers
  // the whole round trip, not just the server work.
  const [summarizeDurationMs, setSummarizeDurationMs] = useState<number | null>(null);
  const [summarizeSections, setSummarizeSections] = useState<number | null>(null);
  const [summarizeError, setSummarizeError] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);

  const handleSummarize = (): void => {
    if (summarizing) return;
    const startedAt = Date.now();
    setSummarizing(true);
    setSummarizeError(null);
    setSummarizeProgress("Starting…");
    setSummarizeDurationMs(null);
    setSummarizeSections(null);

    const finish = (): void => {
      setSummarizing(false);
      setSummarizeProgress(null);
      setSummarizePercent(null);
    };

    const onEvent = (event: SummarizeEvent): void => {
      if (event.type === "start") {
        // Single-call docs have no chunks to count: spinner only, no
        // percentage. Map-reduce starts at 0%.
        setSummarizePercent(event.mode === "map_reduce" ? 0 : null);
        setSummarizeSections(event.mode === "map_reduce" ? event.total : null);
        setSummarizeProgress(
          event.mode === "single"
            ? "Summarizing…"
            : `Summarizing 0/${event.total} sections…`,
        );
      } else if (event.type === "progress") {
        if (event.stage === "reduce") {
          // Keep the last map percentage while combining: claiming 100%
          // here would be wrong, the reduce call is still running.
          setSummarizeProgress("Combining section summaries…");
        } else {
          if (
            typeof event.done === "number" &&
            typeof event.total === "number" &&
            event.total > 0
          ) {
            setSummarizePercent(
              Math.round((event.done / event.total) * 100),
            );
          }
          setSummarizeProgress(
            `Summarizing section ${event.done}/${event.total}…`,
          );
        }
      } else if (event.type === "done") {
        setSummarizeDurationMs(Date.now() - startedAt);
        setSummary(event.summary);
        onSummaryUpdated(event.summary);
        finish();
      }
    };

    streamSummarizeDocument(doc.id, {
      onEvent,
      onError: (err) => {
        setSummarizeError(err.message);
        setSummarizeDurationMs(null);
        finish();
      },
    });
  };

  const handleDelete = async (): Promise<void> => {
    if (
      !window.confirm(
        `Delete "${doc.filename}" from the knowledge base? Its ${doc.chunk_count} chunks will be removed from the index.`,
      )
    )
      return;
    setDeleting(true);
    try {
      await onDelete(doc.id);
    } finally {
      setDeleting(false);
    }
  };

  return (
    <div className="doc-detail">
      <button type="button" className="doc-detail-back" onClick={onBack}>
        ← All documents
      </button>
      <h2 className="doc-detail-title">{doc.filename}</h2>
      <dl className="doc-detail-meta">
        <div>
          <dt>Uploaded</dt>
          <dd>{formatDate(doc.uploaded_at)}</dd>
        </div>
        <div>
          <dt>Size</dt>
          <dd>{formatSize(doc.file_size)}</dd>
        </div>
        <div>
          <dt>Chunks</dt>
          <dd>{doc.chunk_count}</dd>
        </div>
        <div>
          <dt>Characters</dt>
          <dd>{doc.char_count.toLocaleString()}</dd>
        </div>
        <div>
          <dt>Extractor</dt>
          <dd>{doc.extractor}</dd>
        </div>
      </dl>

      <section className="doc-detail-section">
        <h3>Summary</h3>
        {summarizeError && (
          <div className="upload-card-summary-error" role="alert">
            {summarizeError}
          </div>
        )}
        {summary ? (
          <div className="doc-detail-summary">
            <ReactMarkdown>{summary}</ReactMarkdown>
          </div>
        ) : (
          <p className="doc-detail-hint">No summary yet.</p>
        )}
        <button
          type="button"
          className="upload-card-summarize"
          onClick={handleSummarize}
          disabled={summarizing}
        >
          {summarizing ? (
            <>
              <span className="summarize-spinner" aria-hidden="true" />
              {summarizeProgress ?? (summary ? "Regenerating…" : "Summarizing…")}
              {summarizePercent !== null && ` ${summarizePercent}%`}
            </>
          ) : summary ? (
            "↻ Regenerate summary"
          ) : (
            "📝 Summarize"
          )}
        </button>
        {summarizing && summarizePercent !== null && (
          <div
            className="summarize-progress-track"
            role="progressbar"
            aria-valuenow={summarizePercent}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label="Summarize progress"
          >
            <div
              className="summarize-progress-fill"
              style={{ width: `${summarizePercent}%` }}
            />
          </div>
        )}
        {summarizeDurationMs !== null && (
          <p className="doc-detail-hint">
            Summarized in {formatDuration(summarizeDurationMs)}
            {summarizeSections !== null && ` · ${summarizeSections} sections`}
          </p>
        )}
      </section>

      <div className="doc-detail-actions">
        <button
          type="button"
          className="doc-detail-chat"
          onClick={() => onChatAbout(doc.id, doc.filename)}
        >
          💬 Chat about this document
        </button>
        <button
          type="button"
          className="doc-detail-delete"
          onClick={() => void handleDelete()}
          disabled={deleting}
        >
          {deleting ? "Deleting…" : "🗑 Delete"}
        </button>
      </div>
    </div>
  );
}
