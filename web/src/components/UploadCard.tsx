import type { UploadResponse } from "../types";

interface Props {
  upload: UploadResponse;
  onDismiss: () => void;
  onViewDocuments: () => void;
}

export default function UploadCard({ upload, onDismiss, onViewDocuments }: Props) {
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
      <div className="upload-card-actions">
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
