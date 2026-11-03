import { useRef } from "react";

interface Props {
  onFile: (file: File) => void;
  disabled: boolean;
  /** Upload progress 0–100 while a file is uploading; null otherwise. */
  progress: number | null;
}

const ACCEPT = ".pdf,.docx,.txt,.md,.csv";

export default function AttachButton({ onFile, disabled, progress }: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const uploading = progress !== null;

  const pick = (): void => {
    if (uploading) return;
    inputRef.current?.click();
  };

  const onChange = (e: React.ChangeEvent<HTMLInputElement>): void => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (file) onFile(file);
  };

  return (
    <>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT}
        onChange={onChange}
        hidden
        aria-hidden="true"
        tabIndex={-1}
      />
      <button
        type="button"
        className="attach-button"
        onClick={pick}
        disabled={disabled}
        aria-label={
          uploading
            ? `Uploading document… ${progress}%`
            : "Attach a document"
        }
        title={
          uploading
            ? `Uploading… ${progress}%`
            : "Attach a document (PDF, DOCX, TXT, MD, CSV)"
        }
      >
        <span aria-hidden="true">📎</span>
        {uploading && (
          <span
            className="attach-progress"
            style={{ width: `${progress}%` }}
            aria-hidden="true"
          />
        )}
      </button>
    </>
  );
}
