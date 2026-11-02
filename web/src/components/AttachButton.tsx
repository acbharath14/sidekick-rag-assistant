import { useRef } from "react";

interface Props {
  onFile: (file: File) => void;
  disabled: boolean;
  uploading: boolean;
}

const ACCEPT = ".pdf,.docx,.txt,.md,.csv";

export default function AttachButton({ onFile, disabled, uploading }: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null);

  const pick = (): void => {
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
        disabled={disabled || uploading}
        aria-label="Attach a document"
        title="Attach a document (PDF, DOCX, TXT, MD, CSV)"
      >
        {uploading ? (
          <span className="attach-spinner" aria-hidden="true" />
        ) : (
          <span aria-hidden="true">📎</span>
        )}
      </button>
    </>
  );
}
