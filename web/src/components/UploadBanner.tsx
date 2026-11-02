interface Props {
  filename: string;
  onClear: () => void;
}

export default function UploadBanner({ filename, onClear }: Props) {
  return (
    <div className="upload-banner" role="status">
      <span aria-hidden="true">📄</span>
      <span>
        Chatting with: <strong>{filename}</strong>
      </span>
      <button
        type="button"
        className="upload-banner-clear"
        onClick={onClear}
        aria-label="Stop chatting with this file"
        title="Back to the main corpus"
      >
        ✕
      </button>
    </div>
  );
}
