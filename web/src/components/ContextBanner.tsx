import type { ConfigResponse } from "../types";

interface Props {
  config: ConfigResponse | null;
  loading: boolean;
  error: string | null;
}

export default function ContextBanner({ config, loading, error }: Props) {
  if (loading) {
    return (
      <div className="context-banner" role="status">
        <span className="banner-label">Context</span>
        <span className="banner-loading">Loading configuration…</span>
      </div>
    );
  }
  if (error || !config) {
    return (
      <div className="context-banner banner-error" role="alert">
        <span className="banner-label">Context</span>
        <span>Could not load backend config: {error ?? "unknown error"}</span>
      </div>
    );
  }
  return (
    <div className="context-banner" role="status" aria-label="Answer context">
      <span className="banner-label">Answering from</span>
      <span className="banner-item" title="LLM model">
        Model <strong>{config.ollama_model}</strong>
      </span>
      <span className="banner-item" title="Retrieval mode">
        Retrieval <strong>{config.retrieval}</strong>
      </span>
      <span className="banner-item" title="Embedding model">
        Embeddings <strong>{config.embedding_model}</strong>
      </span>
      {config.rerank && (
        <span className="banner-item banner-chip" title="Cross-encoder reranking is on">
          Rerank on
        </span>
      )}
      {config.hybrid_fallback && (
        <span
          className="banner-item banner-chip"
          title="Falls back to general knowledge when the corpus has no answer"
        >
          Hybrid fallback on
        </span>
      )}
    </div>
  );
}
