import type { HealthResponse } from "../types";

interface Props {
  health: HealthResponse | null;
  loading: boolean;
  error: string | null;
  onRetry: () => void;
}

export default function HealthIndicator({
  health,
  loading,
  error,
  onRetry,
}: Props) {
  if (loading) {
    return (
      <div className="health health-loading" role="status">
        <span className="dot dot-unknown" aria-hidden="true" />
        Checking backend…
      </div>
    );
  }
  if (error || !health) {
    return (
      <div className="health health-error" role="alert">
        <span className="dot dot-bad" aria-hidden="true" />
        Backend unreachable ({error ?? "unknown error"})
        <button type="button" className="link-button" onClick={onRetry}>
          Retry
        </button>
      </div>
    );
  }
  const ok = health.status === "ok";
  return (
    <div
      className={ok ? "health health-ok" : "health health-degraded"}
      role="status"
      title={`Ollama reachable: ${health.ollama_reachable ? "yes" : "no"}; index present: ${health.index_present ? "yes" : "no"}`}
    >
      <span
        className={ok ? "dot dot-good" : "dot dot-warn"}
        aria-hidden="true"
      />
      {ok ? "Backend healthy" : "Backend degraded"}
      <span className="health-detail">
        Ollama {health.ollama_reachable ? "✓" : "✗"} · Index{" "}
        {health.index_present ? "✓" : "✗"}
      </span>
    </div>
  );
}
