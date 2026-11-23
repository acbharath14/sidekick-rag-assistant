import { useCallback, useEffect, useRef, useState } from "react";

import {
  fetchConnectors,
  fetchConnectorJob,
  startConnectorSync,
} from "../api";
import type { ConnectorStatus, ConnectorSyncJob } from "../types";

function formatSyncTime(iso: string | null): string {
  if (!iso) return "Never synced";
  return new Date(iso).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

export default function ConnectorsView() {
  const [connectors, setConnectors] = useState<ConnectorStatus[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tokens, setTokens] = useState<Record<string, string>>({});
  const [fullSync, setFullSync] = useState<Record<string, boolean>>({});
  const [jobs, setJobs] = useState<Record<string, ConnectorSyncJob>>({});
  const [syncErrors, setSyncErrors] = useState<Record<string, string>>({});

  const jobsRef = useRef(jobs);
  jobsRef.current = jobs;

  const loadConnectors = useCallback(async (): Promise<void> => {
    try {
      const data = await fetchConnectors();
      setConnectors(data.connectors);
      setError(null);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadConnectors();
  }, [loadConnectors]);

  // Poll any running sync jobs until they finish, then refresh the list.
  useEffect(() => {
    if (!Object.values(jobs).some((j) => j.status === "running")) return;
    const id = window.setInterval(() => {
      void (async () => {
        for (const job of Object.values(jobsRef.current)) {
          if (job.status !== "running") continue;
          try {
            const next = await fetchConnectorJob(job.job_id);
            setJobs((prev) => ({ ...prev, [job.source]: next }));
            if (next.status !== "running") void loadConnectors();
          } catch {
            // Transient failure: the next tick retries.
          }
        }
      })();
    }, 1000);
    return () => window.clearInterval(id);
  }, [jobs, loadConnectors]);

  const handleSync = async (source: string): Promise<void> => {
    const token = (tokens[source] ?? "").trim() || null;
    setSyncErrors((prev) => ({ ...prev, [source]: "" }));
    try {
      const job = await startConnectorSync(source, {
        token,
        full: !!fullSync[source],
      });
      setJobs((prev) => ({ ...prev, [source]: job }));
      // The pasted token was used once; clear it from the form.
      setTokens((prev) => ({ ...prev, [source]: "" }));
    } catch (err) {
      setSyncErrors((prev) => ({ ...prev, [source]: (err as Error).message }));
    }
  };

  if (loading) {
    return (
      <div className="cabinet">
        <div className="empty-state">
          <p>Loading connectors…</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="cabinet">
        <div className="empty-state">
          <p>Couldn’t load connectors: {error}</p>
          <button type="button" onClick={() => void loadConnectors()}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="cabinet">
      <div className="cabinet-header">
        <div>
          <h2>Connectors</h2>
          <p className="cabinet-subtitle">
            Index Confluence, Outlook, and GitHub into the knowledge base.
            A token pasted here is used once for that sync and never stored.
          </p>
        </div>
        <button
          type="button"
          className="cabinet-upload-btn"
          onClick={() => void loadConnectors()}
        >
          Refresh
        </button>
      </div>
      <div className="connector-grid">
        {(connectors ?? []).map((c) => {
          const job = jobs[c.source];
          const running = job?.status === "running";
          const syncError = syncErrors[c.source];
          return (
            <article key={c.source} className="connector-card">
              <div className="connector-card-header">
                <h3>{c.label}</h3>
                <span
                  className={
                    "connector-badge" +
                    (c.configured ? " connector-badge-ok" : "")
                  }
                >
                  {c.configured ? "Configured" : "Not configured"}
                </span>
              </div>
              <p className="connector-auth">{c.auth}</p>
              {!c.configured && (
                <>
                  <p className="connector-hint">{c.setup_hint}</p>
                  <p className="connector-missing">
                    Missing: {c.missing_env.join(", ")}
                  </p>
                </>
              )}
              <p className="connector-stats">
                Last sync: {formatSyncTime(c.last_sync)} · {c.total_docs}{" "}
                {c.total_docs === 1 ? "document" : "documents"}
              </p>
              {!c.configured && (
                <label className="connector-token">
                  Token
                  <input
                    type="password"
                    value={tokens[c.source] ?? ""}
                    onChange={(e) =>
                      setTokens((prev) => ({
                        ...prev,
                        [c.source]: e.target.value,
                      }))
                    }
                    placeholder={`Paste a ${c.label} token`}
                    aria-label={`${c.label} token`}
                    autoComplete="off"
                  />
                </label>
              )}
              <div className="connector-actions">
                <button
                  type="button"
                  className="cabinet-upload-btn"
                  disabled={running}
                  onClick={() => void handleSync(c.source)}
                >
                  {running ? "Syncing…" : "Sync now"}
                </button>
                <label className="connector-full">
                  <input
                    type="checkbox"
                    checked={!!fullSync[c.source]}
                    onChange={(e) =>
                      setFullSync((prev) => ({
                        ...prev,
                        [c.source]: e.target.checked,
                      }))
                    }
                  />
                  Full sync
                </label>
              </div>
              {job?.status === "done" && job.summary && (
                <p className="connector-result" role="status">
                  Synced: {job.summary.fetched} new/changed,{" "}
                  {job.summary.total} total.
                </p>
              )}
              {(job?.status === "error" || syncError) && (
                <p className="connector-error" role="alert">
                  {job?.status === "error" ? job.error : syncError}
                </p>
              )}
            </article>
          );
        })}
      </div>
    </div>
  );
}
