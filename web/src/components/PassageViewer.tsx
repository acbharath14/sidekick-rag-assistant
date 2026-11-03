import { useState } from "react";
import type { Hit } from "../types";

interface Props {
  hits: Hit[];
}

function formatScore(score: number): string {
  return score.toFixed(3);
}

export default function PassageViewer({ hits }: Props) {
  const [open, setOpen] = useState(false);
  if (hits.length === 0) return null;
  return (
    <div className="passages">
      <button
        type="button"
        className="passages-toggle"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        {open ? "▾" : "▸"} Retrieved passages ({hits.length})
      </button>
      {open && (
        <ol className="passages-list">
          {hits.map((h, i) => (
            <li key={i} className="passage">
              <div className="passage-meta">
                <span className="passage-source">{h.source}</span>
                <span
                  className="passage-score"
                  title="Relevance score (higher is more relevant for rerank logits; lower is closer for dense L2)"
                >
                  score {formatScore(h.score)}
                </span>
              </div>
              <pre className="passage-text">{h.text}</pre>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
