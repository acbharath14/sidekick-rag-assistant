import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import type { ChatResponse } from "../types";
import FeedbackButtons from "./FeedbackButtons";
import PassageViewer from "./PassageViewer";

interface Props {
  question: string;
  response: ChatResponse;
  createdAt?: number;
  streamingText?: string;
  isStreaming?: boolean;
}

export function formatTime(ts: number): string {
  return new Date(ts).toLocaleTimeString([], {
    hour: "numeric",
    minute: "2-digit",
  });
}

function formatLatency(ms: number | null): string | null {
  if (ms === null || ms === undefined) return null;
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(1)} s`;
}

function copyToClipboard(text: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    return navigator.clipboard.writeText(text);
  }
  // Fallback for older browsers.
  return new Promise<void>((resolve, reject) => {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    try {
      if (document.execCommand("copy")) {
        resolve();
      } else {
        reject(new Error("copy failed"));
      }
    } catch (err) {
      reject(err);
    } finally {
      document.body.removeChild(ta);
    }
  });
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<number | null>(null);

  useEffect(
    () => () => {
      if (timer.current !== null) window.clearTimeout(timer.current);
    },
    [],
  );

  const onClick = (): void => {
    void copyToClipboard(text)
      .then(() => {
        setCopied(true);
        if (timer.current !== null) window.clearTimeout(timer.current);
        timer.current = window.setTimeout(() => setCopied(false), 2000);
      })
      .catch(() => {
        // Clipboard unavailable — leave the button as-is.
      });
  };

  return (
    <button
      type="button"
      className="copy-btn"
      onClick={onClick}
      title="Copy response"
      aria-label="Copy response"
    >
      {copied ? "✓ Copied!" : "⧉ Copy"}
    </button>
  );
}

export default function ChatMessage({
  question,
  response,
  createdAt,
  streamingText,
  isStreaming = false,
}: Props) {
  const text = isStreaming && streamingText !== undefined ? streamingText : response.text;
  const latency = formatLatency(response.latency_ms);

  return (
    <article className="message message-assistant">
      {!response.grounded && (
        <span
          className="fallback-badge"
          title="The corpus had no confident answer; this came from the model's general knowledge"
        >
          General knowledge — not from the corpus
        </span>
      )}
      {response.standalone_question && (
        <div
          className="standalone-note"
          title="Your follow-up was rewritten against the conversation history before retrieval"
        >
          Retrieved as: “{response.standalone_question}”
        </div>
      )}
      <div className="message-body">
        {isStreaming && !text ? (
          <span className="thinking" aria-label="Model is thinking">
            <span className="thinking-dot" />
            <span className="thinking-dot" />
            <span className="thinking-dot" />
          </span>
        ) : (
          <ReactMarkdown>{text}</ReactMarkdown>
        )}
        {isStreaming && !!text && <span className="caret" aria-hidden="true" />}
      </div>
      {response.sources.length > 0 && (
        <div className="citations" aria-label="Cited sources">
          {response.sources.map((s) => (
            <span key={s} className="citation-pill" title={s}>
              {s}
            </span>
          ))}
        </div>
      )}
      <PassageViewer hits={response.hits} />
      <div className="message-footer">
        {latency && (
          <span className="latency" title="End-to-end answer time">
            {latency}
          </span>
        )}
        {!isStreaming && <CopyButton text={response.text} />}
        {!isStreaming && <FeedbackButtons question={question} />}
        {createdAt !== undefined && (
          <span className="message-time">{formatTime(createdAt)}</span>
        )}
      </div>
    </article>
  );
}
