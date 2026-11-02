import ReactMarkdown from "react-markdown";
import type { ChatResponse } from "../types";
import FeedbackButtons from "./FeedbackButtons";
import PassageViewer from "./PassageViewer";

interface Props {
  question: string;
  response: ChatResponse;
  streamingText?: string;
  isStreaming?: boolean;
}

function formatLatency(ms: number | null): string | null {
  if (ms === null || ms === undefined) return null;
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(1)} s`;
}

export default function ChatMessage({
  question,
  response,
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
        {!isStreaming && <FeedbackButtons question={question} />}
      </div>
    </article>
  );
}
