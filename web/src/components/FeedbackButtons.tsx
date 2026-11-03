import { useState } from "react";
import { postFeedback } from "../api";

interface Props {
  question: string;
}

type VoteState = "idle" | "up" | "down" | "sending" | "error";

export default function FeedbackButtons({ question }: Props) {
  const [state, setState] = useState<VoteState>("idle");

  const vote = async (v: "up" | "down"): Promise<void> => {
    if (state === "sending" || state === v) return;
    setState("sending");
    try {
      const res = await postFeedback(question, v);
      setState(res.ok ? v : "error");
    } catch {
      setState("error");
    }
  };

  return (
    <div
      className="feedback"
      role="group"
      aria-label="Rate this answer"
    >
      <button
        type="button"
        className={state === "up" ? "feedback-btn voted" : "feedback-btn"}
        aria-pressed={state === "up"}
        aria-label="Thumbs up"
        title="This answer was helpful"
        disabled={state === "sending"}
        onClick={() => void vote("up")}
      >
        👍
      </button>
      <button
        type="button"
        className={state === "down" ? "feedback-btn voted" : "feedback-btn"}
        aria-pressed={state === "down"}
        aria-label="Thumbs down"
        title="This answer was not helpful"
        disabled={state === "sending"}
        onClick={() => void vote("down")}
      >
        👎
      </button>
      {state === "error" && (
        <span className="feedback-error" role="alert">
          Could not save feedback
        </span>
      )}
    </div>
  );
}
