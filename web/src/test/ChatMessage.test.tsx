/**
 * Tests for the agent-plan block in components/ChatMessage.tsx.
 */
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import ChatMessage from "../components/ChatMessage";
import type { ChatResponse } from "../types";

afterEach(cleanup);

const RESPONSE: ChatResponse = {
  text: "The answer.",
  sources: ["doc.md"],
  hits: [],
  grounded: true,
  latency_ms: 120,
  standalone_question: null,
};

const PROGRESS = [
  { phase: "thought", message: "Planning search strategy..." },
  { phase: "action", message: "Searching [search_docs]: parcels" },
];

describe("ChatMessage agent plan", () => {
  it("renders the plan steps when agentProgress is provided", () => {
    render(
      <ChatMessage
        question="q"
        response={RESPONSE}
        agentProgress={PROGRESS}
      />,
    );
    expect(screen.getByText(/Agent plan · 2 steps/)).toBeInTheDocument();
    expect(screen.getByText("Planning search strategy...")).toBeInTheDocument();
    expect(
      screen.getByText("Searching [search_docs]: parcels"),
    ).toBeInTheDocument();
    expect(screen.getByText("thought")).toBeInTheDocument();
    expect(screen.getByText("The answer.")).toBeInTheDocument();
  });

  it("renders no plan block without agentProgress", () => {
    render(<ChatMessage question="q" response={RESPONSE} />);
    expect(screen.queryByText(/Agent plan/)).not.toBeInTheDocument();
  });

  it("renders no plan block for an empty progress list", () => {
    render(
      <ChatMessage question="q" response={RESPONSE} agentProgress={[]} />,
    );
    expect(screen.queryByText(/Agent plan/)).not.toBeInTheDocument();
  });

  it("keeps the plan open while streaming", () => {
    const { container } = render(
      <ChatMessage
        question="q"
        response={RESPONSE}
        agentProgress={PROGRESS}
        isStreaming
        streamingText=""
      />,
    );
    expect(container.querySelector("details.agent-plan")).toHaveAttribute(
      "open",
    );
    expect(screen.getByText(/running…/)).toBeInTheDocument();
  });

  it("collapses the plan once the answer is done", () => {
    const { container } = render(
      <ChatMessage
        question="q"
        response={RESPONSE}
        agentProgress={PROGRESS}
      />,
    );
    expect(container.querySelector("details.agent-plan")).not.toHaveAttribute(
      "open",
    );
  });
});
