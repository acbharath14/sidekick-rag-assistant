/**
 * Tests for components/ChatMessage.tsx: the Sidekick avatar next to the
 * answer, and the agent-plan block shown while the ReAct agent works.
 * (The user avatar lives in App.tsx next to the user turn; it is covered
 * by Avatar.test.tsx.)
 */
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import ChatMessage from "../components/ChatMessage";
import type { ChatResponse } from "../types";

afterEach(cleanup);

const groundedResponse: ChatResponse = {
  text: "Tokens expire after 90 days.",
  sources: ["api-reference.md"],
  hits: [
    {
      source: "api-reference.md",
      text: "Tokens expire after 90 days.",
      score: 0.9,
    },
  ],
  grounded: true,
  latency_ms: 1234,
  standalone_question: null,
};

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

describe("ChatMessage avatar", () => {
  it("shows the Sidekick avatar next to a finished answer", () => {
    render(
      <ChatMessage
        question="When do tokens expire?"
        response={groundedResponse}
        createdAt={1763240000000}
      />,
    );
    expect(screen.getByRole("img", { name: "Sidekick" })).toBeInTheDocument();
    expect(screen.getByText("Tokens expire after 90 days.")).toBeInTheDocument();
    expect(screen.queryByRole("img", { name: "You" })).toBeNull();
  });

  it("shows the Sidekick avatar while streaming", () => {
    render(
      <ChatMessage
        question="When do tokens expire?"
        response={groundedResponse}
        streamingText="Tokens expire after"
        isStreaming
      />,
    );
    expect(screen.getByRole("img", { name: "Sidekick" })).toBeInTheDocument();
    expect(screen.getByText("Tokens expire after")).toBeInTheDocument();
  });

  it("shows the Sidekick avatar while thinking, before any text arrives", () => {
    render(
      <ChatMessage
        question="When do tokens expire?"
        response={groundedResponse}
        streamingText=""
        isStreaming
      />,
    );
    expect(screen.getByRole("img", { name: "Sidekick" })).toBeInTheDocument();
    expect(screen.getByLabelText("Model is thinking")).toBeInTheDocument();
  });
});

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
      <ChatMessage question="q" response={RESPONSE} agentProgress={PROGRESS} />,
    );
    expect(container.querySelector("details.agent-plan")).not.toHaveAttribute(
      "open",
    );
  });
});
