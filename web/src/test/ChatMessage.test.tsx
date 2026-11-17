/**
 * Tests that ChatMessage renders the Sidekick avatar next to the answer.
 * (The user avatar lives in App.tsx next to the user turn; it is covered
 * by Avatar.test.tsx.)
 */
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import ChatMessage from "../components/ChatMessage";
import type { ChatResponse } from "../types";

afterEach(() => {
  cleanup();
});

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
