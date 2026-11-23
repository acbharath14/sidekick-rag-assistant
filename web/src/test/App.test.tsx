/**
 * Tests for the Agentic toggle in App.tsx: it flips the mode, persists
 * the choice, and hands the flag to streamChat on send. The api module
 * is mocked wholesale so no network or backend is involved.
 */
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "../App";

vi.mock("../api", () => ({
  clearConversationMessages: vi.fn(),
  createConversation: vi.fn().mockResolvedValue({ id: "c1", title: "New chat" }),
  deleteConversation: vi.fn(),
  deleteDocument: vi.fn(),
  fetchConfig: vi.fn().mockResolvedValue({
    ollama_model: "qwen2.5-coder:7b",
    embedding_model: "all-MiniLM-L6-v2",
    retrieval: "hybrid",
    rerank: false,
    hybrid_fallback: true,
  }),
  fetchHealth: vi.fn().mockResolvedValue({
    status: "ok",
    ollama_reachable: true,
    index_present: true,
  }),
  getConversation: vi.fn().mockResolvedValue({ messages: [] }),
  getDocument: vi.fn(),
  listConversations: vi.fn().mockResolvedValue({ conversations: [] }),
  listDocuments: vi.fn().mockResolvedValue({ documents: [] }),
  renameConversation: vi.fn(),
  streamChat: vi.fn(),
  uploadFile: vi.fn(),
}));

import { streamChat } from "../api";

const mockedStreamChat = vi.mocked(streamChat);

afterEach(cleanup);

beforeEach(() => {
  localStorage.clear();
  vi.clearAllMocks();
  // jsdom has no scrollIntoView; App scrolls to the newest turn on change.
  Element.prototype.scrollIntoView = vi.fn();
});

async function renderApp() {
  render(<App />);
  return screen.findByRole("button", { name: "Agentic" });
}

describe("Agentic toggle", () => {
  it("starts off and flips on click, persisting the choice", async () => {
    const user = userEvent.setup();
    const toggle = await renderApp();
    expect(toggle).toHaveAttribute("aria-pressed", "false");

    await user.click(toggle);
    expect(toggle).toHaveAttribute("aria-pressed", "true");
    expect(localStorage.getItem("sidekick.agenticMode")).toBe("1");

    await user.click(toggle);
    expect(toggle).toHaveAttribute("aria-pressed", "false");
    expect(localStorage.getItem("sidekick.agenticMode")).toBe("0");
  });

  it("starts on when the stored preference is set", async () => {
    localStorage.setItem("sidekick.agenticMode", "1");
    const toggle = await renderApp();
    expect(toggle).toHaveAttribute("aria-pressed", "true");
  });

  it("passes agentic:true to streamChat when the toggle is on", async () => {
    const user = userEvent.setup();
    const toggle = await renderApp();
    await user.click(toggle);

    const input = screen.getByLabelText("Your question");
    await user.type(input, "What is this?");
    await user.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => expect(mockedStreamChat).toHaveBeenCalled());
    const options = mockedStreamChat.mock.calls[0][5] as
      | { agentic?: boolean }
      | undefined;
    expect(options?.agentic).toBe(true);
  });

  it("passes agentic:false to streamChat when the toggle is off", async () => {
    const user = userEvent.setup();
    await renderApp();

    const input = screen.getByLabelText("Your question");
    await user.type(input, "What is this?");
    await user.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => expect(mockedStreamChat).toHaveBeenCalled());
    const options = mockedStreamChat.mock.calls[0][5] as
      | { agentic?: boolean }
      | undefined;
    expect(options?.agentic).toBe(false);
  });
});
