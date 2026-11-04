/**
 * Tests for components/UploadCard.tsx.
 */
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import UploadCard from "../components/UploadCard";
import type { SummarizeEvent } from "../api";
import type { UploadResponse } from "../types";

afterEach(() => cleanup());

vi.mock("../api", () => ({
  streamSummarizeDocument: vi.fn(),
}));

import { streamSummarizeDocument } from "../api";

const mockStream = streamSummarizeDocument as unknown as ReturnType<
  typeof vi.fn
>;

interface StreamCb {
  onEvent: (e: SummarizeEvent) => void;
  onError: (err: Error) => void;
}

function lastCallbacks(): StreamCb {
  return mockStream.mock.calls[mockStream.mock.calls.length - 1][1] as StreamCb;
}

function emitEvents(events: SummarizeEvent[]): void {
  const cb = lastCallbacks();
  for (const e of events) cb.onEvent(e);
}

const upload: UploadResponse = {
  ok: true,
  id: "doc1",
  filename: "report.pdf",
  char_count: 12345,
  chunk_count: 7,
  file_size: 999,
  extractor: "pypdf",
  summary: null,
  uploaded_at: Date.now(),
};

const baseProps = {
  upload,
  onDismiss: () => {},
  onViewDocuments: () => {},
  onChatAbout: () => {},
};

beforeEach(() => {
  vi.resetAllMocks();
  mockStream.mockReturnValue(() => {});
});

describe("UploadCard", () => {
  it("renders the filename and stats", () => {
    render(<UploadCard {...baseProps} />);
    expect(screen.getByText("report.pdf")).toBeInTheDocument();
    expect(screen.getByText(/7 chunks/)).toBeInTheDocument();
    expect(screen.getByText(/12,345 characters/)).toBeInTheDocument();
    expect(screen.getByText(/pypdf/)).toBeInTheDocument();
  });

  it("streams the summary with live progress", async () => {
    const user = userEvent.setup();
    render(<UploadCard {...baseProps} />);
    await user.click(screen.getByRole("button", { name: /summarize/i }));
    expect(mockStream).toHaveBeenCalledTimes(1);
    expect(mockStream.mock.calls[0][0]).toBe("doc1");
    emitEvents([
      { type: "start", mode: "map_reduce", total: 2 },
      { type: "progress", done: 1, total: 2, stage: "map" },
    ]);
    await waitFor(() =>
      expect(screen.getByText(/Summarizing section 1\/2/)).toBeInTheDocument(),
    );
    emitEvents([{ type: "done", summary: "**TL;DR** — good quarter." }]);
    await waitFor(() =>
      expect(screen.getByText(/good quarter/)).toBeInTheDocument(),
    );
  });

  it("shows an error when the stream fails", async () => {
    const user = userEvent.setup();
    render(<UploadCard {...baseProps} />);
    await user.click(screen.getByRole("button", { name: /summarize/i }));
    lastCallbacks().onError(new Error("LLM busy"));
    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent("LLM busy"),
    );
  });

  it("calls onChatAbout with the document id and filename", async () => {
    const user = userEvent.setup();
    const onChatAbout = vi.fn();
    render(<UploadCard {...baseProps} onChatAbout={onChatAbout} />);
    await user.click(
      screen.getByRole("button", { name: /chat about this document/i }),
    );
    expect(onChatAbout).toHaveBeenCalledTimes(1);
    expect(onChatAbout).toHaveBeenCalledWith("doc1", "report.pdf");
  });

  it("calls onDismiss when the ✕ button is clicked", async () => {
    const user = userEvent.setup();
    const onDismiss = vi.fn();
    render(<UploadCard {...baseProps} onDismiss={onDismiss} />);
    await user.click(screen.getByRole("button", { name: /dismiss/i }));
    expect(onDismiss).toHaveBeenCalledTimes(1);
  });

  it("calls onViewDocuments when the cabinet button is clicked", async () => {
    const user = userEvent.setup();
    const onViewDocuments = vi.fn();
    render(<UploadCard {...baseProps} onViewDocuments={onViewDocuments} />);
    await user.click(
      screen.getByRole("button", { name: /open document cabinet/i }),
    );
    expect(onViewDocuments).toHaveBeenCalledTimes(1);
  });
});
