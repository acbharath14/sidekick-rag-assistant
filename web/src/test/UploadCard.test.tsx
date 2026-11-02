/**
 * Tests for components/UploadCard.tsx.
 */
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import UploadCard from "../components/UploadCard";
import type { UploadResponse } from "../types";

afterEach(() => cleanup());

vi.mock("../api", () => ({
  summarizeDocument: vi.fn(),
}));

import { summarizeDocument } from "../api";

const mockSummarize = summarizeDocument as unknown as ReturnType<typeof vi.fn>;

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

beforeEach(() => {
  vi.resetAllMocks();
});

describe("UploadCard", () => {
  it("renders the filename and stats", () => {
    render(
      <UploadCard upload={upload} onDismiss={() => {}} onViewDocuments={() => {}} />,
    );
    expect(screen.getByText("report.pdf")).toBeInTheDocument();
    expect(screen.getByText(/7 chunks/)).toBeInTheDocument();
    expect(screen.getByText(/12,345 characters/)).toBeInTheDocument();
    expect(screen.getByText(/pypdf/)).toBeInTheDocument();
  });

  it("calls summarizeDocument and shows the summary", async () => {
    const user = userEvent.setup();
    mockSummarize.mockResolvedValue("**TL;DR** — good quarter.");
    render(
      <UploadCard upload={upload} onDismiss={() => {}} onViewDocuments={() => {}} />,
    );
    await user.click(screen.getByRole("button", { name: /summarize/i }));
    expect(mockSummarize).toHaveBeenCalledWith("doc1");
    await waitFor(() =>
      expect(screen.getByText(/good quarter/)).toBeInTheDocument(),
    );
  });

  it("shows an error when summarization fails", async () => {
    const user = userEvent.setup();
    mockSummarize.mockRejectedValue(new Error("LLM busy"));
    render(
      <UploadCard upload={upload} onDismiss={() => {}} onViewDocuments={() => {}} />,
    );
    await user.click(screen.getByRole("button", { name: /summarize/i }));
    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent("LLM busy"),
    );
  });

  it("calls onDismiss when the ✕ button is clicked", async () => {
    const user = userEvent.setup();
    const onDismiss = vi.fn();
    render(
      <UploadCard upload={upload} onDismiss={onDismiss} onViewDocuments={() => {}} />,
    );
    await user.click(screen.getByRole("button", { name: /dismiss/i }));
    expect(onDismiss).toHaveBeenCalledTimes(1);
  });

  it("calls onViewDocuments when the cabinet button is clicked", async () => {
    const user = userEvent.setup();
    const onViewDocuments = vi.fn();
    render(
      <UploadCard upload={upload} onDismiss={() => {}} onViewDocuments={onViewDocuments} />,
    );
    await user.click(
      screen.getByRole("button", { name: /open document cabinet/i }),
    );
    expect(onViewDocuments).toHaveBeenCalledTimes(1);
  });
});
