/**
 * Tests for components/ConnectorsView.tsx (mocked ../api).
 */
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import ConnectorsView from "../components/ConnectorsView";
import type { ConnectorStatus, ConnectorSyncJob } from "../types";

afterEach(() => cleanup());

vi.mock("../api", () => ({
  fetchConnectors: vi.fn(),
  startConnectorSync: vi.fn(),
  fetchConnectorJob: vi.fn(),
}));

import {
  fetchConnectors,
  fetchConnectorJob,
  startConnectorSync,
} from "../api";

const mockFetch = fetchConnectors as unknown as ReturnType<typeof vi.fn>;
const mockStart = startConnectorSync as unknown as ReturnType<typeof vi.fn>;
const mockJob = fetchConnectorJob as unknown as ReturnType<typeof vi.fn>;

const CONNECTORS: ConnectorStatus[] = [
  {
    source: "confluence",
    label: "Confluence",
    auth: "Personal Access Token",
    configured: true,
    missing_env: [],
    setup_hint: "Set CONFLUENCE_URL.",
    last_sync: "2026-10-05T10:00:00Z",
    total_docs: 6,
  },
  {
    source: "github",
    label: "GitHub",
    auth: "Personal Access Token",
    configured: false,
    missing_env: ["GITHUB_TOKEN", "GITHUB_REPO"],
    setup_hint: "Create a fine-grained PAT.",
    last_sync: null,
    total_docs: 0,
  },
  {
    source: "outlook",
    label: "Outlook",
    auth: "OAuth2 bearer token",
    configured: false,
    missing_env: ["OUTLOOK_BEARER_TOKEN"],
    setup_hint: "Paste a Graph token.",
    last_sync: null,
    total_docs: 0,
  },
];

const RUNNING_JOB: ConnectorSyncJob = {
  job_id: "job1",
  source: "github",
  status: "running",
  started_at: "2026-11-16T00:00:00Z",
  finished_at: null,
  summary: null,
  error: null,
};

const DONE_JOB: ConnectorSyncJob = {
  ...RUNNING_JOB,
  status: "done",
  finished_at: "2026-11-16T00:00:02Z",
  summary: {
    source: "github",
    fetched: 9,
    total: 9,
    last_sync: "2026-10-05T08:00:00Z",
  },
};

beforeEach(() => {
  vi.resetAllMocks();
  mockFetch.mockResolvedValue({ connectors: CONNECTORS });
});

describe("ConnectorsView", () => {
  it("renders one card per source with status and stats", async () => {
    render(<ConnectorsView />);
    expect(await screen.findByText("Confluence")).toBeInTheDocument();
    expect(screen.getByText("GitHub")).toBeInTheDocument();
    expect(screen.getByText("Outlook")).toBeInTheDocument();
    expect(screen.getByText("Configured")).toBeInTheDocument();
    expect(screen.getAllByText("Not configured")).toHaveLength(2);
    expect(screen.getByText(/6 documents/)).toBeInTheDocument();
    expect(screen.getAllByText(/Never synced/)).toHaveLength(2);
    expect(
      screen.getByText("Missing: GITHUB_TOKEN, GITHUB_REPO"),
    ).toBeInTheDocument();
  });

  it("shows a token input only for unconfigured sources", async () => {
    render(<ConnectorsView />);
    await screen.findByText("GitHub");
    expect(screen.getByLabelText("GitHub token")).toBeInTheDocument();
    expect(screen.getByLabelText("Outlook token")).toBeInTheDocument();
    expect(screen.queryByLabelText("Confluence token")).toBeNull();
  });

  it("syncs with the pasted token, clears it, and shows the result", async () => {
    const user = userEvent.setup();
    mockStart.mockResolvedValue(RUNNING_JOB);
    mockJob.mockResolvedValue(DONE_JOB);
    render(<ConnectorsView />);
    await screen.findByText("GitHub");

    await user.type(screen.getByLabelText("GitHub token"), "FAKE-TOKEN");
    const buttons = screen.getAllByRole("button", { name: "Sync now" });
    await user.click(buttons[1]); // GitHub is the second card
    expect(mockStart).toHaveBeenCalledWith("github", {
      token: "FAKE-TOKEN",
      full: false,
    });
    // The pasted token is cleared once the sync starts.
    expect(
      (screen.getByLabelText("GitHub token") as HTMLInputElement).value,
    ).toBe("");
    expect(
      await screen.findByText(/Synced: 9 new\/changed, 9 total\./, {}, { timeout: 5000 }),
    ).toBeInTheDocument();
    expect(mockJob).toHaveBeenCalledWith("job1");
    // The list refreshes once the job finishes.
    expect(mockFetch.mock.calls.length).toBeGreaterThan(1);
  });

  it("shows the error when starting a sync fails", async () => {
    const user = userEvent.setup();
    mockStart.mockRejectedValue(new Error("POST failed: 409 Conflict"));
    render(<ConnectorsView />);
    await screen.findByText("GitHub");
    const buttons = screen.getAllByRole("button", { name: "Sync now" });
    await user.click(buttons[1]);
    expect(
      await screen.findByText("POST failed: 409 Conflict"),
    ).toBeInTheDocument();
  });

  it("shows a retry when the list fails to load", async () => {
    const user = userEvent.setup();
    mockFetch.mockRejectedValueOnce(new Error("Network down"));
    render(<ConnectorsView />);
    expect(
      await screen.findByText(/Couldn’t load connectors: Network down/),
    ).toBeInTheDocument();
    mockFetch.mockResolvedValue({ connectors: CONNECTORS });
    await user.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByText("GitHub")).toBeInTheDocument();
  });
});
