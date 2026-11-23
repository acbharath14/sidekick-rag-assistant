/**
 * Tests for src/api.ts — uploadFile (mocked XHR) and summarizeDocument (mocked fetch).
 */
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  fetchConnectorJob,
  fetchConnectors,
  startConnectorSync,
  streamChat,
  streamSummarizeDocument,
  summarizeDocument,
  uploadFile,
} from "../api";
import type { AgentProgressEvent, SummarizeEvent } from "../api";
import type { ChatResponse } from "../types";

// --- Mock XMLHttpRequest ----------------------------------------------------

type Listener = (e: unknown) => void;

class MockXHR {
  static instances: MockXHR[] = [];
  upload: { listeners: Map<string, Listener[]>; addEventListener: (t: string, l: Listener) => void };
  listeners = new Map<string, Listener[]>();
  status = 0;
  statusText = "";
  responseText = "";
  opened = "";
  sent: unknown = null;

  constructor() {
    MockXHR.instances.push(this);
    this.upload = {
      listeners: new Map<string, Listener[]>(),
      addEventListener: (t: string, l: Listener) => {
        const arr = this.upload.listeners.get(t) ?? [];
        arr.push(l);
        this.upload.listeners.set(t, arr);
      },
    };
  }
  addEventListener(t: string, l: Listener) {
    const arr = this.listeners.get(t) ?? [];
    arr.push(l);
    this.listeners.set(t, arr);
  }
  open(_m: string, url: string) {
    this.opened = url;
  }
  send(body: unknown) {
    this.sent = body;
  }
  fireUploadProgress(loaded: number, total: number) {
    for (const l of this.upload.listeners.get("progress") ?? []) {
      l({ lengthComputable: true, loaded, total });
    }
  }
  fireLoad() {
    for (const l of this.listeners.get("load") ?? []) l({});
  }
  fireError() {
    for (const l of this.listeners.get("error") ?? []) l({});
  }
}

const realXHR = globalThis.XMLHttpRequest;

beforeEach(() => {
  MockXHR.instances = [];
  // @ts-expect-error — replacing the global with a test double
  globalThis.XMLHttpRequest = MockXHR;
  vi.restoreAllMocks();
});

function restoreXHR() {
  globalThis.XMLHttpRequest = realXHR;
}

// --- uploadFile --------------------------------------------------------------

describe("uploadFile", () => {
  it("resolves with the upload response on success", async () => {
    const p = uploadFile(new File(["x"], "doc.pdf", { type: "application/pdf" }));
    const xhr = MockXHR.instances[0];
    xhr.status = 200;
    xhr.responseText = JSON.stringify({
      ok: true,
      id: "abc123",
      filename: "doc.pdf",
      char_count: 100,
      chunk_count: 2,
      file_size: 1,
      extractor: "pypdf",
      summary: null,
      uploaded_at: Date.now(),
    });
    xhr.fireLoad();
    const res = await p;
    expect(res.ok).toBe(true);
    expect(res.filename).toBe("doc.pdf");
    expect(xhr.opened).toContain("/api/upload");
    restoreXHR();
  });

  it("calls onProgress with percentages during upload", async () => {
    const seen: number[] = [];
    const p = uploadFile(
      new File(["x"], "doc.pdf"),
      (pct) => seen.push(pct),
    );
    const xhr = MockXHR.instances[0];
    xhr.fireUploadProgress(25, 100);
    xhr.fireUploadProgress(75, 100);
    expect(seen).toEqual([25, 75]);
    xhr.status = 200;
    xhr.responseText = JSON.stringify({
      ok: true,
      id: "a",
      filename: "doc.pdf",
      char_count: 1,
      chunk_count: 1,
      file_size: 1,
      extractor: "pypdf",
      summary: null,
      uploaded_at: Date.now(),
    });
    xhr.fireLoad();
    await p;
    restoreXHR();
  });

  it("rejects when the server returns ok:false", async () => {
    const p = uploadFile(new File(["x"], "doc.pdf"));
    const xhr = MockXHR.instances[0];
    xhr.status = 200;
    xhr.responseText = JSON.stringify({ ok: false, error: "No text found" });
    xhr.fireLoad();
    await expect(p).rejects.toThrow("No text found");
    restoreXHR();
  });

  it("rejects on network error", async () => {
    const p = uploadFile(new File(["x"], "doc.pdf"));
    const xhr = MockXHR.instances[0];
    xhr.fireError();
    await expect(p).rejects.toThrow("network error");
    restoreXHR();
  });
});

// --- summarizeDocument ---------------------------------------------------------

describe("summarizeDocument", () => {
  it("returns the summary on success", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ ok: true, summary: "Short summary." }),
      }),
    );
    const s = await summarizeDocument("doc1");
    expect(s).toBe("Short summary.");
    vi.unstubAllGlobals();
  });

  it("throws the server error message when ok:false", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ ok: false, error: "LLM busy" }),
      }),
    );
    await expect(summarizeDocument("doc1")).rejects.toThrow("LLM busy");
    vi.unstubAllGlobals();
  });

  it("throws on HTTP failure", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, status: 500, statusText: "Boom" }),
    );
    await expect(summarizeDocument("doc1")).rejects.toThrow("500");
    vi.unstubAllGlobals();
  });
});

// --- streamSummarizeDocument ---------------------------------------------------

function mockSseFetch(chunks: string[]) {
  const bytes = new TextEncoder().encode(chunks.join(""));
  // Small reads to exercise the client's buffering logic.
  const parts: Uint8Array[] = [];
  for (let i = 0; i < bytes.length; i += 13) parts.push(bytes.slice(i, i + 13));
  let idx = 0;
  const reader = {
    read: async (): Promise<{ done: boolean; value?: Uint8Array }> => {
      if (idx >= parts.length) return { done: true, value: undefined };
      return { done: false, value: parts[idx++] };
    },
  };
  return vi
    .fn()
    .mockResolvedValue({ ok: true, body: { getReader: () => reader } });
}

describe("streamSummarizeDocument", () => {
  it("emits start/progress/done events in order", async () => {
    vi.stubGlobal(
      "fetch",
      mockSseFetch([
        `data: {"type":"start","mode":"map_reduce","total":2}\n\n`,
        `data: {"type":"progress","done":1,"total":2,"stage":"map"}\n\n`,
        `data: {"type":"progress","done":2,"total":2,"stage":"map"}\n\n`,
        `data: {"type":"progress","stage":"reduce"}\n\n`,
        `data: {"type":"done","summary":"Final summary."}\n\n`,
      ]),
    );
    const seen: SummarizeEvent[] = [];
    await new Promise<void>((resolve, reject) => {
      streamSummarizeDocument("doc1", {
        onEvent: (e) => {
          seen.push(e);
          if (e.type === "done") resolve();
        },
        onError: reject,
      });
    });
    expect(seen.map((e) => e.type)).toEqual([
      "start",
      "progress",
      "progress",
      "progress",
      "done",
    ]);
    expect(seen[0]).toMatchObject({ mode: "map_reduce", total: 2 });
    expect(seen[seen.length - 1]).toMatchObject({ summary: "Final summary." });
    vi.unstubAllGlobals();
  });

  it("routes error events to onError", async () => {
    vi.stubGlobal(
      "fetch",
      mockSseFetch([`data: {"type":"error","error":"Document not found."}\n\n`]),
    );
    await new Promise<void>((resolve) => {
      streamSummarizeDocument("nope", {
        onEvent: () => {},
        onError: (err) => {
          expect(err.message).toBe("Document not found.");
          resolve();
        },
      });
    });
    vi.unstubAllGlobals();
  });

  it("routes HTTP failures to onError", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, status: 500, statusText: "Boom" }),
    );
    await new Promise<void>((resolve) => {
      streamSummarizeDocument("doc1", {
        onEvent: () => {},
        onError: (err) => {
          expect(err.message).toContain("500");
          resolve();
        },
      });
    });
    vi.unstubAllGlobals();
  });
});

// --- streamChat ---------------------------------------------------------------

const DONE_RESPONSE: ChatResponse = {
  text: "agent answer",
  sources: ["doc.md"],
  hits: [],
  grounded: true,
  latency_ms: 12,
  standalone_question: null,
};

function runStreamChat(
  options?: { agentic?: boolean },
): Promise<{
  tokens: string[];
  progress: AgentProgressEvent[];
  done: ChatResponse | null;
  error: Error | null;
  fetchMock: ReturnType<typeof mockSseFetch>;
}> {
  const tokens: string[] = [];
  const progress: AgentProgressEvent[] = [];
  let done: ChatResponse | null = null;
  let error: Error | null = null;
  const fetchMock = mockSseFetch([
    `data: {"type":"progress","phase":"thought","message":"Planning search strategy..."}\n\n`,
    `data: {"type":"progress","phase":"action","message":"Searching [search_docs]: parcels"}\n\n`,
    `data: {"token":"agent "}\n\n`,
    `data: {"token":"answer"}\n\n`,
    `data: {"done":${JSON.stringify(DONE_RESPONSE)}}\n\n`,
  ]);
  vi.stubGlobal("fetch", fetchMock);
  return new Promise((resolve) => {
    streamChat(
      "q",
      [],
      {
        onToken: (t) => tokens.push(t),
        onProgress: (e) => progress.push(e),
        onDone: (r) => {
          done = r;
          vi.unstubAllGlobals();
          resolve({ tokens, progress, done, error, fetchMock });
        },
        onError: (err) => {
          error = err;
          vi.unstubAllGlobals();
          resolve({ tokens, progress, done, error, fetchMock });
        },
      },
      null,
      null,
      options,
    );
  });
}

describe("streamChat", () => {
  it("routes progress events to onProgress, then tokens, then done", async () => {
    const { tokens, progress, done, error } = await runStreamChat({
      agentic: true,
    });
    expect(error).toBeNull();
    expect(progress.map((p) => p.phase)).toEqual(["thought", "action"]);
    expect(progress[1].message).toContain("Searching [search_docs]");
    expect(tokens).toEqual(["agent ", "answer"]);
    expect(done).toEqual(DONE_RESPONSE);
  });

  it("sends agentic:true in the request body when enabled", async () => {
    const { fetchMock } = await runStreamChat({ agentic: true });
    const body = JSON.parse(
      String(fetchMock.mock.calls[0][1].body),
    ) as Record<string, unknown>;
    expect(body.agentic).toBe(true);
    expect(body.question).toBe("q");
  });

  it("defaults agentic to false in the request body", async () => {
    const { fetchMock } = await runStreamChat();
    const body = JSON.parse(
      String(fetchMock.mock.calls[0][1].body),
    ) as Record<string, unknown>;
    expect(body.agentic).toBe(false);
  });

  it("routes error events to onError", async () => {
    vi.stubGlobal(
      "fetch",
      mockSseFetch([
        `data: {"type":"error","error":"The model took too long to respond. Try again in a moment."}\n\n`,
      ]),
    );
    await new Promise<void>((resolve) => {
      streamChat(
        "q",
        [],
        {
          onToken: () => {},
          onDone: () => {},
          onError: (err) => {
            expect(err.message).toContain("too long");
            vi.unstubAllGlobals();
            resolve();
          },
        },
        null,
        null,
        { agentic: true },
      );
    });
  });
});

describe("fetchConnectors", () => {
  it("returns the connector list", async () => {
    const payload = {
      connectors: [
        {
          source: "github",
          label: "GitHub",
          auth: "Personal Access Token",
          configured: false,
          missing_env: ["GITHUB_TOKEN", "GITHUB_REPO"],
          setup_hint: "Set GITHUB_TOKEN.",
          last_sync: null,
          total_docs: 0,
        },
      ],
    };
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(payload),
      }),
    );
    const data = await fetchConnectors();
    expect(data.connectors[0].source).toBe("github");
    expect(data.connectors[0].configured).toBe(false);
    vi.unstubAllGlobals();
  });

  it("throws on HTTP failure", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, status: 500, statusText: "Boom" }),
    );
    await expect(fetchConnectors()).rejects.toThrow("500");
    vi.unstubAllGlobals();
  });
});

describe("startConnectorSync", () => {
  it("POSTs the token and full flag, and returns the job", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: () =>
        Promise.resolve({
          job_id: "abc123",
          source: "github",
          status: "running",
          started_at: null,
          finished_at: null,
          summary: null,
          error: null,
        }),
    });
    vi.stubGlobal("fetch", fetchMock);
    const job = await startConnectorSync("github", {
      token: "FAKE-TOKEN",
      full: true,
    });
    expect(job.job_id).toBe("abc123");
    expect(job.status).toBe("running");
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/api/connectors/github/sync"),
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ token: "FAKE-TOKEN", full: true }),
      }),
    );
    vi.unstubAllGlobals();
  });
});

describe("fetchConnectorJob", () => {
  it("returns the job for polling", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            job_id: "abc123",
            source: "github",
            status: "done",
            started_at: "2026-11-16T00:00:00Z",
            finished_at: "2026-11-16T00:00:05Z",
            summary: { source: "github", fetched: 9, total: 9, last_sync: "2026-10-05T08:00:00Z" },
            error: null,
          }),
      }),
    );
    const job = await fetchConnectorJob("abc123");
    expect(job.status).toBe("done");
    expect(job.summary?.fetched).toBe(9);
    vi.unstubAllGlobals();
  });
});
