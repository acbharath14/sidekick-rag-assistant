import type {
  ChatResponse,
  ConfigResponse,
  FeedbackResponse,
  HealthResponse,
  SummarizeResponse,
  UploadResponse,
} from "./types";

const API_BASE: string =
  (import.meta.env.VITE_API_URL as string | undefined) ??
  "http://127.0.0.1:8000";

export function apiBase(): string {
  return API_BASE;
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`POST ${path} failed: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    throw new Error(`GET ${path} failed: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}

export function fetchHealth(): Promise<HealthResponse> {
  return getJson<HealthResponse>("/api/health");
}

export function fetchConfig(): Promise<ConfigResponse> {
  return getJson<ConfigResponse>("/api/config");
}

export function postChat(
  question: string,
  history: Array<[string, string]>,
  uploadId?: string | null,
): Promise<ChatResponse> {
  return postJson<ChatResponse>("/api/chat", {
    question,
    history,
    upload_id: uploadId ?? null,
  });
}

export function postFeedback(
  question: string,
  vote: "up" | "down",
): Promise<FeedbackResponse> {
  return postJson<FeedbackResponse>("/api/feedback", { question, vote });
}

export function uploadFile(
  file: File,
  onProgress?: (percent: number) => void,
): Promise<UploadResponse> {
  // XMLHttpRequest because fetch has no upload-progress events.
  // Progress covers the byte transfer; the server then extracts text and
  // builds the index synchronously, so 100% means "waiting on the server".
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const form = new FormData();
    form.append("file", file, file.name);

    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    });

    xhr.addEventListener("load", () => {
      if (xhr.status < 200 || xhr.status >= 300) {
        reject(
          new Error(`POST /api/upload failed: ${xhr.status} ${xhr.statusText}`),
        );
        return;
      }
      let data: UploadResponse;
      try {
        data = JSON.parse(xhr.responseText) as UploadResponse;
      } catch {
        reject(new Error("Upload failed: invalid server response"));
        return;
      }
      if (!data.ok) {
        reject(new Error(data.error ?? "Upload failed"));
        return;
      }
      resolve(data);
    });

    xhr.addEventListener("error", () =>
      reject(new Error("Upload failed: network error")),
    );
    xhr.addEventListener("abort", () =>
      reject(new Error("Upload cancelled")),
    );
    xhr.addEventListener("timeout", () =>
      reject(new Error("Upload timed out")),
    );

    xhr.open("POST", `${API_BASE}/api/upload`);
    xhr.send(form);
  });
}

export async function summarizeUpload(uploadId: string): Promise<string> {
  const data = await postJson<SummarizeResponse>(
    `/api/upload/${encodeURIComponent(uploadId)}/summarize`,
    {},
  );
  if (!data.ok) {
    throw new Error(data.error ?? "Summarize failed");
  }
  return data.summary;
}

export async function deleteUpload(uploadId: string): Promise<void> {
  try {
    await fetch(`${API_BASE}/api/upload/${encodeURIComponent(uploadId)}`, {
      method: "DELETE",
    });
  } catch {
    // Best-effort cleanup; the server holds uploads in memory only.
  }
}

export interface StreamCallbacks {
  onToken: (token: string) => void;
  onDone: (response: ChatResponse) => void;
  onError: (err: Error) => void;
}

export function streamChat(
  question: string,
  history: Array<[string, string]>,
  cb: StreamCallbacks,
  uploadId?: string | null,
): () => void {
  const ctrl = new AbortController();
  let buffer = "";

  const pump = async (): Promise<void> => {
    try {
      const res = await fetch(`${API_BASE}/api/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          history,
          upload_id: uploadId ?? null,
        }),
        signal: ctrl.signal,
      });
      if (!res.ok || !res.body) {
        throw new Error(
          `POST /api/chat/stream failed: ${res.status} ${res.statusText}`,
        );
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop() ?? "";
        for (const evt of events) {
          const line = evt.trim();
          if (!line.startsWith("data:")) continue;
          const payload = JSON.parse(line.slice(5).trim()) as
            | { token: string }
            | { done: ChatResponse };
          if ("token" in payload) {
            cb.onToken(payload.token);
          } else {
            cb.onDone(payload.done);
          }
        }
      }
    } catch (err) {
      if ((err as Error).name === "AbortError") return;
      cb.onError(err as Error);
    }
  };

  void pump();
  return () => ctrl.abort();
}
