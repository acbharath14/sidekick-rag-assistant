import type {
  ChatResponse,
  ConfigResponse,
  Conversation,
  ConversationDetail,
  DocumentDetail,
  DocumentInfo,
  FeedbackResponse,
  HealthResponse,
  UploadResponse,
} from "./types";

const API_BASE: string =
  (import.meta.env.VITE_API_URL as string | undefined) ??
  "http://127.0.0.1:8000";

const API_KEY: string | undefined =
  (import.meta.env.VITE_API_KEY as string | undefined) || undefined;

export function apiBase(): string {
  return API_BASE;
}

function authHeaders(): Record<string, string> {
  return API_KEY ? { "X-API-Key": API_KEY } : {};
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`POST ${path} failed: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { ...authHeaders() },
  });
  if (!res.ok) {
    throw new Error(`GET ${path} failed: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}

async function patchJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`PATCH ${path} failed: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}

async function deleteJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "DELETE",
    headers: { ...authHeaders() },
  });
  if (!res.ok) {
    throw new Error(`DELETE ${path} failed: ${res.status} ${res.statusText}`);
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
  documentId?: string | null,
  conversationId?: string | null,
): Promise<ChatResponse> {
  return postJson<ChatResponse>("/api/chat", {
    question,
    history,
    document_id: documentId ?? null,
    conversation_id: conversationId ?? null,
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
    if (API_KEY) xhr.setRequestHeader("X-API-Key", API_KEY);
    xhr.send(form);
  });
}

export async function summarizeUpload(uploadId: string): Promise<string> {
  // Legacy alias — the cabinet now uses summarizeDocument.
  return summarizeDocument(uploadId);
}

export interface DocumentListResponse {
  documents: DocumentInfo[];
}

export function listDocuments(): Promise<DocumentListResponse> {
  return getJson<DocumentListResponse>("/api/documents");
}

export async function getDocument(id: string): Promise<DocumentDetail> {
  const data = await getJson<DocumentDetail>(
    `/api/documents/${encodeURIComponent(id)}`,
  );
  if (!data.ok) throw new Error(data.error ?? "Document not found");
  return data;
}

export async function summarizeDocument(docId: string): Promise<string> {
  const data = await postJson<{ ok: boolean; summary: string; error?: string }>(
    `/api/documents/${encodeURIComponent(docId)}/summarize`,
    {},
  );
  if (!data.ok) throw new Error(data.error ?? "Summarize failed");
  return data.summary;
}

/** Summarize event payloads streamed by POST /api/documents/{id}/summarize/stream. */
export type SummarizeEvent =
  | { type: "start"; mode: "single" | "map_reduce"; total: number }
  | { type: "progress"; done?: number; total?: number; stage: "map" | "reduce" }
  | { type: "done"; summary: string }
  | { type: "error"; error: string };

export interface SummarizeStreamCallbacks {
  onEvent: (event: SummarizeEvent) => void;
  onError: (err: Error) => void;
}

/**
 * Stream summarize progress events via fetch + ReadableStream (same pattern
 * as streamChat, so the X-API-Key header keeps working). Returns a cancel
 * function that aborts the stream.
 */
export function streamSummarizeDocument(
  docId: string,
  cb: SummarizeStreamCallbacks,
): () => void {
  const ctrl = new AbortController();
  let buffer = "";

  const pump = async (): Promise<void> => {
    try {
      const res = await fetch(
        `${API_BASE}/api/documents/${encodeURIComponent(docId)}/summarize/stream`,
        {
          method: "POST",
          headers: { ...authHeaders() },
          signal: ctrl.signal,
        },
      );
      if (!res.ok || !res.body) {
        throw new Error(
          `POST /api/documents/${docId}/summarize/stream failed: ${res.status} ${res.statusText}`,
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
          const payload = JSON.parse(line.slice(5).trim()) as SummarizeEvent;
          if (payload.type === "error") {
            throw new Error(payload.error);
          }
          cb.onEvent(payload);
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

export async function deleteDocument(docId: string): Promise<void> {
  const data = await deleteJson<OkResponse>(
    `/api/documents/${encodeURIComponent(docId)}`,
  );
  if (!data.ok) throw new Error(data.error ?? "Delete failed");
}

// --- Conversations ---

export interface ConversationListResponse {
  conversations: Conversation[];
}

export function listConversations(): Promise<ConversationListResponse> {
  return getJson<ConversationListResponse>("/api/conversations");
}

export interface CreateConversationResponse {
  id: string;
  title: string;
  created_at: number;
}

export function createConversation(title?: string): Promise<CreateConversationResponse> {
  return postJson<CreateConversationResponse>(
    "/api/conversations",
    title ? { title } : {},
  );
}

export function getConversation(id: string): Promise<ConversationDetail> {
  return getJson<ConversationDetail>(
    `/api/conversations/${encodeURIComponent(id)}`,
  );
}

export interface OkResponse {
  ok: boolean;
  error?: string;
}

export async function renameConversation(id: string, title: string): Promise<void> {
  const data = await patchJson<OkResponse>(
    `/api/conversations/${encodeURIComponent(id)}`,
    { title },
  );
  if (!data.ok) throw new Error(data.error ?? "Rename failed");
}

export async function deleteConversation(id: string): Promise<void> {
  const data = await deleteJson<OkResponse>(
    `/api/conversations/${encodeURIComponent(id)}`,
  );
  if (!data.ok) throw new Error(data.error ?? "Delete failed");
}

export async function clearConversationMessages(id: string): Promise<void> {
  const data = await deleteJson<OkResponse>(
    `/api/conversations/${encodeURIComponent(id)}/messages`,
  );
  if (!data.ok) throw new Error(data.error ?? "Clear failed");
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
  documentId?: string | null,
  conversationId?: string | null,
): () => void {
  const ctrl = new AbortController();
  let buffer = "";

  const pump = async (): Promise<void> => {
    try {
      const res = await fetch(`${API_BASE}/api/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify({
          question,
          history,
          document_id: documentId ?? null,
          conversation_id: conversationId ?? null,
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
