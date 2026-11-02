export interface Hit {
  source: string;
  text: string;
  score: number;
}

export interface ChatResponse {
  text: string;
  sources: string[];
  hits: Hit[];
  grounded: boolean;
  latency_ms: number | null;
  standalone_question: string | null;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  ollama_reachable: boolean;
  index_present: boolean;
}

export interface ConfigResponse {
  ollama_model: string;
  embedding_model: string;
  retrieval: string;
  rerank: boolean;
  hybrid_fallback: boolean;
}

export interface FeedbackResponse {
  ok: boolean;
  error?: string;
}

export interface UploadResponse {
  ok: boolean;
  upload_id: string;
  filename: string;
  char_count: number;
  chunk_count: number;
  summary: string;
  error?: string;
}

export type ChatRole = "user" | "assistant";

export interface ChatTurn {
  id: number;
  role: ChatRole;
  text: string;
  response?: ChatResponse;
  streaming?: boolean;
}
