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
  id: string;
  filename: string;
  char_count: number;
  chunk_count: number;
  file_size: number;
  extractor: string;
  summary: string | null;
  uploaded_at: number;
  error?: string;
}

export interface DocumentInfo {
  id: string;
  filename: string;
  chunk_count: number;
  char_count: number;
  file_size: number;
  extractor: string;
  has_summary: boolean;
  uploaded_at: number;
}

export interface DocumentDetail {
  ok: boolean;
  id: string;
  filename: string;
  chunk_count: number;
  char_count: number;
  file_size: number;
  extractor: string;
  summary: string | null;
  uploaded_at: number;
  error?: string;
}

export interface SummarizeResponse {
  ok: boolean;
  summary: string;
  error?: string;
}

export type ChatRole = "user" | "assistant";

export interface ChatTurn {
  id: number;
  role: ChatRole;
  text: string;
  createdAt: number;
  response?: ChatResponse;
  streaming?: boolean;
}

export interface Conversation {
  id: string;
  title: string;
  created_at: number;
  updated_at: number;
  message_count: number;
  last_message: string | null;
  last_message_at: number | null;
}

export interface ConversationMessage {
  role: ChatRole;
  text: string;
  sources: string[];
  grounded: boolean | null;
  latency_ms: number | null;
  created_at: number;
}

export interface ConversationDetail {
  id: string;
  title: string;
  created_at: number;
  updated_at: number;
  messages: ConversationMessage[];
}

export interface ChatResponseWithTitle extends ChatResponse {
  conversation_title?: string;
}
