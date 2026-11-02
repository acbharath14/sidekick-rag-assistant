import { useCallback, useEffect, useRef, useState } from "react";
import { fetchConfig, fetchHealth, streamChat } from "./api";
import type {
  ChatResponse,
  ChatTurn,
  ConfigResponse,
  HealthResponse,
} from "./types";
import ChatMessage from "./components/ChatMessage";
import ContextBanner from "./components/ContextBanner";
import HealthIndicator from "./components/HealthIndicator";

let nextId = 1;

export default function App() {
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [config, setConfig] = useState<ConfigResponse | null>(null);
  const [configLoading, setConfigLoading] = useState(true);
  const [configError, setConfigError] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState(true);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [chatError, setChatError] = useState<string | null>(null);
  const cancelRef = useRef<(() => void) | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  const loadConfig = useCallback(async () => {
    setConfigLoading(true);
    setConfigError(null);
    try {
      setConfig(await fetchConfig());
    } catch (err) {
      setConfigError((err as Error).message);
    } finally {
      setConfigLoading(false);
    }
  }, []);

  const loadHealth = useCallback(async () => {
    setHealthLoading(true);
    setHealthError(null);
    try {
      setHealth(await fetchHealth());
    } catch (err) {
      setHealthError((err as Error).message);
    } finally {
      setHealthLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadConfig();
    void loadHealth();
  }, [loadConfig, loadHealth]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns]);

  useEffect(() => {
    return () => cancelRef.current?.();
  }, []);

  const historyFor = (upto: number): Array<[string, string]> => {
    const out: Array<[string, string]> = [];
    const slice = turns.slice(0, upto);
    for (let i = 0; i + 1 < slice.length; i += 2) {
      const userTurn = slice[i];
      const asstTurn = slice[i + 1];
      if (
        userTurn.role === "user" &&
        asstTurn.role === "assistant" &&
        asstTurn.response &&
        !asstTurn.streaming
      ) {
        out.push([userTurn.text, asstTurn.response.text]);
      }
    }
    return out;
  };

  const send = (): void => {
    const question = input.trim();
    if (!question || sending) return;
    setInput("");
    setChatError(null);

    const userId = nextId++;
    const asstId = nextId++;
    const history = historyFor(turns.length);

    setTurns((prev) => [
      ...prev,
      { id: userId, role: "user", text: question },
      { id: asstId, role: "assistant", text: "", streaming: true },
    ]);
    setSending(true);

    let streamed = "";
    cancelRef.current = streamChat(question, history, {
      onToken: (token) => {
        streamed += token;
        const snapshot = streamed;
        setTurns((prev) =>
          prev.map((t) =>
            t.id === asstId ? { ...t, text: snapshot } : t,
          ),
        );
      },
      onDone: (response: ChatResponse) => {
        setTurns((prev) =>
          prev.map((t) =>
            t.id === asstId
              ? { ...t, text: response.text, response, streaming: false }
              : t,
          ),
        );
        setSending(false);
        cancelRef.current = null;
      },
      onError: (err) => {
        setChatError(err.message);
        setTurns((prev) =>
          prev.map((t) =>
            t.id === asstId
              ? {
                  ...t,
                  text: "",
                  streaming: false,
                  response: {
                    text: "",
                    sources: [],
                    hits: [],
                    grounded: false,
                    latency_ms: null,
                    standalone_question: null,
                  },
                }
              : t,
          ),
        );
        setSending(false);
        cancelRef.current = null;
      },
    });
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>): void => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>Sidekick</h1>
        <HealthIndicator
          health={health}
          loading={healthLoading}
          error={healthError}
          onRetry={() => void loadHealth()}
        />
      </header>
      <ContextBanner
        config={config}
        loading={configLoading}
        error={configError}
      />
      <main className="chat" aria-live="polite">
        {turns.length === 0 && (
          <div className="empty-state">
            <p>Ask a question about the indexed corpus.</p>
            <p className="empty-hint">
              Answers stream in with cited sources. Thumbs up/down saves
              feedback for later review.
            </p>
          </div>
        )}
        {turns.map((t) =>
          t.role === "user" ? (
            <article key={t.id} className="message message-user">
              <div className="message-body">{t.text}</div>
            </article>
          ) : t.response ? (
            <ChatMessage
              key={t.id}
              question={
                turns.find((u) => u.role === "user" && u.id === t.id - 1)
                  ?.text ?? ""
              }
              response={t.response}
              streamingText={t.text}
              isStreaming={t.streaming}
            />
          ) : null,
        )}
        {chatError && (
          <div className="chat-error" role="alert">
            {chatError}
          </div>
        )}
        <div ref={bottomRef} />
      </main>
      <footer className="composer">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Ask a question… (Enter to send, Shift+Enter for a new line)"
          rows={2}
          disabled={sending}
          aria-label="Your question"
        />
        <button
          type="button"
          className="send-button"
          onClick={send}
          disabled={sending || !input.trim()}
        >
          {sending ? "…" : "Send"}
        </button>
      </footer>
    </div>
  );
}
