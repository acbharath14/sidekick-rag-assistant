import { useCallback, useEffect, useRef, useState } from "react";
import {
  clearConversationMessages,
  createConversation,
  deleteConversation,
  deleteDocument,
  fetchConfig,
  fetchHealth,
  getConversation,
  getDocument,
  listConversations,
  listDocuments,
  renameConversation,
  streamChat,
  uploadFile,
} from "./api";
import type {
  ChatResponse,
  ChatTurn,
  ConfigResponse,
  Conversation,
  ConversationMessage,
  DocumentDetail as DocumentDetailType,
  DocumentInfo,
  HealthResponse,
  UploadResponse,
} from "./types";
import AttachButton from "./components/AttachButton";
import ChatMessage, { formatTime } from "./components/ChatMessage";
import ContextBanner from "./components/ContextBanner";
import ConversationSidebar from "./components/ConversationSidebar";
import DocumentDetail from "./components/DocumentDetail";
import DocumentsView from "./components/DocumentsView";
import HealthIndicator from "./components/HealthIndicator";
import UploadCard from "./components/UploadCard";

let nextId = 1;

const ACTIVE_CONV_KEY = "sidekick.activeConversationId";
const SIDEBAR_KEY = "sidekick.sidebarCollapsed";

function messageToTurns(msgs: ConversationMessage[]): ChatTurn[] {
  const turns: ChatTurn[] = [];
  for (const m of msgs) {
    if (m.role === "user") {
      turns.push({
        id: nextId++,
        role: "user",
        text: m.text,
        createdAt: m.created_at * 1000,
      });
    } else {
      turns.push({
        id: nextId++,
        role: "assistant",
        text: m.text,
        createdAt: m.created_at * 1000,
        response: {
          text: m.text,
          sources: m.sources,
          hits: [],
          grounded: m.grounded ?? false,
          latency_ms: m.latency_ms,
          standalone_question: null,
        },
      });
    }
  }
  return turns;
}

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
  const [upload, setUpload] = useState<UploadResponse | null>(null);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [view, setView] = useState<"chat" | "documents">("chat");
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [documentsLoading, setDocumentsLoading] = useState(false);
  const [documentsError, setDocumentsError] = useState<string | null>(null);
  const [selectedDoc, setSelectedDoc] = useState<DocumentDetailType | null>(null);
  const [selectedDocLoading, setSelectedDocLoading] = useState(false);
  // Document-scoped chat: when set, questions only search this document.
  const [activeDocumentId, setActiveDocumentId] = useState<string | null>(null);
  const [activeDocumentName, setActiveDocumentName] = useState<string | null>(null);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [conversationsLoading, setConversationsLoading] = useState(true);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(
    () => localStorage.getItem(SIDEBAR_KEY) === "1",
  );
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

  const refreshConversations = useCallback(async () => {
    try {
      const data = await listConversations();
      setConversations(data.conversations);
      return data.conversations;
    } catch {
      return [];
    }
  }, []);

  const loadConversationMessages = useCallback(async (id: string) => {
    const detail = await getConversation(id);
    if ("ok" in detail && detail.ok === false) {
      throw new Error("Conversation not found");
    }
    setTurns(messageToTurns(detail.messages));
  }, []);

  // Boot: load conversations, restore or create the active one.
  useEffect(() => {
    (async () => {
      setConversationsLoading(true);
      const convs = await refreshConversations();
      let activeId = localStorage.getItem(ACTIVE_CONV_KEY);
      if (activeId && !convs.some((c) => c.id === activeId)) {
        activeId = null; // stored id no longer exists
      }
      if (!activeId) {
        if (convs.length > 0) {
          activeId = convs[0].id;
        } else {
          const created = await createConversation();
          activeId = created.id;
          await refreshConversations();
        }
      }
      setActiveConversationId(activeId);
      localStorage.setItem(ACTIVE_CONV_KEY, activeId);
      try {
        await loadConversationMessages(activeId);
      } catch {
        // New/empty conversation — nothing to load.
      } finally {
        setConversationsLoading(false);
      }
    })();
  }, [refreshConversations, loadConversationMessages]);

  const switchConversation = useCallback(
    async (id: string) => {
      if (id === activeConversationId || sending) return;
      cancelRef.current?.();
      cancelRef.current = null;
      setSending(false);
      setChatError(null);
      setUpload(null);
      setUploadError(null);
      // Switching conversations clears any document scope.
      setActiveDocumentId(null);
      setActiveDocumentName(null);
      setActiveConversationId(id);
      localStorage.setItem(ACTIVE_CONV_KEY, id);
      try {
        await loadConversationMessages(id);
      } catch (err) {
        setChatError((err as Error).message);
      }
    },
    [activeConversationId, sending, upload, loadConversationMessages],
  );

  const handleNewChat = useCallback(async () => {
    try {
      const created = await createConversation();
      await refreshConversations();
      await switchConversation(created.id);
    } catch (err) {
      setChatError((err as Error).message);
    }
  }, [refreshConversations, switchConversation]);

  const handleRenameConversation = useCallback(
    async (id: string, title: string) => {
      await renameConversation(id, title);
      await refreshConversations();
    },
    [refreshConversations],
  );

  const handleDeleteConversation = useCallback(
    async (id: string) => {
      await deleteConversation(id);
      const convs = await refreshConversations();
      if (id === activeConversationId) {
        if (convs.length > 0) {
          await switchConversation(convs[0].id);
        } else {
          const created = await createConversation();
          await refreshConversations();
          setActiveConversationId(created.id);
          localStorage.setItem(ACTIVE_CONV_KEY, created.id);
          setTurns([]);
        }
      }
    },
    [activeConversationId, refreshConversations, switchConversation],
  );

  const toggleSidebar = useCallback(() => {
    setSidebarCollapsed((prev) => {
      localStorage.setItem(SIDEBAR_KEY, prev ? "0" : "1");
      return !prev;
    });
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

  const handleStop = (): void => {
    cancelRef.current?.();
    cancelRef.current = null;
    setSending(false);
    setTurns((prev) =>
      prev.map((t) =>
        t.streaming ? { ...t, streaming: false, text: t.text + "\n\n*[stopped]*" } : t,
      ),
    );
  };

  const send = (): void => {
    const question = input.trim();
    if (!question || sending) return;
    setInput("");
    setChatError(null);

    const userId = nextId++;
    const asstId = nextId++;
    const history = historyFor(turns.length);
    const docId = activeDocumentId;
    const convId = activeConversationId;

    setTurns((prev) => [
      ...prev,
      { id: userId, role: "user", text: question, createdAt: Date.now() },
      { id: asstId, role: "assistant", text: "", createdAt: Date.now(), streaming: true },
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
        const titled = response as ChatResponse & { conversation_title?: string };
        if (titled.conversation_title && convId) {
          // First message auto-titled the conversation — refresh the sidebar.
          setConversations((prev) =>
            prev.map((c) =>
              c.id === convId ? { ...c, title: titled.conversation_title as string } : c,
            ),
          );
        }
        setTurns((prev) =>
          prev.map((t) =>
            t.id === asstId
              ? { ...t, text: response.text, response, streaming: false }
              : t,
          ),
        );
        setSending(false);
        cancelRef.current = null;
        void refreshConversations(); // bump ordering / message counts
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
    }, docId, convId);
  };

  const refreshDocuments = useCallback(async () => {
    setDocumentsLoading(true);
    setDocumentsError(null);
    try {
      const data = await listDocuments();
      setDocuments(data.documents);
    } catch (err) {
      setDocumentsError((err as Error).message);
    } finally {
      setDocumentsLoading(false);
    }
  }, []);

  const handleFile = async (file: File): Promise<void> => {
    if (uploadProgress !== null) return;
    setUpload(null);
    setUploadError(null);
    setUploadProgress(0);
    try {
      const res = await uploadFile(file, setUploadProgress);
      setUpload(res);
      await refreshDocuments();
    } catch (err) {
      setUploadError((err as Error).message);
    } finally {
      setUploadProgress(null);
    }
  };

  const handleDismissUpload = (): void => {
    setUpload(null);
  };

  const handleSelectDocument = useCallback(
    async (id: string) => {
      setSelectedDocLoading(true);
      try {
        const doc = await getDocument(id);
        setSelectedDoc(doc);
      } catch (err) {
        setDocumentsError((err as Error).message);
      } finally {
        setSelectedDocLoading(false);
      }
    },
    [],
  );

  const handleDeleteDocument = useCallback(
    async (id: string) => {
      await deleteDocument(id);
      if (selectedDoc?.id === id) setSelectedDoc(null);
      if (activeDocumentId === id) {
        setActiveDocumentId(null);
        setActiveDocumentName(null);
      }
      await refreshDocuments();
    },
    [selectedDoc, activeDocumentId, refreshDocuments],
  );

  const handleChatAboutDocument = useCallback(
    async (docId: string, filename: string) => {
      // Start a fresh conversation scoped to this document.
      cancelRef.current?.();
      cancelRef.current = null;
      setSending(false);
      setChatError(null);
      setTurns([]);
      try {
        const created = await createConversation(`About ${filename}`);
        await refreshConversations();
        setActiveConversationId(created.id);
        localStorage.setItem(ACTIVE_CONV_KEY, created.id);
      } catch (err) {
        setChatError((err as Error).message);
        return;
      }
      setActiveDocumentId(docId);
      setActiveDocumentName(filename);
      setSelectedDoc(null);
      setView("chat");
    },
    [refreshConversations],
  );

  const clearDocumentScope = useCallback(() => {
    setActiveDocumentId(null);
    setActiveDocumentName(null);
  }, []);

  const handleClearChat = (): void => {
    if (turns.length === 0 || !activeConversationId) return;
    if (!window.confirm("Clear the conversation? This cannot be undone.")) return;
    // Stop any in-flight stream; the active upload (if any) is left alone.
    cancelRef.current?.();
    cancelRef.current = null;
    setSending(false);
    const convId = activeConversationId;
    void (async () => {
      try {
        await clearConversationMessages(convId);
      } catch (err) {
        setChatError((err as Error).message);
        return;
      }
      setTurns([]);
      setChatError(null);
      await refreshConversations();
    })();
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>): void => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <div className="app">
      <ConversationSidebar
        conversations={conversations}
        activeId={activeConversationId}
        collapsed={sidebarCollapsed}
        onToggle={toggleSidebar}
        onSelect={(id) => void switchConversation(id)}
        onNew={() => void handleNewChat()}
        onRename={handleRenameConversation}
        onDelete={handleDeleteConversation}
      />
      <div className="app-main">
      <header className="app-header">
        <div className="header-nav">
          <h1>Sidekick</h1>
          <nav className="view-tabs" aria-label="Views">
            <button
              type="button"
              className={"view-tab" + (view === "chat" ? " view-tab-active" : "")}
              onClick={() => {
                setSelectedDoc(null);
                setView("chat");
              }}
            >
              💬 Chat
            </button>
            <button
              type="button"
              className={
                "view-tab" + (view === "documents" ? " view-tab-active" : "")
              }
              onClick={() => {
                setView("documents");
                void refreshDocuments();
              }}
            >
              📁 Documents
            </button>
          </nav>
        </div>
        <div className="header-actions">
          {sending && (
            <span className="processing-pill" aria-live="polite">
              <span className="thinking-dot" />
              <span className="thinking-dot" />
              <span className="thinking-dot" />
              Processing…
            </span>
          )}
          {view === "chat" && (
            <button
              type="button"
              className="clear-chat-btn"
              onClick={handleClearChat}
              disabled={turns.length === 0}
              title="Clear the conversation"
            >
              Clear chat
            </button>
          )}
          <HealthIndicator
            health={health}
            loading={healthLoading}
            error={healthError}
            onRetry={() => void loadHealth()}
          />
        </div>
      </header>
      <ContextBanner
        config={config}
        loading={configLoading}
        error={configError}
      />
      {view === "documents" ? (
        <main className="chat">
          {selectedDoc ? (
            selectedDocLoading ? (
              <div className="empty-state">
                <p>Loading document…</p>
              </div>
            ) : (
              <DocumentDetail
                doc={selectedDoc}
                onBack={() => setSelectedDoc(null)}
                onChatAbout={(id, name) => void handleChatAboutDocument(id, name)}
                onDelete={handleDeleteDocument}
                onSummaryUpdated={(s) =>
                  setSelectedDoc((prev) => (prev ? { ...prev, summary: s } : prev))
                }
              />
            )
          ) : (
            <DocumentsView
              documents={documents}
              loading={documentsLoading}
              error={documentsError}
              onSelect={(id) => void handleSelectDocument(id)}
              onDelete={handleDeleteDocument}
              onChatAbout={(id, name) => void handleChatAboutDocument(id, name)}
              onUpload={(f) => void handleFile(f)}
              uploading={uploadProgress !== null}
              uploadProgress={uploadProgress}
            />
          )}
        </main>
      ) : (
      <>
      <main className="chat" aria-live="polite">
        {conversationsLoading ? (
          <div className="empty-state">
            <p>Loading conversations…</p>
          </div>
        ) : (
          <>
        {turns.length === 0 && (
          <div className="empty-state">
            <p>Ask anything.</p>
            <p className="empty-hint">
              Sidekick searches your documents first and answers with cited
              sources — or from general knowledge when the corpus has no
              answer. Thumbs up/down saves feedback for later review.
            </p>
          </div>
        )}
        {turns.map((t) =>
          t.role === "user" ? (
            <article key={t.id} className="message message-user">
              <div className="message-body">{t.text}</div>
              <div className="message-time message-time-user">
                {formatTime(t.createdAt)}
              </div>
            </article>
          ) : t.response ? (
            <ChatMessage
              key={t.id}
              question={
                turns.find((u) => u.role === "user" && u.id === t.id - 1)
                  ?.text ?? ""
              }
              response={t.response}
              createdAt={t.createdAt}
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
          </>
        )}
        <div ref={bottomRef} />
      </main>
      <footer className="composer">
        {activeDocumentId && activeDocumentName && (
          <div className="doc-scope-banner" role="status">
            <span>
              📄 Chatting about: <strong>{activeDocumentName}</strong>
            </span>
            <button
              type="button"
              className="upload-banner-clear"
              onClick={clearDocumentScope}
              aria-label="Clear document scope"
              title="Ask about the full knowledge base instead"
            >
              ✕
            </button>
          </div>
        )}
        {upload && (
          <UploadCard
            upload={upload}
            onDismiss={handleDismissUpload}
            onViewDocuments={() => {
              setSelectedDoc(null);
              setView("documents");
              void refreshDocuments();
            }}
            onChatAbout={(id, name) => void handleChatAboutDocument(id, name)}
          />
        )}
        {uploadError && (
          <div className="upload-error" role="alert">
            {uploadError}
          </div>
        )}
        <div className="composer-row">
          <AttachButton
            onFile={(f) => void handleFile(f)}
            disabled={false}
            progress={uploadProgress}
          />
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder={
              activeDocumentId
                ? `Ask about ${activeDocumentName}… (Enter to send)`
                : "Ask a question… (Enter to send, Shift+Enter for a new line)"
            }
            rows={2}
            disabled={sending}
            aria-label="Your question"
          />
          <button
            type="button"
            className="send-button"
            onClick={sending ? handleStop : send}
            disabled={!sending && !input.trim()}
          >
            {sending ? "Stop" : "Send"}
          </button>
        </div>
      </footer>
      </>
      )}
      </div>
    </div>
  );
}
