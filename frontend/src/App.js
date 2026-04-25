import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import axios from "axios";
import "./App.css";

const API =
  process.env.REACT_APP_API_URL ||
  (window.location.hostname === "localhost" ? "http://127.0.0.1:8000" : "");

const STREAM_FLUSH_INTERVAL_MS = 50;
const MAX_CONCURRENT_UPLOADS = 3;
const TOKEN_KEY = "ragkb_token";
const USER_KEY = "ragkb_user";

const storage = {
  getToken() {
    return sessionStorage.getItem(TOKEN_KEY);
  },
  setToken(value) {
    sessionStorage.setItem(TOKEN_KEY, value);
  },
  getUser() {
    const raw = sessionStorage.getItem(USER_KEY);
    if (!raw) return null;
    try {
      return JSON.parse(raw);
    } catch (err) {
      sessionStorage.removeItem(USER_KEY);
      return null;
    }
  },
  setUser(value) {
    sessionStorage.setItem(USER_KEY, JSON.stringify(value));
  },
  clear() {
    sessionStorage.removeItem(TOKEN_KEY);
    sessionStorage.removeItem(USER_KEY);
  },
};

const api = axios.create({ baseURL: API });
api.interceptors.request.use((config) => {
  const token = storage.getToken();
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      storage.clear();
      window.location.reload();
    }
    return Promise.reject(error);
  }
);

function formatApiError(err, fallback) {
  if (err?.response?.status === 429) {
    const retry = err.response?.headers?.["retry-after"];
    return retry
      ? `Rate limit reached. Retry in about ${retry} second(s).`
      : "Rate limit reached. Please wait and try again.";
  }
  return err?.response?.data?.detail || fallback;
}

function Brand() {
  return (
    <div className="brand">
      <div className="brand-mark" aria-hidden>
        KB
      </div>
      <div>
        <div className="brand-title">RAG Knowledge Base</div>
        <div className="brand-subtitle">Private answers from your own documents</div>
      </div>
    </div>
  );
}

function PrivacyNote({ compact = false }) {
  return (
    <div className={`privacy-note ${compact ? "compact" : ""}`}>
      <span className="privacy-dot" />
      Privacy mode: token is session-only, API responses are no-store, and you can delete account data.
    </div>
  );
}

function AuthPage({ onLogin }) {
  const [isRegister, setIsRegister] = useState(false);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError("");
    setLoading(true);
    try {
      const endpoint = isRegister ? "/api/register" : "/api/login";
      const response = await axios.post(`${API}${endpoint}`, {
        username: username.trim(),
        password,
      });
      storage.setToken(response.data.token);
      storage.setUser(response.data.user);
      onLogin(response.data.user);
    } catch (err) {
      setError(formatApiError(err, "Unable to sign in right now."));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-shell">
      <div className="aurora aurora-one" />
      <div className="aurora aurora-two" />
      <div className="auth-card">
        <Brand />
        <h1>{isRegister ? "Create your secure workspace" : "Welcome back"}</h1>
        <p className="auth-copy">
          Ask questions across your own docs with retrieval grounded answers and visible sources.
        </p>
        <PrivacyNote compact />

        {error && <div className="error-panel">{error}</div>}

        <form onSubmit={handleSubmit} className="auth-form">
          <label htmlFor="username">Username</label>
          <input
            id="username"
            type="text"
            autoComplete="username"
            value={username}
            onChange={(event) => setUsername(event.target.value)}
            placeholder="e.g. acme_ops"
            required
          />

          <label htmlFor="password">Password</label>
          <input
            id="password"
            type="password"
            autoComplete={isRegister ? "new-password" : "current-password"}
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            placeholder={isRegister ? "At least 10 chars with symbol" : "Your password"}
            required
          />

          <button type="submit" disabled={loading}>
            {loading ? "Working..." : isRegister ? "Create Account" : "Sign In"}
          </button>
        </form>

        <button
          type="button"
          className="auth-switch"
          onClick={() => {
            setIsRegister((value) => !value);
            setError("");
          }}
        >
          {isRegister ? "Already registered? Sign in" : "Need an account? Register"}
        </button>
      </div>
    </div>
  );
}

function SourceChips({ sources }) {
  if (!sources || sources.length === 0) return null;
  return (
    <div className="source-list">
      {sources.map((source, index) => (
        <span key={`${source.name}-${source.page}-${index}`} className="source-chip">
          {source.name}
          {source.page !== "" && source.page != null ? ` p.${source.page}` : ""}
          {typeof source.confidence === "number"
            ? ` (${Math.round(source.confidence * 100)}%)`
            : ""}
        </span>
      ))}
    </div>
  );
}

function Message({ message, loading }) {
  const isStreamingPlaceholder =
    message.role === "assistant" && loading && !message.text && !message.error;

  return (
    <div className={`message-row ${message.role}`}>
      <div className={`message-bubble ${message.error ? "error" : ""}`}>
        {isStreamingPlaceholder ? (
          <div className="typing">
            <span />
            <span />
            <span />
          </div>
        ) : (
          message.text
        )}
      </div>
      <SourceChips sources={message.sources} />
    </div>
  );
}

function MainApp({ user, onLogout }) {
  const [documents, setDocuments] = useState([]);
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [pendingDelete, setPendingDelete] = useState(false);

  const bottomRef = useRef(null);
  const menuRef = useRef(null);
  const textareaRef = useRef(null);
  const streamBufferRef = useRef("");
  const streamTextRef = useRef("");
  const streamSourcesRef = useRef([]);
  const streamTimerRef = useRef(null);

  const fetchDocuments = useCallback(async () => {
    try {
      const response = await api.get("/api/documents");
      setDocuments(response.data.documents);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "system", text: formatApiError(err, "Failed to refresh documents."), error: true },
      ]);
    }
  }, []);

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length]);

  useEffect(() => {
    const closeMenu = (event) => {
      if (!menuRef.current || menuRef.current.contains(event.target)) return;
      setMenuOpen(false);
    };
    document.addEventListener("mousedown", closeMenu);
    return () => document.removeEventListener("mousedown", closeMenu);
  }, []);

  useEffect(() => {
    if (!textareaRef.current) return;
    textareaRef.current.style.height = "auto";
    textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 160)}px`;
  }, [question]);

  const flushAssistantMessage = useCallback(() => {
    setMessages((prev) => {
      if (prev.length === 0) return prev;
      const last = prev[prev.length - 1];
      if (last.role !== "assistant") return prev;
      const updated = {
        ...last,
        text: streamTextRef.current,
        sources: streamSourcesRef.current,
      };
      const next = [...prev];
      next[next.length - 1] = updated;
      return next;
    });
  }, []);

  const scheduleFlush = useCallback(
    (immediate = false) => {
      if (immediate) {
        if (streamTimerRef.current) {
          clearTimeout(streamTimerRef.current);
          streamTimerRef.current = null;
        }
        flushAssistantMessage();
        return;
      }
      if (streamTimerRef.current) return;
      streamTimerRef.current = setTimeout(() => {
        streamTimerRef.current = null;
        flushAssistantMessage();
      }, STREAM_FLUSH_INTERVAL_MS);
    },
    [flushAssistantMessage]
  );

  useEffect(
    () => () => {
      if (streamTimerRef.current) {
        clearTimeout(streamTimerRef.current);
      }
    },
    []
  );

  const uploadFiles = async (event) => {
    const files = Array.from(event.target.files || []);
    event.target.value = "";
    if (files.length === 0) return;

    setUploading(true);
    let successCount = 0;
    let pointer = 0;

    const worker = async () => {
      while (pointer < files.length) {
        const file = files[pointer];
        pointer += 1;
        try {
          const body = new FormData();
          body.append("file", file);
          await api.post("/api/upload", body);
          successCount += 1;
        } catch (err) {
          setMessages((prev) => [
            ...prev,
            {
              role: "system",
              text: `${file.name}: ${formatApiError(err, "Upload failed.")}`,
              error: true,
            },
          ]);
        }
      }
    };

    try {
      await Promise.all(
        Array.from({ length: Math.min(MAX_CONCURRENT_UPLOADS, files.length) }, () => worker())
      );
      await fetchDocuments();
      setMessages((prev) => [
        ...prev,
        {
          role: "system",
          text: `Uploaded ${successCount}/${files.length} document(s)`,
        },
      ]);
    } finally {
      setUploading(false);
    }
  };

  const handleDeleteDocument = async (doc) => {
    try {
      await api.delete(`/api/documents/${doc.id}`);
      if (selectedDoc?.id === doc.id) {
        setSelectedDoc(null);
      }
      await fetchDocuments();
      setMessages((prev) => [...prev, { role: "system", text: `Deleted "${doc.name}"` }]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "system",
          text: formatApiError(err, "Document deletion failed."),
          error: true,
        },
      ]);
    }
  };

  const handleDeleteAccount = async () => {
    if (pendingDelete) return;
    const confirmed = window.confirm(
      "Delete your account and all uploaded documents? This cannot be undone."
    );
    if (!confirmed) return;

    setPendingDelete(true);
    try {
      await api.delete("/api/me");
      storage.clear();
      window.location.reload();
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "system",
          text: formatApiError(err, "Account deletion failed."),
          error: true,
        },
      ]);
    } finally {
      setPendingDelete(false);
      setMenuOpen(false);
    }
  };

  const sendMessage = async () => {
    const text = question.trim();
    if (!text || loading) return;

    setQuestion("");
    setMessages((prev) => [...prev, { role: "user", text }, { role: "assistant", text: "", sources: [] }]);
    setLoading(true);

    streamBufferRef.current = "";
    streamTextRef.current = "";
    streamSourcesRef.current = [];

    const processLine = (line) => {
      const value = line.trim();
      if (!value) return;
      let payload;
      try {
        payload = JSON.parse(value);
      } catch (err) {
        return;
      }

      if (payload.type === "sources") {
        streamSourcesRef.current = Array.isArray(payload.data) ? payload.data : [];
        scheduleFlush(true);
        return;
      }
      if (payload.type === "token") {
        streamTextRef.current += payload.data || "";
        scheduleFlush(false);
        return;
      }
      if (payload.type === "error") {
        throw new Error(payload.data || "Unknown server error");
      }
    };

    try {
      const response = await fetch(`${API}/api/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${storage.getToken()}`,
        },
        body: JSON.stringify({
          question: text,
          document_id: selectedDoc?.id || undefined,
        }),
      });

      if (!response.ok) {
        throw new Error(`Request failed (${response.status})`);
      }
      if (!response.body) {
        throw new Error("Missing response stream");
      }

      const decoder = new TextDecoder("utf-8");
      const reader = response.body.getReader();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (!value) continue;

        streamBufferRef.current += decoder.decode(value, { stream: true });
        let newlineIndex = streamBufferRef.current.indexOf("\n");
        while (newlineIndex !== -1) {
          const line = streamBufferRef.current.slice(0, newlineIndex);
          streamBufferRef.current = streamBufferRef.current.slice(newlineIndex + 1);
          processLine(line);
          newlineIndex = streamBufferRef.current.indexOf("\n");
        }
      }

      streamBufferRef.current += decoder.decode();
      if (streamBufferRef.current.trim()) {
        processLine(streamBufferRef.current);
      }
      streamBufferRef.current = "";
      scheduleFlush(true);
    } catch (err) {
      const textError = err instanceof Error ? err.message : "Chat failed.";
      setMessages((prev) => {
        if (prev.length === 0) return prev;
        const next = [...prev];
        const lastIndex = next.length - 1;
        if (next[lastIndex].role === "assistant") {
          next[lastIndex] = {
            ...next[lastIndex],
            text: next[lastIndex].text || textError,
            error: true,
          };
          return next;
        }
        return [...next, { role: "assistant", text: textError, error: true }];
      });
    } finally {
      setLoading(false);
    }
  };

  const documentScopeText = useMemo(() => {
    if (selectedDoc) return `Focused on: ${selectedDoc.name}`;
    if (documents.length > 0) return "Searching across all documents";
    return "Upload documents to start asking questions";
  }, [selectedDoc, documents.length]);

  const formatSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="app-shell">
      <div className="aurora aurora-one" />
      <div className="aurora aurora-two" />

      <aside className="sidebar">
        <Brand />
        <PrivacyNote />

        <div className="sidebar-actions">
          <label className="upload-button">
            {uploading ? "Uploading..." : "Upload documents"}
            <input
              type="file"
              accept=".pdf,.txt,.md"
              onChange={uploadFiles}
              disabled={uploading}
              multiple
              hidden
            />
          </label>
        </div>

        <div className="section-header">Documents</div>
        <div className="document-list">
          {documents.length === 0 && <p className="empty-docs">No documents yet.</p>}
          {documents.map((doc) => (
            <button
              key={doc.id}
              type="button"
              className={`document-item ${selectedDoc?.id === doc.id ? "active" : ""}`}
              onClick={() => setSelectedDoc((prev) => (prev?.id === doc.id ? null : doc))}
            >
              <span className="document-name" title={doc.name}>
                {doc.name}
              </span>
              <span className="document-meta">
                {doc.pages} pages - {formatSize(doc.size)}
              </span>
              <span
                className="document-delete"
                onClick={(event) => {
                  event.stopPropagation();
                  handleDeleteDocument(doc);
                }}
                role="button"
                tabIndex={0}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    event.stopPropagation();
                    handleDeleteDocument(doc);
                  }
                }}
              >
                Delete
              </span>
            </button>
          ))}
        </div>

        <div className="scope-chip">{documentScopeText}</div>

        <div className="user-menu" ref={menuRef}>
          <button
            type="button"
            className="user-button"
            onClick={() => setMenuOpen((value) => !value)}
          >
            <span>{user.username}</span>
            <span className="caret">{menuOpen ? "▲" : "▼"}</span>
          </button>
          {menuOpen && (
            <div className="user-dropdown">
              <button type="button" onClick={onLogout}>
                Sign out
              </button>
              <button type="button" className="danger" onClick={handleDeleteAccount}>
                {pendingDelete ? "Deleting..." : "Delete account and data"}
              </button>
            </div>
          )}
        </div>
      </aside>

      <main className="chat-pane">
        <div className="message-list">
          {messages.length === 0 && (
            <div className="empty-state">
              <h2>Ask anything grounded in your documents</h2>
              <p>
                Upload files, choose a document scope, then ask your question. Answers include the
                retrieved source references.
              </p>
            </div>
          )}

          {messages.map((message, index) => (
            <Message key={`${message.role}-${index}`} message={message} loading={loading} />
          ))}
          <div ref={bottomRef} />
        </div>

        <div className="composer">
          <textarea
            ref={textareaRef}
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
              }
            }}
            rows={1}
            placeholder={
              selectedDoc
                ? `Ask about "${selectedDoc.name}"...`
                : "Ask a question across your knowledge base..."
            }
          />
          <button type="button" onClick={sendMessage} disabled={!question.trim() || loading}>
            Send
          </button>
        </div>
      </main>
    </div>
  );
}

export default function App() {
  const [user, setUser] = useState(() => storage.getUser());

  if (!user) {
    return <AuthPage onLogin={setUser} />;
  }
  return (
    <MainApp
      user={user}
      onLogout={() => {
        storage.clear();
        setUser(null);
      }}
    />
  );
}
