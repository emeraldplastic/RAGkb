import { useState, useEffect, useRef, useCallback } from "react";
import axios from "axios";
import "./App.css";

const API =
  process.env.REACT_APP_API_URL ||
  (window.location.hostname === "localhost" ? "http://127.0.0.1:8000" : "");
const STREAM_FLUSH_INTERVAL_MS = 50;
const MAX_CONCURRENT_UPLOADS = 3;

// ── Axios instance with auth header ─────────────────────
const api = axios.create({ baseURL: API });

api.interceptors.request.use((config) => {
  const token = localStorage.getItem("token");
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem("token");
      localStorage.removeItem("user");
      window.location.reload();
    }
    return Promise.reject(err);
  }
);

// ═══════════════════════════════════════════════════════════
//  AUTH PAGE
// ═══════════════════════════════════════════════════════════

function AuthPage({ onLogin }) {
  const [isRegister, setIsRegister] = useState(false);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      const endpoint = isRegister ? "/api/register" : "/api/login";
      const res = await axios.post(`${API}${endpoint}`, { username, password });
      localStorage.setItem("token", res.data.token);
      localStorage.setItem("user", JSON.stringify(res.data.user));
      onLogin(res.data.user);
    } catch (err) {
      setError(err.response?.data?.detail || "Something went wrong");
    }
    setLoading(false);
  };

  return (
    <>
      <div className="bg-animate" />
      <div className="auth-container">
        <div className="auth-card">
          <div className="logo">
            <div className="logo-icon">🧠</div>
            <span className="logo-text">RAG KB</span>
          </div>
          <p className="subtitle">Your private AI knowledge base</p>

          <h2>{isRegister ? "Create Account" : "Welcome Back"}</h2>

          {error && <div className="error-msg">{error}</div>}

          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label htmlFor="username">Username</label>
              <input
                id="username"
                type="text"
                placeholder="Enter your username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                autoComplete="username"
                required
              />
            </div>
            <div className="form-group">
              <label htmlFor="password">Password</label>
              <input
                id="password"
                type="password"
                placeholder="Enter your password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                autoComplete={isRegister ? "new-password" : "current-password"}
                required
              />
            </div>
            <button type="submit" className="btn-primary" disabled={loading}>
              {loading ? "Please wait..." : isRegister ? "Create Account" : "Sign In"}
            </button>
          </form>

          <div className="auth-switch">
            {isRegister ? "Already have an account? " : "Don't have an account? "}
            <button onClick={() => { setIsRegister(!isRegister); setError(""); }}>
              {isRegister ? "Sign In" : "Create one"}
            </button>
          </div>
        </div>
      </div>
    </>
  );
}

// ═══════════════════════════════════════════════════════════
//  MAIN APP (Authenticated)
// ═══════════════════════════════════════════════════════════

function MainApp({ user, onLogout }) {
  const [documents, setDocuments] = useState([]);
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);
  const bottomRef = useRef(null);
  const dropdownRef = useRef(null);
  const streamBufferRef = useRef("");
  const streamTextRef = useRef("");
  const streamSourcesRef = useRef([]);
  const streamFlushTimerRef = useRef(null);

  const fetchDocuments = useCallback(async () => {
    try {
      const res = await api.get("/api/documents");
      setDocuments(res.data.documents);
    } catch (err) {
      console.error("Failed to fetch documents:", err);
    }
  }, []);

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  const scrollToBottom = useCallback((behavior = "auto") => {
    bottomRef.current?.scrollIntoView({ behavior });
  }, []);

  useEffect(() => {
    scrollToBottom("smooth");
  }, [messages.length, scrollToBottom]);

  const flushAssistantMessage = useCallback(() => {
    setMessages((prev) => {
      if (prev.length === 0) return prev;
      const lastIndex = prev.length - 1;
      const last = prev[lastIndex];
      if (last.role !== "assistant") return prev;

      const updated = {
        ...last,
        text: streamTextRef.current,
        sources: streamSourcesRef.current,
      };
      const next = [...prev];
      next[lastIndex] = updated;
      return next;
    });
    scrollToBottom("auto");
  }, [scrollToBottom]);

  const scheduleAssistantFlush = useCallback(
    (immediate = false) => {
      if (immediate) {
        if (streamFlushTimerRef.current) {
          clearTimeout(streamFlushTimerRef.current);
          streamFlushTimerRef.current = null;
        }
        flushAssistantMessage();
        return;
      }

      if (streamFlushTimerRef.current) return;
      streamFlushTimerRef.current = setTimeout(() => {
        streamFlushTimerRef.current = null;
        flushAssistantMessage();
      }, STREAM_FLUSH_INTERVAL_MS);
    },
    [flushAssistantMessage]
  );

  useEffect(
    () => () => {
      if (streamFlushTimerRef.current) {
        clearTimeout(streamFlushTimerRef.current);
      }
    },
    []
  );

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClick = (e) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setShowDropdown(false);
      }
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  const uploadFile = async (e) => {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;
    setUploading(true);

    let successCount = 0;
    let currentIndex = 0;

    const worker = async () => {
      while (currentIndex < files.length) {
        const file = files[currentIndex];
        currentIndex += 1;

        const formData = new FormData();
        formData.append("file", file);

        try {
          await api.post("/api/upload", formData);
          successCount += 1;
        } catch (err) {
          console.error("Upload failed for", file.name, err);
        }
      }
    };

    try {
      await Promise.all(
        Array.from(
          { length: Math.min(MAX_CONCURRENT_UPLOADS, files.length) },
          () => worker()
        )
      );
      await fetchDocuments();
      setMessages((prev) => [
        ...prev,
        {
          role: "system",
          text: `Uploaded ${successCount}/${files.length} document(s)`,
        },
      ]);
      scrollToBottom("smooth");
    } finally {
      e.target.value = "";
      setUploading(false);
    }
  };

  const deleteDocument = async (doc) => {
    try {
      await api.delete(`/api/documents/${doc.id}`);
      if (selectedDoc?.id === doc.id) setSelectedDoc(null);
      await fetchDocuments();
      setMessages((prev) => [
        ...prev,
        { role: "system", text: `🗑️ Deleted "${doc.name}"` },
      ]);
    } catch (err) {
      console.error("Delete failed:", err);
    }
  };

  const sendMessage = async () => {
    if (!question.trim() || loading) return;
    const q = question.trim();
    setQuestion("");
    streamBufferRef.current = "";
    streamTextRef.current = "";
    streamSourcesRef.current = [];
    setMessages((prev) => [
      ...prev,
      { role: "user", text: q },
      { role: "assistant", text: "", sources: [] },
    ]);
    setLoading(true);
    scrollToBottom("smooth");

    const processLine = (line) => {
      const trimmed = line.trim();
      if (!trimmed) return;

      let parsed;
      try {
        parsed = JSON.parse(trimmed);
      } catch (parseError) {
        console.error("Streaming parse error:", parseError, trimmed);
        return;
      }

      if (parsed.type === "sources") {
        streamSourcesRef.current = Array.isArray(parsed.data) ? parsed.data : [];
        scheduleAssistantFlush(true);
        return;
      }

      if (parsed.type === "token") {
        streamTextRef.current += parsed.data || "";
        scheduleAssistantFlush();
        return;
      }

      if (parsed.type === "error") {
        throw new Error(parsed.data || "Unknown server error");
      }
    };

    try {
      const token = localStorage.getItem("token");
      const res = await fetch(`${API}/api/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          question: q,
          document_id: selectedDoc?.id || undefined,
        }),
      });

      if (!res.ok) {
        throw new Error(`Server returned ${res.status}`);
      }

      const reader = res.body?.getReader();
      if (!reader) {
        throw new Error("Empty response stream");
      }

      const decoder = new TextDecoder("utf-8");

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
        streamBufferRef.current = "";
      }
      scheduleAssistantFlush(true);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Unknown error";
      if (streamFlushTimerRef.current) {
        clearTimeout(streamFlushTimerRef.current);
        streamFlushTimerRef.current = null;
      }
      setMessages((prev) => {
        if (prev.length === 0) {
          return [{ role: "assistant", text: `Error: ${errorMessage}` }];
        }

        const lastIndex = prev.length - 1;
        const last = prev[lastIndex];
        if (last.role === "assistant") {
          const next = [...prev];
          next[lastIndex] = {
            ...last,
            text: last.text ? last.text : `Error: ${errorMessage}`,
          };
          return next;
        }

        return [...prev, { role: "assistant", text: `Error: ${errorMessage}` }];
      });
    } finally {
      setLoading(false);
    }
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const formatSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1048576).toFixed(1)} MB`;
  };

  const getFileIcon = (type) => {
    if (type === ".pdf") return "📕";
    if (type === ".txt") return "📝";
    if (type === ".md") return "📘";
    return "📄";
  };

  return (
    <>
      <div className="bg-animate" />
      <div className="app">
        {/* ── Sidebar ─────────────────────────────── */}
        <aside className="sidebar">
          <div className="sidebar-header">
            <div className="sidebar-brand">
              <div className="logo-icon">🧠</div>
              <h1>RAG KB</h1>
            </div>
            <div className="user-menu" ref={dropdownRef}>
              <div
                className="user-avatar"
                onClick={() => setShowDropdown(!showDropdown)}
                title={user.username}
              >
                {user.username[0].toUpperCase()}
              </div>
              {showDropdown && (
                <div className="user-dropdown">
                  <div className="user-info">
                    <div className="name">{user.username}</div>
                    <div className="label">Signed in</div>
                  </div>
                  <button onClick={onLogout}>⏻ Sign Out</button>
                </div>
              )}
            </div>
          </div>

          <label className="upload-btn" id="upload-button">
            <span className="icon">{uploading ? "⏳" : "＋"}</span>
            <span>{uploading ? "Uploading..." : "Upload Document"}</span>
            <input
              type="file"
              accept=".pdf,.txt,.md"
              onChange={uploadFile}
              hidden
              disabled={uploading}
              multiple
              webkitdirectory="true"
            />
          </label>

          <div className="section-label">Your Documents</div>
          <div className="doc-list">
            {documents.length === 0 && (
              <p className="no-docs">No documents uploaded yet</p>
            )}
            {documents.map((doc) => (
              <div
                key={doc.id}
                className={`doc-item ${selectedDoc?.id === doc.id ? "active" : ""}`}
                onClick={() =>
                  setSelectedDoc(selectedDoc?.id === doc.id ? null : doc)
                }
              >
                <span className="doc-icon">{getFileIcon(doc.type)}</span>
                <div className="doc-info">
                  <div className="doc-name" title={doc.name}>{doc.name}</div>
                  <div className="doc-meta">
                    {doc.pages} pages · {formatSize(doc.size)}
                  </div>
                </div>
                <button
                  className="delete-btn"
                  title="Delete document"
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteDocument(doc);
                  }}
                >
                  ✕
                </button>
              </div>
            ))}
          </div>

          {selectedDoc && (
            <div className="filter-badge">
              <span className="badge-icon">🔍</span>
              Searching: {selectedDoc.name}
            </div>
          )}
          {!selectedDoc && documents.length > 0 && (
            <div className="filter-badge all">
              <span className="badge-icon">📚</span>
              Searching all documents
            </div>
          )}
        </aside>

        {/* ── Chat Area ───────────────────────────── */}
        <main className="chat">
          <div className="messages" id="messages-container">
            {messages.length === 0 && (
              <div className="empty-state">
                <div className="empty-icon">💬</div>
                <h2>Ask your documents anything</h2>
                <p>
                  Upload PDFs, text files, or markdown documents, then ask
                  questions. Your AI assistant will find answers from your
                  personal knowledge base.
                </p>
              </div>
            )}

            {messages.map((msg, i) => {
              const isStreamingPlaceholder =
                msg.role === "assistant" &&
                loading &&
                i === messages.length - 1 &&
                !msg.text;

              return (
                <div key={i} className={`message ${msg.role}`}>
                  <div className={`bubble ${isStreamingPlaceholder ? "thinking" : ""}`}>
                    {isStreamingPlaceholder ? (
                      <>
                        <span className="dot" />
                        <span className="dot" />
                        <span className="dot" />
                      </>
                    ) : (
                      msg.text
                    )}
                  </div>
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="sources">
                      {msg.sources.map((s, j) => (
                        <span key={j} className="source-tag">
                          {s.name} {s.page !== "" && s.page != null ? `p.${s.page}` : ""}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
            <div ref={bottomRef} />
          </div>

          <div className="input-row">
            <textarea
              id="chat-input"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={handleKey}
              placeholder={
                selectedDoc
                  ? `Ask about ${selectedDoc.name}...`
                  : "Ask anything across your documents..."
              }
              rows={1}
            />
            <button
              className="send-btn"
              id="send-button"
              onClick={sendMessage}
              disabled={loading || !question.trim()}
              title="Send message"
            >
              ➤
            </button>
          </div>
        </main>
      </div>
    </>
  );
}

// ═══════════════════════════════════════════════════════════
//  ROOT — Routes between Auth and Main
// ═══════════════════════════════════════════════════════════

export default function App() {
  const [user, setUser] = useState(() => {
    const stored = localStorage.getItem("user");
    return stored ? JSON.parse(stored) : null;
  });

  const handleLogin = (userData) => {
    setUser(userData);
  };

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    setUser(null);
  };

  if (!user) {
    return <AuthPage onLogin={handleLogin} />;
  }

  return <MainApp user={user} onLogout={handleLogout} />;
}
