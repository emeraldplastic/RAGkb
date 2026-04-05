import { useState, useEffect, useRef } from "react";
import axios from "axios";
import "./App.css";

const API = "http://127.0.0.1:8000";

export default function App() {
  const [documents, setDocuments] = useState([]);
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    fetchDocuments();
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const fetchDocuments = async () => {
    const res = await axios.get(`${API}/documents`);
    setDocuments(res.data.documents);
  };

  const uploadFile = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setUploading(true);
    const formData = new FormData();
    formData.append("file", file);
    try {
      await axios.post(`${API}/upload`, formData);
      await fetchDocuments();
      setMessages(prev => [...prev, {
        role: "system",
        text: `"${file.name}" uploaded successfully.`
      }]);
    } catch (err) {
      setMessages(prev => [...prev, {
        role: "system",
        text: `Upload failed: ${err.response?.data?.detail || err.message}`
      }]);
    }
    setUploading(false);
  };

  const deleteDocument = async (filename) => {
    await axios.delete(`${API}/documents/${filename}`);
    if (selectedDoc === filename) setSelectedDoc(null);
    await fetchDocuments();
  };

  const sendMessage = async () => {
    if (!question.trim() || loading) return;
    const q = question.trim();
    setQuestion("");
    setMessages(prev => [...prev, { role: "user", text: q }]);
    setLoading(true);
    try {
      const res = await axios.post(`${API}/chat`, {
    question: q,
    source_filter: selectedDoc || undefined
    });
      setMessages(prev => [...prev, {
        role: "assistant",
        text: res.data.answer,
        sources: res.data.sources
      }]);
    } catch (err) {
      setMessages(prev => [...prev, {
        role: "assistant",
        text: "Something went wrong. Is the backend running?"
      }]);
    }
    setLoading(false);
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="sidebar-header">
          <h1>RAG Knowledge Base</h1>
          <p>Chat with your documents</p>
        </div>

        <label className="upload-btn">
          {uploading ? "Uploading..." : "+ Upload PDF"}
          <input type="file" accept=".pdf" onChange={uploadFile} hidden />
        </label>

        <div className="doc-list">
          <p className="doc-list-label">Documents</p>
          {documents.length === 0 && (
            <p className="no-docs">No documents yet</p>
          )}
          {documents.map(doc => (
            <div
              key={doc}
              className={`doc-item ${selectedDoc === doc ? "active" : ""}`}
              onClick={() => setSelectedDoc(selectedDoc === doc ? null : doc)}
            >
              <span className="doc-name">{doc}</span>
              <button
                className="delete-btn"
                onClick={e => { e.stopPropagation(); deleteDocument(doc); }}
              >x</button>
            </div>
          ))}
        </div>

        {selectedDoc && (
          <div className="filter-badge">
            Filtering: {selectedDoc}
          </div>
        )}
        {!selectedDoc && documents.length > 0 && (
          <div className="filter-badge all">
            Searching all documents
          </div>
        )}
      </aside>

      <main className="chat">
        <div className="messages">
          {messages.length === 0 && (
            <div className="empty-state">
              <p>Upload a PDF and start asking questions</p>
            </div>
          )}
          {messages.map((msg, i) => (
            <div key={i} className={`message ${msg.role}`}>
              <div className="bubble">{msg.text}</div>
              {msg.sources && msg.sources.length > 0 && (
                <div className="sources">
                  {msg.sources.map((s, j) => (
                    <span key={j} className="source-tag">{s}</span>
                  ))}
                </div>
              )}
            </div>
          ))}
          {loading && (
            <div className="message assistant">
              <div className="bubble thinking">Thinking...</div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        <div className="input-row">
          <textarea
            value={question}
            onChange={e => setQuestion(e.target.value)}
            onKeyDown={handleKey}
            placeholder={selectedDoc ? `Ask about ${selectedDoc}...` : "Ask anything across all documents..."}
            rows={1}
          />
          <button onClick={sendMessage} disabled={loading || !question.trim()}>
            Send
          </button>
        </div>
      </main>
    </div>
  );
}