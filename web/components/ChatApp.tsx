"use client";

import { useEffect, useMemo, useRef, useState, useTransition, type KeyboardEvent } from "react";
import MarkdownMessage from "@/components/MarkdownMessage";
import {
  ApiError,
  newSessionId,
  postQuery,
  type AgentIntent,
  type QueryResponse,
  type RetrievedDocument,
} from "@/lib/api";

type ChatRole = "user" | "assistant";

interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  intent?: AgentIntent;
  blocked?: boolean;
  documents?: RetrievedDocument[];
  status?: string;
  error?: string | null;
}

function intentLabel(intent?: AgentIntent): string {
  if (!intent) return "—";
  if (intent === "technical") return "Technical";
  if (intent === "conversational") return "Conversational";
  if (intent === "blocked") return "Blocked";
  return String(intent);
}

function IntentBadge({ intent, blocked }: { intent?: AgentIntent; blocked?: boolean }) {
  const kind = blocked ? "blocked" : intent;
  return (
    <span className={`intent-badge intent-${kind || "unknown"}`} title="Planner intent">
      {intentLabel(kind)}
    </span>
  );
}

function SourcesPanel({
  open,
  documents,
  onClose,
}: {
  open: boolean;
  documents: RetrievedDocument[];
  onClose: () => void;
}) {
  return (
    <aside className={`sources-panel ${open ? "open" : ""}`} aria-hidden={!open}>
      <div className="sources-head">
        <h2>Sources</h2>
        <button type="button" className="ghost-btn" onClick={onClose} aria-label="Close sources">
          Close
        </button>
      </div>
      {documents.length === 0 ? (
        <p className="sources-empty">No retrieved chunks for this turn.</p>
      ) : (
        <ul className="sources-list">
          {documents.map((doc, i) => {
            const title =
              doc.filename ||
              doc.source ||
              (doc.metadata && typeof doc.metadata.source === "string"
                ? doc.metadata.source
                : null) ||
              `Chunk ${doc.rank ?? i + 1}`;
            const score =
              typeof doc.score === "number" ? doc.score.toFixed(3) : null;
            return (
              <li key={`${title}-${i}`} className="source-item">
                <div className="source-meta">
                  <strong>{title}</strong>
                  {score ? <span className="score">{score}</span> : null}
                </div>
                <p>{(doc.text || "").slice(0, 420)}</p>
              </li>
            );
          })}
        </ul>
      )}
    </aside>
  );
}

export default function ChatApp() {
  const [sessionId, setSessionId] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [pending, startTransition] = useTransition();
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [activeDocs, setActiveDocs] = useState<RetrievedDocument[]>([]);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    setSessionId(newSessionId());
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, pending]);

  const hasTranscript = messages.length > 0;

  const lastAssistant = useMemo(
    () => [...messages].reverse().find((m) => m.role === "assistant"),
    [messages],
  );

  function startNewSession() {
    setSessionId(newSessionId());
    setMessages([]);
    setError(null);
    setActiveDocs([]);
    setSourcesOpen(false);
    setDraft("");
    inputRef.current?.focus();
  }

  function openSources(docs: RetrievedDocument[]) {
    setActiveDocs(docs);
    setSourcesOpen(true);
  }

  function submit() {
    const question = draft.trim();
    if (!question || pending || !sessionId) return;

    const userMsg: ChatMessage = {
      id: `u-${Date.now()}`,
      role: "user",
      content: question,
    };
    setMessages((prev) => [...prev, userMsg]);
    setDraft("");
    setError(null);

    startTransition(async () => {
      try {
        const res: QueryResponse = await postQuery({
          question,
          session_id: sessionId,
        });
        if (res.session_id && res.session_id !== sessionId) {
          setSessionId(res.session_id);
        }
        const assistant: ChatMessage = {
          id: `a-${Date.now()}`,
          role: "assistant",
          content: res.answer || (res.blocked ? "This request was blocked by guardrails." : ""),
          intent: res.intent,
          blocked: res.blocked,
          documents: res.documents || [],
          status: res.status,
          error: res.error,
        };
        setMessages((prev) => [...prev, assistant]);
        if ((res.documents || []).length > 0) {
          setActiveDocs(res.documents);
        }
      } catch (err) {
        const detail =
          err instanceof ApiError
            ? err.detail
            : err instanceof Error
              ? err.message
              : "Something went wrong";
        setError(detail);
        setMessages((prev) => [
          ...prev,
          {
            id: `e-${Date.now()}`,
            role: "assistant",
            content: "I couldn’t complete that turn. Check that the API is running, then try again.",
            intent: "unknown",
            blocked: false,
            error: detail,
          },
        ]);
      }
    });
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  return (
    <div className="shell">
      <div className="glow glow-a" aria-hidden />
      <div className="glow glow-b" aria-hidden />

      <header className="topbar">
        <div className="brand-block">
          <p className="brand">AskPod</p>
          <p className="tagline">Kubernetes knowledge, answered instantly</p>
        </div>
        <div className="top-actions">
          {lastAssistant?.intent ? (
            <IntentBadge intent={lastAssistant.intent} blocked={lastAssistant.blocked} />
          ) : null}
          <button type="button" className="ghost-btn" onClick={startNewSession}>
            New session
          </button>
        </div>
      </header>

      <div className={`workspace ${sourcesOpen ? "with-sources" : ""}`}>
        <main className="chat-pane">
          {!hasTranscript ? (
            <section className="hero-empty">
              <h1 className="brand-hero">AskPod</h1>
              <p className="hero-copy">
                Ask about your Kubernetes docs — or just say hello. Guardrails keep the room safe.
              </p>
            </section>
          ) : (
            <div className="transcript" role="log" aria-live="polite">
              {messages.map((m) => (
                <article
                  key={m.id}
                  className={`bubble ${m.role} ${m.blocked ? "blocked" : ""}`}
                >
                  <div className="bubble-meta">
                    <span className="who">{m.role === "user" ? "You" : "AskPod"}</span>
                    {m.role === "assistant" ? (
                      <IntentBadge intent={m.intent} blocked={m.blocked} />
                    ) : null}
                  </div>
                  <div className="bubble-body">
                    {m.role === "assistant" ? (
                      <MarkdownMessage content={m.content} />
                    ) : (
                      m.content
                    )}
                  </div>
                  {m.role === "assistant" && (m.documents?.length ?? 0) > 0 ? (
                    <button
                      type="button"
                      className="linkish"
                      onClick={() => openSources(m.documents || [])}
                    >
                      View {m.documents!.length} sources
                    </button>
                  ) : null}
                  {m.blocked ? (
                    <p className="blocked-note">Held by guardrails — this reply isn’t from the knowledge base.</p>
                  ) : null}
                </article>
              ))}
              {pending ? (
                <div className="bubble assistant pending">
                  <div className="bubble-meta">
                    <span className="who">AskPod</span>
                  </div>
                  <div className="typing">
                    <span />
                    <span />
                    <span />
                  </div>
                </div>
              ) : null}
              <div ref={bottomRef} />
            </div>
          )}

          <form
            className="composer"
            onSubmit={(e) => {
              e.preventDefault();
              submit();
            }}
          >
            {error ? <p className="form-error">{error}</p> : null}
            <textarea
              ref={inputRef}
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder="Ask AskPod anything… Shift+Enter for a new line"
              rows={2}
              disabled={pending}
              aria-label="Message"
            />
            <div className="composer-row">
              <span className="thread-hint">Session {sessionId.slice(0, 8)}…</span>
              <button type="submit" className="send-btn" disabled={pending || !draft.trim()}>
                {pending ? "Thinking…" : "Send"}
              </button>
            </div>
          </form>
        </main>

        <SourcesPanel
          open={sourcesOpen}
          documents={activeDocs}
          onClose={() => setSourcesOpen(false)}
        />
      </div>
    </div>
  );
}
