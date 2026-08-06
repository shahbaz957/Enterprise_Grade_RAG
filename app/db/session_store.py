"""Neon/Postgres chat session store — messages only (not full agent state).

Why not LangGraph checkpointer?
--------------------------------
Checkpointers snapshot the whole AgentState (docs, plan, scores, …). For long
chats that bloats storage and reloads junk the next turn does not need.
We only persist ``user`` / ``assistant`` messages keyed by ``session_id``.
Each request: load recent messages → run graph (stateless) → append new turn.

If Neon is unreachable (DNS / network), we degrade to ephemeral in-memory
sessions so ``/query`` still works without a 500.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, Literal
from uuid import uuid4

import logfire
import psycopg
from psycopg.rows import dict_row

from app.config import settings

Role = Literal["user", "assistant"]

_SCHEMA_READY = False
_DB_UNAVAILABLE = False

# Ephemeral fallback when Neon cannot be reached (this process only).
_memory_sessions: set[str] = set()
_memory_messages: dict[str, list[dict[str, Any]]] = {}

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    id UUID PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created
    ON chat_messages (session_id, created_at);
"""


def _dsn() -> str:
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is not set")
    return settings.database_url


def _mark_db_down(exc: BaseException) -> None:
    global _DB_UNAVAILABLE, _SCHEMA_READY
    _DB_UNAVAILABLE = True
    _SCHEMA_READY = False
    logfire.warn(
        "Neon/Postgres unreachable — using in-memory chat sessions",
        error=str(exc)[:240],
    )


def _use_memory() -> bool:
    return _DB_UNAVAILABLE or not settings.has_database


@contextmanager
def _conn() -> Iterator[psycopg.Connection]:
    with psycopg.connect(_dsn(), row_factory=dict_row, connect_timeout=5) as conn:
        yield conn


def ensure_schema() -> None:
    """Idempotent DDL — safe to call on startup / first request."""
    global _SCHEMA_READY, _DB_UNAVAILABLE
    if not settings.has_database:
        return
    if _SCHEMA_READY:
        return
    try:
        with _conn() as conn:
            conn.execute(_SCHEMA_SQL)
            conn.commit()
        _SCHEMA_READY = True
        _DB_UNAVAILABLE = False
        logfire.info("Chat session schema ready")
    except Exception as exc:  # noqa: BLE001 — degrade, do not crash API startup
        _mark_db_down(exc)


def create_session(session_id: str | None = None) -> str:
    """Create a new chat session; returns UUID string."""
    sid = (session_id or "").strip() or str(uuid4())
    if _use_memory():
        _memory_sessions.add(sid)
        _memory_messages.setdefault(sid, [])
        return sid

    ensure_schema()
    if _use_memory():
        _memory_sessions.add(sid)
        _memory_messages.setdefault(sid, [])
        return sid

    try:
        with _conn() as conn:
            conn.execute(
                """
                INSERT INTO chat_sessions (id)
                VALUES (%s::uuid)
                ON CONFLICT (id) DO NOTHING
                """,
                (sid,),
            )
            conn.commit()
        return sid
    except Exception as exc:  # noqa: BLE001
        _mark_db_down(exc)
        _memory_sessions.add(sid)
        _memory_messages.setdefault(sid, [])
        return sid


def get_or_create_session(session_id: str | None) -> str:
    """Reuse ``session_id`` if present; otherwise mint a new session."""
    sid = (session_id or "").strip()
    if not sid:
        return create_session()

    if _use_memory():
        _memory_sessions.add(sid)
        _memory_messages.setdefault(sid, [])
        return sid

    ensure_schema()
    if _use_memory():
        _memory_sessions.add(sid)
        _memory_messages.setdefault(sid, [])
        return sid

    try:
        with _conn() as conn:
            row = conn.execute(
                "SELECT id FROM chat_sessions WHERE id = %s::uuid",
                (sid,),
            ).fetchone()
            if row:
                return str(row["id"])
        return create_session(sid)
    except Exception as exc:  # noqa: BLE001
        # Invalid UUID or DB down — prefer memory reuse of client thread_id.
        if "invalid input syntax for type uuid" in str(exc).lower():
            logfire.warn("Invalid session_id; creating fresh session", session_id=sid)
            return create_session()
        _mark_db_down(exc)
        _memory_sessions.add(sid)
        _memory_messages.setdefault(sid, [])
        return sid


def append_message(session_id: str, role: Role, content: str) -> dict[str, Any]:
    """Append one message; returns the stored row."""
    text = (content or "").strip()
    if not text:
        raise ValueError("message content must be non-empty")
    if role not in ("user", "assistant"):
        raise ValueError(f"invalid role: {role}")

    if _use_memory():
        row = {
            "id": len(_memory_messages.get(session_id, [])) + 1,
            "session_id": session_id,
            "role": role,
            "content": text,
            "created_at": datetime.now(timezone.utc),
        }
        _memory_messages.setdefault(session_id, []).append(row)
        return _serialize_message(row)

    ensure_schema()
    if _use_memory():
        return append_message(session_id, role, content)

    try:
        with _conn() as conn:
            row = conn.execute(
                """
                INSERT INTO chat_messages (session_id, role, content)
                VALUES (%s::uuid, %s, %s)
                RETURNING id, session_id, role, content, created_at
                """,
                (session_id, role, text),
            ).fetchone()
            conn.execute(
                "UPDATE chat_sessions SET updated_at = NOW() WHERE id = %s::uuid",
                (session_id,),
            )
            conn.commit()
        assert row is not None
        return _serialize_message(row)
    except Exception as exc:  # noqa: BLE001
        _mark_db_down(exc)
        return append_message(session_id, role, content)


def load_messages(
    session_id: str,
    *,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Load newest-N messages in chronological order for the UI / planner."""
    limit = max(1, min(int(limit), 200))

    if _use_memory():
        msgs = _memory_messages.get(session_id, [])
        return [_serialize_message(m) for m in msgs[-limit:]]

    ensure_schema()
    if _use_memory():
        return load_messages(session_id, limit=limit)

    try:
        with _conn() as conn:
            rows = conn.execute(
                """
                SELECT id, session_id, role, content, created_at
                FROM (
                    SELECT id, session_id, role, content, created_at
                    FROM chat_messages
                    WHERE session_id = %s::uuid
                    ORDER BY created_at DESC, id DESC
                    LIMIT %s
                ) recent
                ORDER BY created_at ASC, id ASC
                """,
                (session_id, limit),
            ).fetchall()
        return [_serialize_message(r) for r in rows]
    except Exception as exc:  # noqa: BLE001
        _mark_db_down(exc)
        return load_messages(session_id, limit=limit)


def _serialize_message(row: dict[str, Any]) -> dict[str, Any]:
    created = row.get("created_at")
    if isinstance(created, datetime):
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        created_s = created.isoformat()
    else:
        created_s = str(created) if created else None
    return {
        "id": row["id"],
        "session_id": str(row["session_id"]),
        "role": row["role"],
        "content": row["content"],
        "created_at": created_s,
    }
