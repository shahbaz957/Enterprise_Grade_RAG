from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import settings


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if settings.has_database:
        from app.db.session_store import ensure_schema

        ensure_schema()
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: str | None = Field(
        default=None,
        description="Browser session id. Omit to create a new session.",
    )
    thread_id: str | None = Field(
        default=None,
        description="Alias for session_id (back-compat).",
    )


class QueryResponse(BaseModel):
    answer: str
    session_id: str
    thread_id: str
    intent: str
    status: str
    documents: list[dict] = Field(default_factory=list)
    messages: list[dict] = Field(default_factory=list)
    error: str | None = None


class SessionResponse(BaseModel):
    session_id: str


class MessagesResponse(BaseModel):
    session_id: str
    messages: list[dict]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name}


@app.get("/ready")
async def ready() -> dict[str, object]:
    return {
        "status": "ready",
        "database": settings.has_database,
        "langfuse": settings.has_langfuse,
        "guardrails": settings.has_guardrails,
    }


@app.get("/metrics")
async def metrics() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/graph")
async def graph_info() -> dict[str, object]:
    return {
        "status": "ok",
        "entry": "planner",
        "nodes": ["planner", "retriever", "responder"],
        "memory": "neon_messages",
        "guardrails": {
            "enabled": settings.has_guardrails,
            "input": "before graph.invoke",
            "output": "on final_answer",
        },
        "routes": {
            "planner→retriever": "intent=technical",
            "planner→responder": "intent=conversational",
            "retriever→responder": "always",
        },
    }


@app.post("/sessions", response_model=SessionResponse)
async def create_chat_session() -> SessionResponse:
    """Call when the RAG UI first loads — store returned id in the browser."""
    if not settings.has_database:
        raise HTTPException(status_code=503, detail="DATABASE_URL not configured")
    from app.db.session_store import create_session

    return SessionResponse(session_id=create_session())


@app.get("/sessions/{session_id}/messages", response_model=MessagesResponse)
async def get_session_messages(session_id: str, limit: int = 100) -> MessagesResponse:
    """Reload transcript for the UI when the user returns."""
    if not settings.has_database:
        raise HTTPException(status_code=503, detail="DATABASE_URL not configured")
    from app.db.session_store import get_or_create_session, load_messages

    sid = get_or_create_session(session_id)
    return MessagesResponse(session_id=sid, messages=load_messages(sid, limit=limit))


@app.post("/query", response_model=QueryResponse)
async def query(payload: QueryRequest) -> QueryResponse:
    from app.agents.graph import invoke_agent

    result = invoke_agent(
        payload.question,
        session_id=payload.session_id or payload.thread_id,
    )
    docs = []
    for d in result.get("documents") or []:
        if hasattr(d, "model_dump"):
            docs.append(d.model_dump())
        elif isinstance(d, dict):
            docs.append(d)
    sid = str(result.get("session_id") or "")
    return QueryResponse(
        answer=result.get("final_answer") or "",
        session_id=sid,
        thread_id=sid,
        intent=str(result.get("intent") or "unknown"),
        status=str(result.get("status") or "unknown"),
        documents=docs,
        messages=list(result.get("messages") or []),
        error=result.get("error"),
    )
