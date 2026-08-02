"""Smoke: Langfuse configured + one conversational agent turn traced."""

from app.config import get_settings

get_settings.cache_clear()

from app.config import settings
from app.observability.langfuse_tracing import (
    build_langfuse_config,
    configure_langfuse,
    flush_langfuse,
    get_langfuse_handler,
)
from app.agents.graph import invoke_agent

print("has_langfuse", settings.has_langfuse)
print("configured", configure_langfuse())
handler = get_langfuse_handler()
print("handler", type(handler).__name__ if handler else None)
cfg = build_langfuse_config(session_id="langfuse-smoke")
print("callbacks", bool(cfg.get("callbacks")), "session", cfg["metadata"].get("langfuse_session_id"))

result = invoke_agent("hi there", session_id=None)
print("intent", result.get("intent"), "status", result.get("status"))
print("answer", (result.get("final_answer") or "")[:120])
flush_langfuse()
print("flushed — check Langfuse UI traces for session", result.get("session_id"))
