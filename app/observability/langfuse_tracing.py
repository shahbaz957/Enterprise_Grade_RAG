"""Langfuse tracing for the agent graph and LLM calls."""

from __future__ import annotations

import os
from typing import Any

from app.config import settings

_configured = False


def configure_langfuse() -> bool:
    """Push Langfuse keys into env (SDK / CallbackHandler read these)."""
    global _configured
    if not settings.has_langfuse:
        return False
    if _configured:
        return True

    os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key.strip().strip('"')
    os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key.strip().strip('"')
    os.environ["LANGFUSE_HOST"] = (
        settings.langfuse_base_url.strip().strip('"') or "https://cloud.langfuse.com"
    )
    _configured = True
    return True


def get_langfuse_handler() -> Any | None:
    """LangChain/LangGraph CallbackHandler, or None if Langfuse is disabled."""
    if not configure_langfuse():
        return None
    from langfuse.langchain import CallbackHandler

    return CallbackHandler()


def build_langfuse_config(
    *,
    session_id: str | None = None,
    run_name: str = "enterprise-rag-agent",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Invoke config for LangGraph / LangChain (callbacks + session metadata)."""
    config: dict[str, Any] = {
        "run_name": run_name,
        "tags": tags or ["enterprise-rag", "agent"],
        "metadata": {
            "langfuse_session_id": session_id,
            "app": settings.app_name,
            "version": settings.app_version,
        },
    }
    handler = get_langfuse_handler()
    if handler is not None:
        config["callbacks"] = [handler]
    return config


def flush_langfuse() -> None:
    """Force export of buffered traces (call after each agent turn)."""
    if not settings.has_langfuse:
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception:
        # Observability must never break the request path.
        pass
