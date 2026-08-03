"""Shared chat model factory for agent nodes.

Prefer Portkey gateway (virtual keys, fallback/loadbalance, cache, retries,
timeouts, metadata/cost logs). Fall back to direct ChatGroq / ChatOpenAI when
Portkey is not configured.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from app.config import settings
from app.observability.langfuse_tracing import get_langfuse_handler


def get_chat_model(
    *,
    temperature: float = 0.2,
    user_id: str | None = None,
    route: str | None = None,
    feature: str | None = None,
    extra_metadata: Optional[dict[str, Any]] = None,
) -> Any:
    """Return a LangChain chat model (Portkey-wrapped when configured)."""
    if settings.has_portkey:
        from app.gateway.portkey_client import get_portkey_chat_model

        return get_portkey_chat_model(
            temperature=temperature,
            user_id=user_id,
            route=route,
            feature=feature,
            extra_metadata=extra_metadata,
        )

    if settings.has_groq:
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.groq_chat_model,
            api_key=settings.active_groq_api_key,
            temperature=temperature,
        )
    if settings.has_openai:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_chat_model,
            api_key=settings.openai_api_key,
            temperature=temperature,
        )
    raise RuntimeError(
        "No LLM configured (set PORTKEY_API_KEY + virtual keys, "
        "or GROQ_API_KEY / OPENAI_API_KEY)"
    )


def invoke_chat(
    messages: Sequence[Any],
    *,
    temperature: float = 0.2,
    run_name: str | None = None,
    user_id: str | None = None,
    route: str | None = None,
    feature: str | None = None,
    extra_metadata: Optional[dict[str, Any]] = None,
) -> Any:
    """Invoke the chat model and attach Langfuse callbacks when configured."""
    llm = get_chat_model(
        temperature=temperature,
        user_id=user_id,
        route=route or run_name or "llm",
        feature=feature or "agent_chat",
        extra_metadata=extra_metadata,
    )
    config: dict[str, Any] = {}
    handler = get_langfuse_handler()
    if handler is not None:
        config["callbacks"] = [handler]
    if run_name:
        config["run_name"] = run_name
    if config:
        return llm.invoke(messages, config=config)
    return llm.invoke(messages)
