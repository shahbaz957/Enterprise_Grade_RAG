"""Shared chat model factory for agent nodes.

Prefer Portkey when configured. Otherwise use USE_OPENAI_LLM (OpenAI vs Groq).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import logfire

from app.config import settings
from app.observability.langfuse_tracing import get_langfuse_handler


def is_portkey_config_error(exc: BaseException) -> bool:
    """True when Portkey rejects inline config (org ``block_inline_config``)."""
    msg = str(exc).lower()
    return "inline_config_blocked" in msg or "block_inline_config" in msg


def get_direct_chat_model(*, temperature: float = 0.2) -> Any:
    """Direct provider chat (bypasses Portkey). Controlled by USE_OPENAI_LLM."""
    if settings.use_openai_llm:
        if not settings.has_openai:
            raise RuntimeError("USE_OPENAI_LLM=true but OPENAI_API_KEY is unset")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_chat_model,
            api_key=settings.openai_api_key,
            temperature=temperature,
        )

    if not settings.has_groq:
        raise RuntimeError("USE_OPENAI_LLM=false but GROQ_API_KEY is unset")
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=settings.groq_chat_model,
        api_key=settings.active_groq_api_key,
        temperature=temperature,
    )


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
        has_vk = bool(
            settings.portkey_virtual_key_primary
            or settings.portkey_virtual_key
            or (settings.portkey_config_id or "").strip()
        )
        if not has_vk:
            return get_direct_chat_model(temperature=temperature)

        from app.gateway.portkey_client import get_portkey_chat_model

        return get_portkey_chat_model(
            temperature=temperature,
            user_id=user_id,
            route=route,
            feature=feature,
            extra_metadata=extra_metadata,
        )

    return get_direct_chat_model(temperature=temperature)


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
    """Invoke the chat model; on Portkey inline_config_blocked, retry direct LLM."""
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

    try:
        if config:
            return llm.invoke(messages, config=config)
        return llm.invoke(messages)
    except Exception as exc:
        if settings.has_portkey and is_portkey_config_error(exc):
            logfire.warn(
                "Portkey blocked inline config; falling back to direct LLM. "
                "Set PORTKEY_CONFIG_ID=pc-...",
                error=str(exc)[:240],
            )
            direct = get_direct_chat_model(temperature=temperature)
            if config:
                return direct.invoke(messages, config=config)
            return direct.invoke(messages)
        raise
