"""Shared chat model factory for agent nodes (Groq first, OpenAI fallback)."""

from __future__ import annotations

from typing import Any, Sequence

from app.config import settings
from app.observability.langfuse_tracing import get_langfuse_handler


def get_chat_model(*, temperature: float = 0.2) -> Any:
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
    raise RuntimeError("No LLM configured (set GROQ_API_KEY or OPENAI_API_KEY)")


def invoke_chat(
    messages: Sequence[Any],
    *,
    temperature: float = 0.2,
    run_name: str | None = None,
) -> Any:
    """Invoke the chat model and attach Langfuse callbacks when configured."""
    llm = get_chat_model(temperature=temperature)
    config: dict[str, Any] = {}
    handler = get_langfuse_handler()
    if handler is not None:
        config["callbacks"] = [handler]
    if run_name:
        config["run_name"] = run_name
    if config:
        return llm.invoke(messages, config=config)
    return llm.invoke(messages)
