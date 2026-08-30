"""Shared chat-history helpers for planner and responder nodes."""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.agents.state import AgentState

MAX_HISTORY_MESSAGES = 12


def message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(p for p in parts if p).strip()
    return str(content or "").strip()


def is_human(message: Any) -> bool:
    role = getattr(message, "type", None) or getattr(message, "role", None)
    return role in {"human", "user"} or isinstance(message, HumanMessage)


def is_ai(message: Any) -> bool:
    role = getattr(message, "type", None) or getattr(message, "role", None)
    return role in {"ai", "assistant"} or isinstance(message, AIMessage)


def build_dialogue(state: AgentState) -> list[BaseMessage]:
    """Prior turns + latest user query (history-aware input for the LLM)."""
    history: list[BaseMessage] = []
    for msg in state.get("messages") or []:
        if isinstance(msg, BaseMessage):
            history.append(msg)
        elif isinstance(msg, dict):
            role = msg.get("role") or msg.get("type")
            text = str(msg.get("content") or "").strip()
            if not text:
                continue
            if role in {"human", "user"}:
                history.append(HumanMessage(content=text))
            elif role in {"ai", "assistant"}:
                history.append(AIMessage(content=text))

    if len(history) > MAX_HISTORY_MESSAGES:
        history = history[-MAX_HISTORY_MESSAGES:]

    query = (state.get("current_query") or "").strip()
    if query:
        if not history or not is_human(history[-1]) or message_text(history[-1]) != query:
            history = [*history, HumanMessage(content=query)]
    return history


def format_history_block(dialogue: Sequence[BaseMessage]) -> str:
    """Readable transcript with explicit roles."""
    lines: list[str] = []
    for msg in dialogue:
        text = message_text(msg)
        if not text:
            continue
        if is_human(msg):
            lines.append(f"User: {text}")
        elif is_ai(msg):
            lines.append(f"Assistant: {text}")
        else:
            lines.append(f"Other: {text}")
    return "\n".join(lines) if lines else "(no prior turns)"
