"""Responder node — dual prompts for conversational vs grounded technical answers."""

from __future__ import annotations

from typing import Any, Sequence

import logfire
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.llm import get_chat_model
from app.agents.nodes.planner import _format_history_block, _message_text, build_dialogue
from app.agents.state import AgentState
from app.ingestion.loaders.base import ensure_logfire

FRIENDLY_SYSTEM_PROMPT = """You are a warm, concise enterprise assistant with conversation memory.

Reply naturally to the user. Use prior turns when helpful. Do not invent product
docs or run a fake knowledge-base lookup. Keep answers short unless the user
asks for detail. If they want technical depth from internal docs, invite them
to ask a concrete question.
"""

ARCHITECT_SYSTEM_PROMPT = """You are a senior platform / systems architect answering from
retrieved enterprise documentation.

Rules:
- Prefer the provided CONTEXT. If context is thin or missing, say what you know
  vs what you cannot verify from the docs.
- Be precise, structured, and practical (steps, caveats, failure modes).
- Cite sources by filename when you lean on a chunk.
- Do not invent APIs, flags, or cluster settings that are not supported by context.
- Resolve follow-ups using the conversation history + context together.
"""


def _format_context(documents: Sequence[Any]) -> str:
    if not documents:
        return "(no retrieved documents)"
    blocks: list[str] = []
    for raw in documents:
        doc = raw if isinstance(raw, dict) else (
            raw.model_dump() if hasattr(raw, "model_dump") else dict(raw)
        )
        label = doc.get("filename") or doc.get("source") or f"chunk-{doc.get('rank')}"
        score = doc.get("score")
        score_s = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
        blocks.append(
            f"[{doc.get('rank')}] {label} (score={score_s})\n{(doc.get('text') or '').strip()}"
        )
    return "\n\n---\n\n".join(blocks)


def _latest_user_text(state: AgentState) -> str:
    dialogue = build_dialogue(state)
    for msg in reversed(dialogue):
        text = _message_text(msg)
        if text and getattr(msg, "type", None) in {"human", "user"}:
            return text
    return (state.get("current_query") or "").strip()


def _friendly_reply(state: AgentState) -> str:
    """Conversational path — memory-aware, no retrieval required."""
    plan = state.get("plan") or {}
    draft = ""
    if isinstance(plan, dict):
        draft = (plan.get("conversational_reply") or "").strip()
    else:
        draft = (getattr(plan, "conversational_reply", None) or "").strip()

    try:
        llm = get_chat_model(temperature=0.4)
    except RuntimeError:
        return draft or state.get("final_answer") or "Hello — how can I help?"

    transcript = _format_history_block(build_dialogue(state))
    user_payload = (
        f"Conversation so far:\n{transcript}\n\n"
        f"Planner draft reply (optional to refine):\n{draft or '(none)'}\n\n"
        "Write the final user-facing reply."
    )
    raw = llm.invoke(
        [
            SystemMessage(content=FRIENDLY_SYSTEM_PROMPT),
            HumanMessage(content=user_payload),
        ]
    )
    text = _message_text(raw).strip()
    return text or draft or "Hello — how can I help?"


def _architect_reply(state: AgentState) -> str:
    """Technical path — senior architect grounded on retrieved docs."""
    docs = list(state.get("documents") or [])
    question = _latest_user_text(state)
    search_q = (state.get("current_query") or question).strip()
    context = _format_context(docs)

    try:
        llm = get_chat_model(temperature=0.2)
    except RuntimeError:
        # Offline fallback: return stitched context snippets.
        if not docs:
            return "I could not reach an LLM and have no retrieved context."
        bits = []
        for d in docs[:3]:
            row = d if isinstance(d, dict) else d.model_dump()
            bits.append(
                f"- {(row.get('filename') or row.get('source'))}: {(row.get('text') or '')[:240]}"
            )
        return "Retrieved context (LLM unavailable):\n" + "\n".join(bits)

    transcript = _format_history_block(build_dialogue(state))
    user_payload = (
        f"Conversation so far:\n{transcript}\n\n"
        f"User question:\n{question}\n\n"
        f"Search query used:\n{search_q}\n\n"
        f"CONTEXT:\n{context}\n\n"
        "Write the final architect-grade answer."
    )
    raw = llm.invoke(
        [
            SystemMessage(content=ARCHITECT_SYSTEM_PROMPT),
            HumanMessage(content=user_payload),
        ]
    )
    return _message_text(raw).strip() or "I could not produce an answer from the retrieved context."


def responder_node(state: AgentState) -> dict[str, Any]:
    """Produce final_answer and append an AI message to session history."""
    ensure_logfire()
    intent = state.get("intent") or "unknown"

    with logfire.span("agent.responder", intent=intent):
        try:
            if state.get("status") == "error" and state.get("error"):
                answer = (
                    "Sorry — I hit an internal error while handling that. "
                    f"({state.get('error')})"
                )
            elif intent == "technical":
                answer = _architect_reply(state)
            else:
                answer = _friendly_reply(state)

            logfire.info(
                "Responder complete",
                intent=intent,
                answer_chars=len(answer),
                docs=len(state.get("documents") or []),
            )
            return {
                "final_answer": answer,
                "status": "done",
                "messages": [AIMessage(content=answer)],
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            logfire.error("Responder failed", error=str(exc))
            return {
                "final_answer": "Sorry — I could not complete that response.",
                "status": "error",
                "error": f"responder failed: {exc}",
                "messages": [
                    AIMessage(content="Sorry — I could not complete that response.")
                ],
            }
