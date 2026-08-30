"""Responder node — planner passthrough (conversational) or grounded technical answers."""

from __future__ import annotations

from typing import Any, Sequence

import logfire
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.dialogue import build_dialogue, format_history_block, message_text
from app.agents.llm import invoke_chat
from app.agents.state import AgentState
from app.ingestion.loaders.base import ensure_logfire

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
        text = message_text(msg)
        if text and getattr(msg, "type", None) in {"human", "user"}:
            return text
    return (state.get("current_query") or "").strip()


def _planner_conversational_reply(state: AgentState) -> str:
    """Reply already produced by the planner on the conversational path."""
    plan = state.get("plan") or {}
    if isinstance(plan, dict):
        reply = (plan.get("conversational_reply") or "").strip()
    else:
        reply = (getattr(plan, "conversational_reply", None) or "").strip()
    if reply:
        return reply
    return (state.get("final_answer") or "").strip()


def _conversational_answer(state: AgentState) -> str:
    """Return planner reply directly — avoids a second LLM call."""
    reply = _planner_conversational_reply(state)
    if reply:
        logfire.info("Using planner conversational reply (skipped responder LLM)")
        return reply
    return "Hello — how can I help?"


def _architect_reply(state: AgentState) -> str:
    """Technical path — senior architect grounded on retrieved docs."""
    docs = list(state.get("documents") or [])
    question = _latest_user_text(state)
    search_q = (state.get("current_query") or question).strip()
    context = _format_context(docs)

    transcript = format_history_block(build_dialogue(state))
    user_payload = (
        f"Conversation so far:\n{transcript}\n\n"
        f"User question:\n{question}\n\n"
        f"Search query used:\n{search_q}\n\n"
        f"CONTEXT:\n{context}\n\n"
        "Write the final architect-grade answer."
    )
    try:
        raw = invoke_chat(
            [
                SystemMessage(content=ARCHITECT_SYSTEM_PROMPT),
                HumanMessage(content=user_payload),
            ],
            temperature=0.2,
            run_name="agent.responder.architect",
            route="agent.responder",
            feature="responder_architect",
        )
    except RuntimeError:
        if not docs:
            return "I could not reach an LLM and have no retrieved context."
        bits = []
        for d in docs[:3]:
            row = d if isinstance(d, dict) else d.model_dump()
            bits.append(
                f"- {(row.get('filename') or row.get('source'))}: {(row.get('text') or '')[:240]}"
            )
        return "Retrieved context (LLM unavailable):\n" + "\n".join(bits)

    return message_text(raw).strip() or "I could not produce an answer from the retrieved context."


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
                answer = _conversational_answer(state)

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
