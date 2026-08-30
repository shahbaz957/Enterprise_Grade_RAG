"""LangGraph orchestration — planner → (retriever?) → responder.

Chat memory lives in Neon (messages only), not in a LangGraph checkpointer.
Pass the same ``session_id`` on each request to continue a browser session.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

import logfire
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from app.agents.nodes.planner import planner_node
from app.agents.nodes.responder import responder_node
from app.agents.nodes.retriever import retriever_node
from app.agents.state import AgentState
from app.db.session_store import (
    append_message,
    get_or_create_session,
    load_messages,
)
from app.ingestion.loaders.base import ensure_logfire
from app.observability.langfuse_tracing import (
    build_langfuse_config,
    configure_langfuse,
    flush_langfuse,
)

RouteAfterPlanner = Literal["retriever", "responder", "__end__"]

# How many prior DB messages to hydrate into the graph (planner still windows).
HISTORY_LIMIT = 40


def _route_after_planner(state: AgentState) -> RouteAfterPlanner:
    if state.get("status") == "error":
        return END
    if (state.get("intent") or "") == "technical":
        return "retriever"
    return "responder"


@lru_cache(maxsize=1)
def get_compiled_graph():
    ensure_logfire()
    configure_langfuse()
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("responder", responder_node)
    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner",
        _route_after_planner,
        {
            "retriever": "retriever",
            "responder": "responder",
            END: END,
        },
    )
    graph.add_edge("retriever", "responder")
    graph.add_edge("responder", END)
    return graph.compile()


def _history_to_lc_messages(rows: list[dict[str, Any]]) -> list:
    out: list = []
    for row in rows:
        role = row.get("role")
        content = (row.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
    return out


def invoke_agent(
    question: str,
    *,
    session_id: str | None = None,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Run one user turn with Neon-backed session memory.

    ``session_id`` is preferred. ``thread_id`` is accepted as an alias for
    older clients.

    NeMo Guardrails gate: input rails (+ dialog Colang) run before the graph;
    output rails run on ``final_answer`` before Neon persistence.
    """
    ensure_logfire()
    configure_langfuse()
    q = (question or "").strip()
    if not q:
        raise ValueError("question must be non-empty")

    sid = get_or_create_session(session_id or thread_id)
    prior = load_messages(sid, limit=HISTORY_LIMIT)
    history = _history_to_lc_messages(prior)

    from app.guardrails.engine import RailDecisionKind, check_input, check_output

    with logfire.span("agent.invoke", session_id=sid, question=q[:200]):
        # --- Input rails (block / dialog canned reply before graph) ---
        inbound = check_input(q)
        if inbound.skip_agent:
            answer = inbound.content
            status = (
                "blocked"
                if inbound.kind == RailDecisionKind.BLOCKED
                else "guardrailed"
            )
            intent = (
                "blocked"
                if inbound.kind == RailDecisionKind.BLOCKED
                else "conversational"
            )
            append_message(sid, "user", q)
            if answer:
                append_message(sid, "assistant", answer)
            flush_langfuse()
            logfire.info(
                "Agent turn blocked/handled by input rails",
                session_id=sid,
                status=status,
                intent=intent,
                rail=inbound.rail,
                answer_chars=len(answer),
            )
            return {
                "messages": load_messages(sid, limit=HISTORY_LIMIT),
                "current_query": q,
                "documents": [],
                "plan": None,
                "status": status,
                "intent": intent,
                "final_answer": answer,
                "error": inbound.rail if inbound.kind == RailDecisionKind.BLOCKED else None,
                "session_id": sid,
                "thread_id": sid,
            }

        q = inbound.content  # may be unchanged; mask path would update here
        turn: dict[str, Any] = {
            "messages": [*history, HumanMessage(content=q)],
            "current_query": q,
            "documents": [],
            "plan": None,
            "status": "planning",
            "intent": "unknown",
            "final_answer": "",
            "error": None,
        }

        graph = get_compiled_graph()
        lf_config = build_langfuse_config(
            session_id=sid,
            tags=["enterprise-rag", "agent", "guardrails"],
        )
        result = graph.invoke(turn, config=lf_config)
        answer = (result.get("final_answer") or "").strip()

        # --- Output rails (mask / refuse unsafe final answers) ---
        if answer:
            outbound = check_output(q, answer)
            if outbound.kind == RailDecisionKind.BLOCKED:
                answer = outbound.content
                result = dict(result)
                result["final_answer"] = answer
                result["status"] = "blocked"
                result["error"] = outbound.rail or "output_rail"
            elif outbound.kind == RailDecisionKind.MODIFIED:
                answer = outbound.content
                result = dict(result)
                result["final_answer"] = answer

        # Persist only the chat turn — not docs/plan/scores.
        append_message(sid, "user", q)
        if answer:
            append_message(sid, "assistant", answer)

        flush_langfuse()

        logfire.info(
            "Agent turn complete",
            session_id=sid,
            intent=result.get("intent"),
            status=result.get("status"),
            docs=len(result.get("documents") or []),
            answer_chars=len(answer),
            history_loaded=len(prior),
            langfuse=bool(lf_config.get("callbacks")),
        )

        out = dict(result)
        out["session_id"] = sid
        out["thread_id"] = sid  # alias for older clients / UI
        out["messages"] = load_messages(sid, limit=HISTORY_LIMIT)
        return out
