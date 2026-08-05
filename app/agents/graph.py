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

RouteAfterPlanner = Literal["retriever", "responder", "__end__"]

# How many prior DB messages to hydrate into the graph (planner still windows).
HISTORY_LIMIT = 40


def _route_after_planner(state: AgentState) -> RouteAfterPlanner:
    if state.get("status") == "error":
        return END
    if (state.get("intent") or "") == "technical":
        return "retriever"
    return "responder"


def build_graph():
    """Compile a stateless graph (no checkpointer)."""
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


@lru_cache(maxsize=1)
def get_compiled_graph():
    ensure_logfire()
    return build_graph()


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
    """
    ensure_logfire()
    q = (question or "").strip()
    if not q:
        raise ValueError("question must be non-empty")

    sid = get_or_create_session(session_id or thread_id)
    prior = load_messages(sid, limit=HISTORY_LIMIT)
    history = _history_to_lc_messages(prior)

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
    with logfire.span("agent.invoke", session_id=sid, question=q[:200]):
        result = graph.invoke(turn)
        answer = (result.get("final_answer") or "").strip()

        # Persist only the chat turn — not docs/plan/scores.
        append_message(sid, "user", q)
        if answer:
            append_message(sid, "assistant", answer)

        logfire.info(
            "Agent turn complete",
            session_id=sid,
            intent=result.get("intent"),
            status=result.get("status"),
            docs=len(result.get("documents") or []),
            answer_chars=len(answer),
            history_loaded=len(prior),
        )

        out = dict(result)
        out["session_id"] = sid
        out["thread_id"] = sid  # alias for older clients / UI
        out["messages"] = load_messages(sid, limit=HISTORY_LIMIT)
        return out
