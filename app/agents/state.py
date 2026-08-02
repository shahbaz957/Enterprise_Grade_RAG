"""LangGraph agent state for enterprise RAG orchestration.

`messages` uses LangGraph's `add_messages` reducer so each node can append
without wiping prior turns. All other fields are last-write-wins (replaced
when a node returns them).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, model_validator


AgentStatus = Literal[
    "idle",
    "planning",
    "retrieving",
    "responding",
    "done",
    "error",
]

AgentIntent = Literal["conversational", "technical", "unknown"]


class RetrievedDocument(BaseModel):
    """Normalized retrieval hit carried through the graph."""

    text: str
    score: float | None = None
    vector_score: float | None = None
    rank: int | None = None
    source: str | None = None
    filename: str | None = None
    doc_type: str | None = None
    corpus: str | None = None
    chunk_index: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlanState(BaseModel):
    """Planner decision on the graph — XOR contract is enforced in code.

    Exactly one of ``conversational_reply`` / ``rewritten_query`` must be set
    (non-empty). Never both, never neither. ``intent`` is aligned to whichever
    field is present so routing cannot disagree with the payload.
    """

    intent: AgentIntent = "unknown"
    conversational_reply: str | None = None
    rewritten_query: str | None = None
    rationale: str | None = None

    @model_validator(mode="after")
    def enforce_xor(self) -> PlanState:
        reply = (self.conversational_reply or "").strip() or None
        query = (self.rewritten_query or "").strip() or None
        self.conversational_reply = reply
        self.rewritten_query = query

        has_reply = reply is not None
        has_query = query is not None

        # both True or both False → illegal
        if has_reply == has_query:
            raise ValueError(
                "PlanState XOR violated: set exactly one of "
                f"conversational_reply / rewritten_query "
                f"(got reply={has_reply}, query={has_query})"
            )

        # Intent must match the filled branch (prevents conversational + leftover query).
        expected: AgentIntent = "conversational" if has_reply else "technical"
        if self.intent not in (expected, "unknown"):
            raise ValueError(
                f"PlanState intent={self.intent!r} disagrees with filled field "
                f"(expected {expected!r})"
            )
        self.intent = expected
        return self


class AgentState(TypedDict):
    """Shared state across planner → retriever → responder."""

    # Chat history — reducer appends / merges message ids.
    messages: Annotated[list, add_messages]

    # Latest user question (raw).
    current_query: str

    # Retrieved / reranked docs for the technical path.
    documents: list[RetrievedDocument]

    # Planner output (conversational reply XOR rewritten query).
    plan: PlanState | None

    # Pipeline lifecycle flag.
    status: AgentStatus

    # Shortcut of plan.intent for routing / UI.
    intent: AgentIntent

    # Final user-facing answer (conversational reply or grounded response).
    final_answer: str

    # Optional error detail when status == "error".
    error: NotRequired[str | None]


def initial_agent_state(query: str = "") -> AgentState:
    """Fresh state for a new graph invocation."""
    return {
        "messages": [],
        "current_query": query,
        "documents": [],
        "plan": None,
        "status": "idle",
        "intent": "unknown",
        "final_answer": "",
        "error": None,
    }
