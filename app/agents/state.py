"""LangGraph agent state for enterprise RAG orchestration.

Persistence note
----------------
Long-term chat memory lives in Neon (``chat_messages`` by ``session_id``),
not in this TypedDict and not in a LangGraph checkpointer.

``AgentState`` is **ephemeral per request**:
- ``invoke_agent`` loads prior Neon messages → fills ``messages``
- the graph runs (planner / retriever / responder)
- only the new user + assistant texts are written back to Neon
- ``documents``, ``plan``, scores, etc. are throwaway turn data

``session_id`` is owned by the API / ``invoke_agent`` layer — it does not
need to live on AgentState for the current design.
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
    """Normalized retrieval hit carried through one graph turn."""

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
    """Planner decision for this turn — XOR contract enforced in code.

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

        if has_reply == has_query:
            raise ValueError(
                "PlanState XOR violated: set exactly one of "
                f"conversational_reply / rewritten_query "
                f"(got reply={has_reply}, query={has_query})"
            )

        expected: AgentIntent = "conversational" if has_reply else "technical"
        if self.intent not in (expected, "unknown"):
            raise ValueError(
                f"PlanState intent={self.intent!r} disagrees with filled field "
                f"(expected {expected!r})"
            )
        self.intent = expected
        return self


class AgentState(TypedDict):
    """In-memory working state for a single graph invocation."""

    # Hydrated from Neon + this turn's HumanMessage; add_messages merges
    # mid-graph AIMessage appends (planner/responder) within the same turn.
    messages: Annotated[list, add_messages]

    # Latest user question / rewritten search query for this turn.
    current_query: str

    # Retrieved docs for this turn only (not persisted).
    documents: list[dict[str, Any]]

    # Planner output for this turn only (not persisted).
    plan: dict[str, Any] | None

    status: AgentStatus
    intent: AgentIntent
    final_answer: str
    error: NotRequired[str | None]


def initial_agent_state(query: str = "") -> AgentState:
    """Empty in-memory state (tests / manual node calls)."""
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
