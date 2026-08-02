"""Planner node — route to conversational reply OR technical search rewrite.

Contract
--------
The planner must output **exactly one** of:

1. ``intent="conversational"`` + ``conversational_reply``
   Small-talk / meta / out-of-scope chat. No retrieval. The reply is the
   answer (graph can skip retriever and go straight to done/responder).

2. ``intent="technical"`` + ``rewritten_query``
   Knowledge question. Do **not** answer here — only rewrite the user
   question into a focused search query for Qdrant + Jina.

Nothing else (no mixed fields, no free-form prose as the sole output).
"""

from __future__ import annotations

from typing import Any, Literal

import logfire
from pydantic import BaseModel, Field, model_validator

from app.agents.state import AgentIntent, AgentState, PlanState
from app.ingestion.loaders.base import ensure_logfire

PlannerIntent = Literal["conversational", "technical"]


class PlannerDecision(BaseModel):
    """Strict planner output schema (LLM / rule engine must match this)."""

    intent: PlannerIntent = Field(
        description="conversational = reply only; technical = search rewrite only"
    )
    conversational_reply: str | None = Field(
        default=None,
        description="Required when intent=conversational. Direct reply to the user.",
    )
    rewritten_query: str | None = Field(
        default=None,
        description="Required when intent=technical. Search query for retrieval.",
    )
    rationale: str | None = Field(
        default=None,
        description="Optional short reason for logging / debugging (not shown to user).",
    )

    @model_validator(mode="after")
    def enforce_exclusive_contract(self) -> PlannerDecision:
        reply = (self.conversational_reply or "").strip() or None
        query = (self.rewritten_query or "").strip() or None
        self.conversational_reply = reply
        self.rewritten_query = query

        has_reply = reply is not None
        has_query = query is not None

        # Do NOT silently drop a field — that hides mixed LLM output
        # ("thanks, and also what's a DaemonSet?") behind a greeter reply.
        if has_reply == has_query:
            raise ValueError(
                "PlannerDecision XOR violated: set exactly one of "
                f"conversational_reply / rewritten_query "
                f"(got reply={has_reply}, query={has_query})"
            )

        expected: PlannerIntent = "conversational" if has_reply else "technical"
        if self.intent != expected:
            raise ValueError(
                f"PlannerDecision intent={self.intent!r} disagrees with filled "
                f"field (expected {expected!r}). Mixed/ambiguous outputs must be "
                "resolved to a single path before continuing."
            )
        return self

    def to_plan_state(self) -> PlanState:
        # PlanState re-checks XOR — double gate at the graph boundary.
        return PlanState(
            intent=self.intent,
            conversational_reply=self.conversational_reply,
            rewritten_query=self.rewritten_query,
            rationale=self.rationale,
        )


PLANNER_SYSTEM_PROMPT = """You are the planner for an enterprise RAG assistant.

Classify the user message and output JSON that matches this contract EXACTLY:

A) Conversational path — greetings, thanks, chit-chat, capability questions that
   do not need document retrieval, or clearly off-topic small talk:
   {
     "intent": "conversational",
     "conversational_reply": "<your direct reply to the user>",
     "rewritten_query": null,
     "rationale": "<optional>"
   }

B) Technical path — questions about company/docs/systems/Kubernetes/etc. that
   should be answered from the knowledge base:
   {
     "intent": "technical",
     "conversational_reply": null,
     "rewritten_query": "<concise search query rewritten for retrieval>",
     "rationale": "<optional>"
   }

Rules:
- Output ONLY one path. Never both a reply and a search query.
- On the technical path: do NOT answer the question. Only rewrite for search.
- On the conversational path: do NOT invent a search query.
- rewritten_query should be specific, keyword-rich, and free of filler.
- If the user mixes chit-chat AND a real question (e.g. "thanks, also what is a
  DaemonSet?"), choose the technical path only — put the knowledge question into
  rewritten_query and leave conversational_reply null. Do not answer with just
  "You're welcome!"
"""


def parse_planner_decision(data: dict[str, Any] | PlannerDecision) -> PlannerDecision:
    """Validate raw model JSON against the planner contract."""
    if isinstance(data, PlannerDecision):
        return data
    return PlannerDecision.model_validate(data)


def apply_planner_decision(state: AgentState, decision: PlannerDecision) -> dict[str, Any]:
    """Map a validated decision onto AgentState updates."""
    plan = decision.to_plan_state()
    intent: AgentIntent = decision.intent

    updates: dict[str, Any] = {
        "plan": plan,
        "intent": intent,
        "status": "planning",
        "error": None,
    }

    if decision.intent == "conversational":
        # Conversational path can short-circuit with the planner reply.
        updates["final_answer"] = decision.conversational_reply or ""
        updates["documents"] = []
        updates["status"] = "done"
    else:
        # Technical path: hand a rewritten query to the retriever.
        updates["current_query"] = decision.rewritten_query or state.get(
            "current_query", ""
        )
        updates["final_answer"] = ""
        updates["status"] = "retrieving"

    return updates


def planner_node(state: AgentState) -> dict[str, Any]:
    """Planner graph node.

    LLM wiring comes next. For now this is a deterministic stub that:
    - treats empty / pure greetings as conversational
    - mixed chit-chat + real question → technical (search rewrite)
    - otherwise emits a technical rewritten_query (= current_query stripped)

    Replace the stub body with an LLM call that returns PlannerDecision JSON.
    """
    ensure_logfire()
    query = (state.get("current_query") or "").strip()

    with logfire.span("agent.planner", query=query[:200]):
        try:
            decision = _stub_plan(query)
            updates = apply_planner_decision(state, decision)
            logfire.info(
                "Planner decision",
                intent=decision.intent,
                rewritten_query=decision.rewritten_query,
                has_reply=bool(decision.conversational_reply),
            )
            return updates
        except Exception as exc:  # noqa: BLE001 — surface contract failures on state
            logfire.error("Planner XOR/contract failure", error=str(exc))
            return {
                "status": "error",
                "error": str(exc),
                "plan": None,
                "intent": "unknown",
                "final_answer": "",
            }


_GREETING_PREFIXES = (
    "hi",
    "hello",
    "hey",
    "thanks",
    "thank you",
    "good morning",
    "good afternoon",
    "good evening",
    "who are you",
    "what can you do",
)

_TECH_HINTS = (
    "what",
    "how",
    "why",
    "when",
    "where",
    "which",
    "configure",
    "daemon",
    "pod",
    "job",
    "k8s",
    "kubernetes",
    "probe",
    "deploy",
    "also",
    "?",
)


def _is_pure_greeting(lowered: str) -> bool:
    """True only for bare greetings — not 'thanks, and also what's a DaemonSet?'."""
    for g in _GREETING_PREFIXES:
        if lowered == g or lowered in {f"{g}!", f"{g}?", f"{g}."}:
            return True
        if lowered.startswith(g):
            rest = lowered[len(g) :].lstrip(" ,!.?")
            if not rest:
                return True
            # Anything that looks like a real ask → not a pure greeting.
            if len(rest) >= 12 or any(h in rest for h in _TECH_HINTS):
                return False
            return len(rest) < 10
    return False


def _stub_plan(query: str) -> PlannerDecision:
    """Temporary rule-based planner until the LLM node is wired."""
    lowered = query.lower().strip()
    if not lowered:
        return PlannerDecision(
            intent="conversational",
            conversational_reply=(
                "Hi — ask me a technical question about your enterprise docs "
                "and I'll search the knowledge base."
            ),
            rationale="empty query",
        )

    if _is_pure_greeting(lowered):
        return PlannerDecision(
            intent="conversational",
            conversational_reply=(
                "Hello! I can search your enterprise knowledge base for "
                "technical answers. What do you want to look up?"
            ),
            rationale="greeting / meta",
        )

    return PlannerDecision(
        intent="technical",
        rewritten_query=query,
        rationale="stub: pass-through until LLM rewriter is connected",
    )
