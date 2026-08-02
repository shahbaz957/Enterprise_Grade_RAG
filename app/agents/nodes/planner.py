"""Planner node — history-aware route: conversational reply XOR search rewrite.

Uses chat history so follow-ups resolve correctly, e.g.:

    User: "What is a DaemonSet?"
    Assistant: [explains…]
    User: "how do I monitor it?"
    → rewritten_query ≈ "how to monitor Kubernetes DaemonSet"

Contract (XOR) is enforced by ``PlannerDecision`` / ``PlanState``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal, Sequence

import logfire
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError, model_validator

from app.agents.llm import get_chat_model
from app.agents.state import AgentIntent, AgentState, PlanState
from app.config import settings
from app.ingestion.loaders.base import ensure_logfire
from app.observability.langfuse_tracing import get_langfuse_handler

PlannerIntent = Literal["conversational", "technical"]

_MAX_HISTORY_MESSAGES = 12  # recent turns only — keeps latency/cost down


class PlannerDecision(BaseModel):
    """Strict planner output schema (LLM must match this)."""

    intent: PlannerIntent = Field(
        description="conversational = reply only; technical = search rewrite only"
    )
    conversational_reply: str | None = Field(
        default=None,
        description="Required when intent=conversational. Direct reply to the user.",
    )
    rewritten_query: str | None = Field(
        default=None,
        description="Required when intent=technical. Standalone search query.",
    )
    rationale: str | None = Field(
        default=None,
        description="Optional short reason for logging (not shown to user).",
    )

    @model_validator(mode="after")
    def enforce_exclusive_contract(self) -> PlannerDecision:
        reply = (self.conversational_reply or "").strip() or None
        query = (self.rewritten_query or "").strip() or None
        self.conversational_reply = reply
        self.rewritten_query = query

        has_reply = reply is not None
        has_query = query is not None

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
        return PlanState(
            intent=self.intent,
            conversational_reply=self.conversational_reply,
            rewritten_query=self.rewritten_query,
            rationale=self.rationale,
        )


PLANNER_SYSTEM_PROMPT = """You are the planner for an enterprise RAG assistant.

You see the FULL conversation so far. Use it — especially for short follow-ups
that rely on pronouns or omitted subjects ("it", "that", "the same thing",
"how do I monitor it?", "and the probes?").

Classify the LATEST user message and output JSON that matches this contract EXACTLY:

A) Conversational — greetings, thanks-only, chit-chat, capability questions that
   need no document retrieval:
   {
     "intent": "conversational",
     "conversational_reply": "<direct reply>",
     "rewritten_query": null,
     "rationale": "<optional>"
   }

B) Technical — anything that should hit the knowledge base (including follow-ups
   that only make sense with prior turns):
   {
     "intent": "technical",
     "conversational_reply": null,
     "rewritten_query": "<standalone search query>",
     "rationale": "<optional>"
   }

History rules (critical for UX):
- If the user asks a follow-up like "how do I monitor it?" after discussing
  DaemonSets, rewritten_query MUST name the subject explicitly, e.g.
  "how to monitor Kubernetes DaemonSet" — never leave "it" unresolved.
- Rewrite into a self-contained, keyword-rich query a search engine could run
  with NO chat history.
- Prefer the most recent concrete topic in the thread when pronouns are ambiguous.

XOR rules:
- Output ONLY one path. Never both a reply and a search query.
- Technical path: do NOT answer the question — only rewrite for search.
- Conversational path: do NOT invent a search query.
- Mixed chit-chat + real question ("thanks, also what is a DaemonSet?") →
  technical path only (put the knowledge ask in rewritten_query).

Output raw JSON only — no markdown fences.
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
        "plan": plan.model_dump(),
        "intent": intent,
        "status": "planning",
        "error": None,
    }

    if decision.intent == "conversational":
        updates["final_answer"] = decision.conversational_reply or ""
        updates["documents"] = []
        updates["status"] = "done"
        updates["messages"] = [AIMessage(content=updates["final_answer"])]
    else:
        updates["current_query"] = decision.rewritten_query or state.get(
            "current_query", ""
        )
        updates["final_answer"] = ""
        updates["status"] = "retrieving"

    return updates


def _message_text(message: Any) -> str:
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


def _is_human(message: Any) -> bool:
    role = getattr(message, "type", None) or getattr(message, "role", None)
    return role in {"human", "user"} or isinstance(message, HumanMessage)


def _is_ai(message: Any) -> bool:
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

    # Keep a short window of recent turns.
    if len(history) > _MAX_HISTORY_MESSAGES:
        history = history[-_MAX_HISTORY_MESSAGES:]

    query = (state.get("current_query") or "").strip()
    if query:
        if not history or not _is_human(history[-1]) or _message_text(history[-1]) != query:
            history = [*history, HumanMessage(content=query)]
    return history


def _format_history_block(dialogue: Sequence[BaseMessage]) -> str:
    """Readable transcript for the planner (explicit roles)."""
    lines: list[str] = []
    for msg in dialogue:
        text = _message_text(msg)
        if not text:
            continue
        if _is_human(msg):
            lines.append(f"User: {text}")
        elif _is_ai(msg):
            lines.append(f"Assistant: {text}")
        else:
            lines.append(f"Other: {text}")
    return "\n".join(lines) if lines else "(no prior turns)"


def _extract_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Planner returned non-JSON output: {raw[:300]!r}")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("Planner JSON root must be an object")
    return data


def _llm_plan(dialogue: Sequence[BaseMessage]) -> PlannerDecision:
    """Call the chat model with history + enforce XOR via PlannerDecision."""
    llm = get_chat_model(temperature=0.1)
    transcript = _format_history_block(dialogue)
    latest = _message_text(dialogue[-1]) if dialogue else ""

    user_payload = (
        f"Conversation so far:\n{transcript}\n\n"
        f"Latest user message:\n{latest}\n\n"
        "Decide the path for the latest user message. "
        "If it is a follow-up, resolve pronouns using earlier turns."
    )
    prompt = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=user_payload),
    ]
    call_config: dict[str, Any] = {"run_name": "agent.planner.llm"}
    handler = get_langfuse_handler()
    if handler is not None:
        call_config["callbacks"] = [handler]

    # Prefer native structured output when the provider supports it.
    try:
        structured = llm.with_structured_output(PlannerDecision)
        result = structured.invoke(prompt, config=call_config)
        if isinstance(result, PlannerDecision):
            return result
        if isinstance(result, dict):
            return parse_planner_decision(result)
    except Exception as exc:  # noqa: BLE001 — fall through to JSON parse path
        logfire.warn("structured_output unavailable; using JSON prompt", error=str(exc))

    raw = llm.invoke(prompt, config=call_config)
    content = _message_text(raw)
    return parse_planner_decision(_extract_json_object(content))


def planner_node(state: AgentState) -> dict[str, Any]:
    """History-aware planner graph node."""
    ensure_logfire()
    dialogue = build_dialogue(state)
    query = (state.get("current_query") or "").strip()
    if not query and dialogue:
        query = _message_text(dialogue[-1])

    with logfire.span(
        "agent.planner",
        query=query[:200],
        history_turns=len(dialogue),
    ):
        try:
            if not query:
                decision = PlannerDecision(
                    intent="conversational",
                    conversational_reply=(
                        "Hi — ask me a technical question about your enterprise "
                        "docs and I'll search the knowledge base."
                    ),
                    rationale="empty query",
                )
            elif settings.has_groq or settings.has_openai:
                decision = _llm_plan(dialogue)
            else:
                logfire.warn("No LLM keys — using rule stub planner")
                decision = _stub_plan(query, dialogue)

            updates = apply_planner_decision(state, decision)
            logfire.info(
                "Planner decision",
                intent=decision.intent,
                rewritten_query=decision.rewritten_query,
                has_reply=bool(decision.conversational_reply),
                history_turns=len(dialogue),
            )
            return updates
        except (ValidationError, ValueError, RuntimeError) as exc:
            logfire.error("Planner failure", error=str(exc))
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
    "monitor",
    "also",
    "?",
)


def _is_pure_greeting(lowered: str) -> bool:
    for g in _GREETING_PREFIXES:
        if lowered == g or lowered in {f"{g}!", f"{g}?", f"{g}."}:
            return True
        if lowered.startswith(g):
            rest = lowered[len(g) :].lstrip(" ,!.?")
            if not rest:
                return True
            if len(rest) >= 12 or any(h in rest for h in _TECH_HINTS):
                return False
            return len(rest) < 10
    return False


def _stub_plan(query: str, dialogue: Sequence[BaseMessage] = ()) -> PlannerDecision:
    """Offline fallback when no LLM keys are configured."""
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

    # Cheap history hint for follow-ups when LLM is unavailable.
    prior_topics: list[str] = []
    for msg in dialogue:
        text = _message_text(msg)
        if _is_human(msg) and text and text.lower() != lowered:
            prior_topics.append(text)

    if prior_topics and any(
        token in lowered for token in ("it", "that", "this", "them", "those")
    ):
        rewritten = f"{query} (context: {prior_topics[-1]})"
        return PlannerDecision(
            intent="technical",
            rewritten_query=rewritten,
            rationale="stub: pronoun follow-up with prior user turn",
        )

    return PlannerDecision(
        intent="technical",
        rewritten_query=query,
        rationale="stub: pass-through",
    )
