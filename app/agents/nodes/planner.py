"""Planner node — history-aware route: conversational reply XOR search rewrite.

Uses chat history so follow-ups resolve correctly, e.g.:

    User: "What is a DaemonSet?"
    Assistant: [explains…]
    User: "how do I monitor it?"
    → rewritten_query ≈ "how to monitor Kubernetes DaemonSet"

Contract (XOR) is enforced by ``PlanState``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

import logfire
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from app.agents.dialogue import (
    build_dialogue,
    format_history_block,
    is_human,
    message_text,
)
from app.agents.llm import (
    get_chat_model,
    is_portkey_config_error,
    with_portkey_fallback,
)
from app.agents.state import AgentIntent, AgentState, PlanState
from app.config import settings
from app.ingestion.loaders.base import ensure_logfire
from app.observability.langfuse_tracing import get_langfuse_handler

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


def parse_planner_decision(data: dict[str, Any] | PlanState) -> PlanState:
    """Validate raw model JSON against the planner contract."""
    if isinstance(data, PlanState):
        return data
    return PlanState.model_validate(data)


def apply_planner_decision(state: AgentState, plan: PlanState) -> dict[str, Any]:
    """Map a validated plan onto AgentState updates."""
    intent: AgentIntent = plan.intent

    updates: dict[str, Any] = {
        "plan": plan.model_dump(),
        "intent": intent,
        "status": "planning",
        "error": None,
    }

    if plan.intent == "conversational":
        updates["final_answer"] = plan.conversational_reply or ""
        updates["documents"] = []
        updates["status"] = "done"
        updates["messages"] = [AIMessage(content=updates["final_answer"])]
    else:
        updates["current_query"] = plan.rewritten_query or state.get("current_query", "")
        updates["final_answer"] = ""
        updates["status"] = "retrieving"

    return updates


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


def _llm_plan_with_model(
    llm: Any,
    prompt: list[BaseMessage],
    call_config: dict[str, Any],
) -> PlanState:
    """Run planner against one chat model (structured, then JSON fallback)."""
    try:
        structured = llm.with_structured_output(PlanState)
        result = structured.invoke(prompt, config=call_config)
        if isinstance(result, PlanState):
            return result
        if isinstance(result, dict):
            return parse_planner_decision(result)
    except Exception as exc:  # noqa: BLE001 — fall through to JSON parse path
        if is_portkey_config_error(exc):
            raise
        logfire.warn("structured_output unavailable; using JSON prompt", error=str(exc))

    raw = llm.invoke(prompt, config=call_config)
    content = message_text(raw)
    return parse_planner_decision(_extract_json_object(content))


def _llm_plan(dialogue: Sequence[BaseMessage]) -> PlanState:
    """Call the chat model with history + enforce XOR via PlanState."""
    transcript = format_history_block(dialogue)
    latest = message_text(dialogue[-1]) if dialogue else ""

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

    return with_portkey_fallback(
        lambda llm: _llm_plan_with_model(llm, prompt, call_config),
        temperature=0.1,
        route="agent.planner",
        feature="planner",
    )


def planner_node(state: AgentState) -> dict[str, Any]:
    """History-aware planner graph node."""
    ensure_logfire()
    dialogue = build_dialogue(state)
    query = (state.get("current_query") or "").strip()
    if not query and dialogue:
        query = message_text(dialogue[-1])

    with logfire.span(
        "agent.planner",
        query=query[:200],
        history_turns=len(dialogue),
    ):
        try:
            if not query:
                plan = PlanState(
                    intent="conversational",
                    conversational_reply=(
                        "Hi — ask me a technical question about your enterprise "
                        "docs and I'll search the knowledge base."
                    ),
                    rationale="empty query",
                )
            elif settings.has_portkey or settings.has_groq or settings.has_openai:
                plan = _llm_plan(dialogue)
            else:
                logfire.warn("No LLM keys — using rule stub planner")
                plan = _stub_plan(query, dialogue)

            updates = apply_planner_decision(state, plan)
            logfire.info(
                "Planner decision",
                intent=plan.intent,
                rewritten_query=plan.rewritten_query,
                has_reply=bool(plan.conversational_reply),
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


def _stub_plan(query: str, dialogue: Sequence[BaseMessage] = ()) -> PlanState:
    """Offline fallback when no LLM keys are configured."""
    lowered = query.lower().strip()
    if not lowered:
        return PlanState(
            intent="conversational",
            conversational_reply=(
                "Hi — ask me a technical question about your enterprise docs "
                "and I'll search the knowledge base."
            ),
            rationale="empty query",
        )

    if _is_pure_greeting(lowered):
        return PlanState(
            intent="conversational",
            conversational_reply=(
                "Hello! I can search your enterprise knowledge base for "
                "technical answers. What do you want to look up?"
            ),
            rationale="greeting / meta",
        )

    prior_topics: list[str] = []
    for msg in dialogue:
        text = message_text(msg)
        if is_human(msg) and text and text.lower() != lowered:
            prior_topics.append(text)

    if prior_topics and any(
        token in lowered for token in ("it", "that", "this", "them", "those")
    ):
        rewritten = f"{query} (context: {prior_topics[-1]})"
        return PlanState(
            intent="technical",
            rewritten_query=rewritten,
            rationale="stub: pronoun follow-up with prior user turn",
        )

    return PlanState(
        intent="technical",
        rewritten_query=query,
        rationale="stub: pass-through",
    )
