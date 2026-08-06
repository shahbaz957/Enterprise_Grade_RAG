"""NeMo Guardrails engine — input/output gates around the LangGraph agent.

No NVIDIA account is required. The checker LLM follows USE_OPENAI_LLM
(OpenAI when true, Groq when false). Dialog Colang flows use FastEmbed + LLM
intent confirm; passthrough mode forwards unmatched turns to the agent.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Optional, TypeVar

import logfire

from app.config import settings

# Returned by passthrough when no dialog/input rail handled the turn.
CONTINUE_TO_AGENT = "__CONTINUE_TO_AGENT__"

DEFAULT_BLOCK_MESSAGE = (
    "I'm sorry, I can't respond to that. Please ask a question about your "
    "enterprise documents or technical topics."
)

_rails_lock = threading.Lock()
_rails: Any = None
_rails_init_error: Optional[str] = None
_T = TypeVar("_T")


class RailDecisionKind(str, Enum):
    PASSED = "passed"
    MODIFIED = "modified"
    BLOCKED = "blocked"
    HANDLED = "handled"  # dialog canned reply (greeting / refuse) — skip agent


@dataclass(frozen=True)
class RailDecision:
    kind: RailDecisionKind
    content: str
    rail: Optional[str] = None

    @property
    def blocked(self) -> bool:
        return self.kind == RailDecisionKind.BLOCKED

    @property
    def skip_agent(self) -> bool:
        return self.kind in (RailDecisionKind.BLOCKED, RailDecisionKind.HANDLED)


def _run_coro_sync(coro: Awaitable[_T]) -> _T:
    """Run an async NeMo API from sync code even when FastAPI's loop is running.

    NeMo's sync ``generate`` / ``check`` raise if an event loop is already
    running (our ``async def /query`` path). Prefer ``generate_async`` /
    ``check_async`` and bridge via a worker thread when needed.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    def _runner() -> _T:
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_runner).result()


def _config_path() -> Path:
    raw = (settings.guardrails_config_path or "").strip()
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = settings.root_dir / p
        return p
    return Path(__file__).resolve().parent / "config"


def _model_entry_from_settings() -> dict[str, Any]:
    """NeMo checker LLM — same USE_OPENAI_LLM flag as the rest of the app."""
    if settings.use_openai_llm:
        if not settings.has_openai:
            raise RuntimeError("USE_OPENAI_LLM=true but OPENAI_API_KEY is unset")
        return {
            "type": "main",
            "engine": "openai",
            "model": settings.openai_chat_model,
            "parameters": {
                "api_key": settings.openai_api_key,
            },
        }
    if not settings.has_groq:
        raise RuntimeError("USE_OPENAI_LLM=false but GROQ_API_KEY is unset")
    return {
        "type": "main",
        "engine": "openai",
        "model": settings.groq_chat_model,
        "parameters": {
            "base_url": "https://api.groq.com/openai/v1",
            "api_key": settings.active_groq_api_key,
        },
    }

async def _passthrough_fn(context: dict, events: list) -> str:
    return CONTINUE_TO_AGENT


def _extract_response_text(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return response.strip()
    if isinstance(response, dict):
        content = response.get("content")
        if isinstance(content, str):
            return content.strip()
        if "response" in response and isinstance(response["response"], list):
            for msg in reversed(response["response"]):
                if isinstance(msg, dict) and msg.get("content"):
                    return str(msg["content"]).strip()
    content = getattr(response, "response", None) or getattr(response, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        for msg in reversed(content):
            if isinstance(msg, dict) and msg.get("content"):
                return str(msg["content"]).strip()
            text = getattr(msg, "content", None)
            if text:
                return str(text).strip()
    return str(response).strip()


def _blocking_rail_name(response: Any) -> Optional[str]:
    log = getattr(response, "log", None)
    if log is None and isinstance(response, dict):
        log = response.get("log")
    if log is None:
        return None
    activated = getattr(log, "activated_rails", None)
    if activated is None and isinstance(log, dict):
        activated = log.get("activated_rails")
    if not activated:
        return None
    for rail in activated:
        stopped = getattr(rail, "stop", None)
        if stopped is None and isinstance(rail, dict):
            stopped = rail.get("stop")
        name = getattr(rail, "name", None)
        if name is None and isinstance(rail, dict):
            name = rail.get("name")
        decisions = getattr(rail, "decisions", None)
        if decisions is None and isinstance(rail, dict):
            decisions = rail.get("decisions")
        if stopped or (isinstance(decisions, list) and "stop" in decisions):
            return str(name) if name else "rail"
    return None


def get_rails(*, force_reload: bool = False) -> Any:
    """Lazy singleton LLMRails. Thread-safe."""
    global _rails, _rails_init_error
    if not settings.guardrails_enabled:
        return None
    if _rails is not None and not force_reload:
        return _rails
    with _rails_lock:
        if _rails is not None and not force_reload:
            return _rails
        try:
            from nemoguardrails import LLMRails, RailsConfig

            from app.ingestion.loaders.base import ensure_logfire

            ensure_logfire()
            path = _config_path()
            if not path.is_dir():
                raise FileNotFoundError(f"Guardrails config not found: {path}")

            config = RailsConfig.from_path(str(path))
            config.passthrough = True
            model_dict = _model_entry_from_settings()
            from nemoguardrails.rails.llm.config import Model

            config.models = [Model(**model_dict)]

            rails = LLMRails(config=config, verbose=False)
            rails.passthrough_fn = _passthrough_fn
            _rails = rails
            _rails_init_error = None
            logfire.info("NeMo Guardrails loaded", config_path=str(path))
            return _rails
        except Exception as exc:
            _rails_init_error = str(exc)
            logfire.exception("Failed to initialize NeMo Guardrails", error=str(exc))
            raise


def guardrails_available() -> bool:
    if not settings.guardrails_enabled:
        return False
    if not (settings.has_groq or settings.has_openai):
        return False
    return _config_path().is_dir()


def check_input(text: str) -> RailDecision:
    """Run input rails + dialog Colang. Sync for invoke_agent."""
    q = (text or "").strip()
    if not q:
        return RailDecision(
            kind=RailDecisionKind.BLOCKED,
            content=DEFAULT_BLOCK_MESSAGE,
            rail="empty",
        )

    if not settings.guardrails_enabled:
        return RailDecision(kind=RailDecisionKind.PASSED, content=q)

    with logfire.span("guardrails.input", question=q[:200]):
        try:
            rails = get_rails()
        except Exception as exc:
            if settings.guardrails_fail_open:
                logfire.warn("Guardrails init failed; fail-open", error=str(exc))
                return RailDecision(kind=RailDecisionKind.PASSED, content=q)
            return RailDecision(
                kind=RailDecisionKind.BLOCKED,
                content=DEFAULT_BLOCK_MESSAGE,
                rail="init_error",
            )

        try:
            response = _run_coro_sync(
                rails.generate_async(
                    messages=[{"role": "user", "content": q}],
                    options={
                        "rails": {
                            "input": True,
                            "dialog": True,
                            "retrieval": False,
                            "output": False,
                        },
                        "log": {"activated_rails": True},
                    },
                )
            )
        except Exception as exc:
            logfire.exception("Guardrails input check failed", error=str(exc))
            if settings.guardrails_fail_open:
                return RailDecision(kind=RailDecisionKind.PASSED, content=q)
            return RailDecision(
                kind=RailDecisionKind.BLOCKED,
                content=DEFAULT_BLOCK_MESSAGE,
                rail="input_error",
            )

        content = _extract_response_text(response)
        blocking = _blocking_rail_name(response)

        if content == CONTINUE_TO_AGENT or content.startswith(CONTINUE_TO_AGENT):
            logfire.info("Guardrails input passed", rail=None)
            return RailDecision(kind=RailDecisionKind.PASSED, content=q)

        if blocking:
            logfire.info(
                "Guardrails input blocked",
                rail=blocking,
                answer_chars=len(content),
            )
            return RailDecision(
                kind=RailDecisionKind.BLOCKED,
                content=content or DEFAULT_BLOCK_MESSAGE,
                rail=blocking,
            )

        logfire.info(
            "Guardrails input handled by dialog",
            answer_chars=len(content),
            preview=content[:120],
        )
        return RailDecision(
            kind=RailDecisionKind.HANDLED,
            content=content or DEFAULT_BLOCK_MESSAGE,
            rail="dialog",
        )


def _mask_pii_in_text(text: str) -> str:
    """Defense-in-depth output masking when Presidio spaCy model is unavailable."""
    import re

    phone = re.compile(
        r"(?<!\w)(?:\+?\d{1,3}[\s.\-]?)?(?:\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}|\d{10,15})(?!\w)"
    )
    email = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
    out = phone.sub("[PHONE]", text)
    out = email.sub("[EMAIL]", out)
    return out


def check_output(user_text: str, answer: str) -> RailDecision:
    """Run output rails on the agent's final answer. Sync for invoke_agent."""
    a = (answer or "").strip()
    if not a:
        return RailDecision(kind=RailDecisionKind.PASSED, content=a)

    if not settings.guardrails_enabled:
        return RailDecision(kind=RailDecisionKind.PASSED, content=a)

    with logfire.span("guardrails.output", answer_chars=len(a)):
        try:
            from nemoguardrails.rails.llm.options import RailStatus, RailType

            rails = get_rails()
            result = _run_coro_sync(
                rails.check_async(
                    [
                        {"role": "user", "content": (user_text or "").strip()},
                        {"role": "assistant", "content": a},
                    ],
                    rail_types=[RailType.OUTPUT],
                )
            )
        except Exception as exc:
            logfire.exception("Guardrails output check failed", error=str(exc))
            if settings.guardrails_fail_open:
                masked = _mask_pii_in_text(a)
                kind = (
                    RailDecisionKind.MODIFIED
                    if masked != a
                    else RailDecisionKind.PASSED
                )
                return RailDecision(kind=kind, content=masked)
            return RailDecision(
                kind=RailDecisionKind.BLOCKED,
                content=DEFAULT_BLOCK_MESSAGE,
                rail="output_error",
            )

        status = result.status
        content = (result.content or "").strip()
        rail = result.rail

        if status == RailStatus.BLOCKED:
            logfire.info("Guardrails output blocked", rail=rail)
            return RailDecision(
                kind=RailDecisionKind.BLOCKED,
                content=content or DEFAULT_BLOCK_MESSAGE,
                rail=rail,
            )

        final = content or a
        masked = _mask_pii_in_text(final)
        if status == RailStatus.MODIFIED or masked != final:
            logfire.info("Guardrails output modified", rail=rail)
            return RailDecision(
                kind=RailDecisionKind.MODIFIED,
                content=masked,
                rail=rail or "pii_mask",
            )
        return RailDecision(kind=RailDecisionKind.PASSED, content=final)
