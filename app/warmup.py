"""Startup warmup — load heavy runtimes before the first user request."""

from __future__ import annotations

from typing import Any

import logfire

from app.config import settings
from app.ingestion.loaders.base import ensure_logfire


def warmup_runtime() -> dict[str, Any]:
    """Pre-load guardrails and the compiled agent graph.

    Document vectors are embedded at ingest time and stored in Qdrant — there is
    no CDC or live re-embedding of chunks. At query time we only embed the user's
    question (one small OpenAI call); that is lazy and fast enough without warmup.
    """
    ensure_logfire()
    status: dict[str, Any] = {
        "graph": False,
        "guardrails": False,
        "errors": [],
    }

    with logfire.span("startup.warmup"):
        try:
            from app.agents.graph import get_compiled_graph

            get_compiled_graph()
            status["graph"] = True
            logfire.info("Warmup: agent graph ready")
        except Exception as exc:  # noqa: BLE001
            status["errors"].append(f"graph: {exc}")
            logfire.warn("Warmup: agent graph failed", error=str(exc))

        if settings.guardrails_enabled and settings.has_guardrails:
            try:
                from app.guardrails.engine import get_rails

                get_rails()
                status["guardrails"] = True
                logfire.info("Warmup: NeMo guardrails ready")
            except Exception as exc:  # noqa: BLE001
                status["errors"].append(f"guardrails: {exc}")
                logfire.warn("Warmup: guardrails load failed", error=str(exc))
        else:
            logfire.info("Warmup: skipping guardrails (disabled or unavailable)")

    logfire.info("Startup warmup complete", **status)
    return status
