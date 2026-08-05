"""Step 2 — Live eval pipeline: POST /query for each golden case and enrich results.

Requires the API to be running (default http://127.0.0.1:8000).

Usage:
  uv run python -m evals.pipeline
  uv run python -m evals.pipeline --base-url http://127.0.0.1:8000 --limit 5
  uv run python -m evals.pipeline --kinds rag
  uv run python -m evals.pipeline --kinds guardrail
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from evals.common import (
    DEFAULT_RESULTS_PATH,
    context_texts_from_documents,
    filenames_from_documents,
    infer_actual_tools,
    load_golden,
    save_results,
    was_blocked,
)


# In-memory store (also written to disk). Useful if a Streamlit UI imports this module.
session_state: dict[str, Any] = {
    "enriched_cases": [],
    "run_meta": {},
}


def _post_query(
    client: httpx.Client,
    *,
    question: str,
    session_id: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"question": question}
    if session_id:
        payload["session_id"] = session_id
    r = client.post("/query", json=payload)
    r.raise_for_status()
    return r.json()


def enrich_case(case: dict[str, Any], api_row: dict[str, Any], latency_s: float) -> dict[str, Any]:
    documents = list(api_row.get("documents") or [])
    contexts = context_texts_from_documents(documents)
    actual_tools = infer_actual_tools(
        {
            "status": api_row.get("status"),
            "intent": api_row.get("intent"),
            "documents": documents,
        }
    )
    blocked = was_blocked(api_row)
    return {
        **case,
        "actual_response": api_row.get("answer") or "",
        "contexts": contexts,
        "documents": documents,
        "retrieved_filenames": filenames_from_documents(documents),
        "tools": actual_tools,
        "actual_tools": actual_tools,
        "blocked": blocked,
        "status": api_row.get("status"),
        "intent": api_row.get("intent"),
        "error": api_row.get("error"),
        "session_id": api_row.get("session_id"),
        "latency_s": round(latency_s, 3),
    }


def run_pipeline(
    *,
    base_url: str,
    kinds: set[str] | None = None,
    limit: int | None = None,
    timeout_s: float = 180.0,
    golden_path: Any = None,
) -> dict[str, Any]:
    golden = load_golden(golden_path)
    cases = list(golden.get("cases") or [])
    if kinds:
        cases = [c for c in cases if c.get("kind") in kinds]
    if limit is not None:
        cases = cases[: max(0, limit)]

    enriched: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    with httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_s) as client:
        # Health probe
        try:
            health = client.get("/health")
            health.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                f"API not reachable at {base_url} ({exc}). "
                "Start uvicorn first: uv run python -m uvicorn app.main:app --port 8000"
            ) from exc

        for i, case in enumerate(cases, start=1):
            q = case.get("question") or ""
            print(f"[{i}/{len(cases)}] {case.get('id')} ({case.get('kind')}): {q[:80]!r}")
            t0 = time.perf_counter()
            try:
                api_row = _post_query(client, question=q)
                row = enrich_case(case, api_row, time.perf_counter() - t0)
                enriched.append(row)
                print(
                    f"  → status={row.get('status')} tools={row.get('tools')} "
                    f"ctx={len(row.get('contexts') or [])} blocked={row.get('blocked')} "
                    f"({row['latency_s']}s)"
                )
            except Exception as exc:
                err = {
                    **case,
                    "actual_response": "",
                    "contexts": [],
                    "documents": [],
                    "tools": [],
                    "actual_tools": [],
                    "blocked": False,
                    "error": str(exc),
                    "latency_s": round(time.perf_counter() - t0, 3),
                }
                errors.append(err)
                enriched.append(err)
                print(f"  → ERROR: {exc}")

    payload = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "n_cases": len(enriched),
        "n_errors": len(errors),
        "cases": enriched,
    }
    session_state["enriched_cases"] = enriched
    session_state["run_meta"] = {
        "run_at": payload["run_at"],
        "base_url": base_url,
        "n_cases": len(enriched),
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Live /query eval pipeline")
    parser.add_argument(
        "--base-url",
        default=None,
        help="API base URL (default: settings.backend_url or http://127.0.0.1:8000)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max cases to run")
    parser.add_argument(
        "--kinds",
        default="rag,guardrail",
        help="Comma-separated kinds: rag,guardrail",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_RESULTS_PATH),
        help="Output JSON path",
    )
    args = parser.parse_args()

    try:
        from app.config import settings

        base = args.base_url or settings.backend_url or "http://127.0.0.1:8000"
    except Exception:
        base = args.base_url or "http://127.0.0.1:8000"

    kinds = {k.strip() for k in args.kinds.split(",") if k.strip()}
    payload = run_pipeline(base_url=base, kinds=kinds, limit=args.limit)
    out = save_results(payload, path=__import__("pathlib").Path(args.out))
    print(f"\nWrote {payload['n_cases']} enriched cases → {out}")


if __name__ == "__main__":
    main()
