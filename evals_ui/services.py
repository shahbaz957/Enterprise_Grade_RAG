"""Shared helpers for the Streamlit eval dashboard (load/save + run suite)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Repo root = parent of evals_ui/
ROOT = Path(__file__).resolve().parent.parent
EVALS_RESULTS = ROOT / "evals" / "results"
LATEST_RUN = EVALS_RESULTS / "latest_run.json"
METRICS_REPORT = EVALS_RESULTS / "metrics_report.json"
GUARDRAILS_REPORT = EVALS_RESULTS / "guardrails_report.json"


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def run_eval_suite(
    *,
    base_url: str,
    kinds: set[str],
    limit: int | None,
    skip_ragas: bool,
) -> dict[str, Any]:
    """Run pipeline → metrics → guardrails; persist reports; return combined payload."""
    from evals.common import DEFAULT_RESULTS_PATH, save_results
    from evals.guardrails_eval import confusion_for_cases
    from evals.metrics import compute_tool_correctness, run_ragas
    from evals.pipeline import run_pipeline

    payload = run_pipeline(base_url=base_url, kinds=kinds, limit=limit)
    save_results(payload, DEFAULT_RESULTS_PATH)

    cases = list(payload.get("cases") or [])
    tool = compute_tool_correctness(cases)

    metrics_report: dict[str, Any] = {
        "source": str(DEFAULT_RESULTS_PATH),
        "tool_correctness": {
            "mean_jaccard": tool["mean_jaccard"],
            "mean_exact": tool["mean_exact"],
        },
        "tool_rows": tool["rows"],
        "ragas_means": {},
        "n_ragas": 0,
        "n_tool_rows": len(tool["rows"]),
    }

    ragas_error: str | None = None
    if not skip_ragas:
        try:
            ragas = run_ragas(cases)
            metrics_report["ragas_means"] = ragas.get("means") or {}
            metrics_report["n_ragas"] = ragas.get("n", 0)
            metrics_report["ragas"] = ragas
        except Exception as exc:
            ragas_error = str(exc)
            metrics_report["ragas_error"] = ragas_error
    else:
        metrics_report["ragas_error"] = "skipped"

    save_json(METRICS_REPORT, metrics_report)

    guardrails_report = confusion_for_cases(cases)
    save_json(GUARDRAILS_REPORT, guardrails_report)

    return {
        "run": payload,
        "metrics": metrics_report,
        "guardrails": guardrails_report,
        "ragas_error": ragas_error,
    }


def load_saved_bundle() -> dict[str, Any] | None:
    run = load_json(LATEST_RUN)
    metrics = load_json(METRICS_REPORT)
    guardrails = load_json(GUARDRAILS_REPORT)
    if not run and not metrics and not guardrails:
        return None
    return {
        "run": run or {"cases": []},
        "metrics": metrics or {},
        "guardrails": guardrails or {},
        "ragas_error": (metrics or {}).get("ragas_error"),
    }
