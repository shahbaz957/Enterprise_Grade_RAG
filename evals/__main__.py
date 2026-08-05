"""Run the full eval suite: pipeline → metrics → guardrails matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Full Enterprise RAG eval suite")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Skip LLM-judge RAGAS (still runs tool correctness + guardrails)",
    )
    parser.add_argument(
        "--kinds",
        default="rag,guardrail",
        help="Comma-separated: rag,guardrail",
    )
    args = parser.parse_args(argv)

    from evals.common import DEFAULT_RESULTS_PATH, RESULTS_DIR, save_results
    from evals.pipeline import run_pipeline

    try:
        from app.config import settings

        base = args.base_url or settings.backend_url or "http://127.0.0.1:8000"
    except Exception:
        base = args.base_url or "http://127.0.0.1:8000"

    kinds = {k.strip() for k in args.kinds.split(",") if k.strip()}
    print("=== Step 2: live pipeline ===")
    payload = run_pipeline(base_url=base, kinds=kinds, limit=args.limit)
    save_results(payload, DEFAULT_RESULTS_PATH)
    print(f"Saved → {DEFAULT_RESULTS_PATH}")

    print("\n=== Step 3a: tool correctness + RAGAS ===")
    from evals.metrics import compute_tool_correctness, run_ragas, _print_tables

    cases = list(payload.get("cases") or [])
    tool = compute_tool_correctness(cases)
    report: dict = {
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
    if not args.skip_ragas:
        try:
            ragas = run_ragas(cases)
            report["ragas_means"] = ragas.get("means") or {}
            report["n_ragas"] = ragas.get("n", 0)
            report["ragas"] = ragas
        except Exception as exc:
            print(f"RAGAS skipped/failed: {exc}")
            report["ragas_error"] = str(exc)
    else:
        print("Skipping RAGAS (--skip-ragas)")

    _print_tables(report)
    metrics_path = RESULTS_DIR / "metrics_report.json"
    import json

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"Wrote {metrics_path}")

    print("\n=== Step 3b: guardrails confusion matrix ===")
    from evals.guardrails_eval import confusion_for_cases

    g = confusion_for_cases(cases)
    m = g["matrix"]
    print(f"  TP={m['TP']} TN={m['TN']} FP={m['FP']} FN={m['FN']}")
    print(
        f"  precision={g['precision']} recall={g['recall']} "
        f"accuracy={g['accuracy']} f1={g['f1']}"
    )
    g_path = RESULTS_DIR / "guardrails_report.json"
    with g_path.open("w", encoding="utf-8") as f:
        json.dump(g, f, indent=2, ensure_ascii=False)
    print(f"Wrote {g_path}")

    fn = [r for r in g["rows"] if r.get("kind") == "guardrail" and r["label"] == "FN"]
    return 1 if fn else 0


if __name__ == "__main__":
    sys.exit(main())
