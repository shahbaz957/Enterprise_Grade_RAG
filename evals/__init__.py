"""Enterprise RAG evaluation suite (RAGAS + guardrail confusion matrix).

Commands:
  uv run python -m evals.pipeline          # live POST /query → results JSON
  uv run python -m evals.metrics           # RAGAS + tool correctness tables
  uv run python -m evals.guardrails_eval   # TP/TN/FP/FN matrix
  uv run python -m evals                   # pipeline → metrics → guardrails
"""

from __future__ import annotations

__all__ = ["GOLDEN_PATH", "RESULTS_DIR"]

from evals.common import GOLDEN_PATH, RESULTS_DIR
