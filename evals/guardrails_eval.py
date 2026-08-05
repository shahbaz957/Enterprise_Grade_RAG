"""Guardrails acceptance eval (off-topic, jailbreak, PII).

Run:
  uv run python evals/guardrails_eval.py
"""

from __future__ import annotations

import sys

from app.scripts.test_guardrails import run_cases


def main() -> None:
    code = run_cases()
    if code == 0:
        print("All guardrails acceptance cases passed.")
    else:
        print("One or more guardrails acceptance cases failed.")
    sys.exit(code)


if __name__ == "__main__":
    main()
