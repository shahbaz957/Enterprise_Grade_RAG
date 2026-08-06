"""Streamlit entrypoint for the Eval Monitor (avoids package/script double-import).

Usage (from repo root):
  uv run streamlit run eval_monitor.py
"""

from evals_ui.app import main

if __name__ == "__main__":
    main()
