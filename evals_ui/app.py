"""
Enterprise RAG — Eval Monitor (Streamlit)

Separate from the future Next.js chat UI. Use this to watch RAGAS scores,
tool correctness, and guardrail confusion matrices over time.

Run (from repo root) — prefer the root launcher to avoid double-import:
  uv run streamlit run eval_monitor.py

Or:
  uv run streamlit run evals_ui/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure repo root is on path when launched via `streamlit run evals_ui/app.py`
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _default_base_url() -> str:
    try:
        from app.config import settings

        return settings.backend_url or "http://127.0.0.1:8000"
    except Exception:
        return "http://127.0.0.1:8000"


def _ragas_means_df(means: dict) -> pd.DataFrame:
    if not means:
        return pd.DataFrame(columns=["metric", "score"])
    rows = [{"metric": str(k), "score": float(v)} for k, v in means.items()]
    return pd.DataFrame(rows).sort_values("metric")


def _confusion_heatmap(matrix: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            [matrix.get("TN", 0), matrix.get("FP", 0)],
            [matrix.get("FN", 0), matrix.get("TP", 0)],
        ],
        index=["should_block=False", "should_block=True"],
        columns=["blocked=False", "blocked=True"],
    )


def render_overview(bundle: dict) -> None:
    run = bundle.get("run") or {}
    metrics = bundle.get("metrics") or {}
    guard = bundle.get("guardrails") or {}
    means = metrics.get("ragas_means") or {}
    tool = metrics.get("tool_correctness") or {}

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Cases run", run.get("n_cases") or len(run.get("cases") or []))
    c2.metric("Tool Jaccard", tool.get("mean_jaccard", "—"))
    c3.metric("Tool exact", tool.get("mean_exact", "—"))
    c4.metric("Guardrail F1", guard.get("f1", "—"))
    faith = means.get("faithfulness") or means.get("Faithfulness")
    c5.metric("Faithfulness", faith if faith is not None else "—")

    if run.get("run_at"):
        st.caption(
            f"Last run: `{run.get('run_at')}` · API `{run.get('base_url', '')}` · "
            f"errors={run.get('n_errors', 0)}"
        )


def render_ragas(bundle: dict) -> None:
    st.subheader("RAGAS scores")
    metrics = bundle.get("metrics") or {}
    means = metrics.get("ragas_means") or {}
    err = bundle.get("ragas_error") or metrics.get("ragas_error")

    if err and err != "skipped" and not means:
        st.warning(f"RAGAS unavailable: {err}")
        st.info(
            "Need `OPENAI_API_KEY` (embeddings) and `JUDGE_GROQ_API_KEY` or `GROQ_API_KEY`. "
            "Re-run with Skip RAGAS unchecked."
        )
        return
    if err == "skipped" and not means:
        st.info("RAGAS was skipped for this run. Uncheck **Skip RAGAS** and re-run.")
        return
    if not means:
        st.info("No RAGAS means yet. Run an eval that includes RAG cases.")
        return

    df = _ragas_means_df(means)
    left, right = st.columns([1.1, 1])
    with left:
        st.dataframe(df, width="stretch", hide_index=True)
    with right:
        chart_df = df.set_index("metric")
        st.bar_chart(chart_df)

    ragas = metrics.get("ragas") or {}
    per = ragas.get("rows") or []
    if per:
        with st.expander(f"Per-case RAGAS rows ({len(per)})", expanded=False):
            st.dataframe(pd.DataFrame(per), width="stretch", hide_index=True)

    st.caption(
        "Faithfulness ≈ grounded claims / total claims (hallucination catch). "
        "Typical target ≥ 0.8. Relevancy checks answer↔question fit; "
        "precision/recall judge retrieval; correctness mixes factual + semantic match."
    )


def render_tools(bundle: dict) -> None:
    st.subheader("Tool correctness")
    metrics = bundle.get("metrics") or {}
    tool = metrics.get("tool_correctness") or {}
    rows = metrics.get("tool_rows") or []

    a, b = st.columns(2)
    a.metric("Mean Jaccard", tool.get("mean_jaccard", "—"))
    b.metric("Mean exact match", tool.get("mean_exact", "—"))

    if not rows:
        st.info("No tool rows. Run the pipeline first.")
        return

    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)
    st.caption(
        "Compares `expected_tools` vs `actual_tools` (e.g. whether `retriever` ran). "
        "Zero judge-LLM cost — required for the agentic path."
    )


def render_guardrails(bundle: dict) -> None:
    st.subheader("Guardrails confusion matrix")
    guard = bundle.get("guardrails") or {}
    matrix = guard.get("matrix") or {}
    if not matrix:
        st.info("No guardrails report yet.")
        return

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("TP", matrix.get("TP", 0))
    m2.metric("TN", matrix.get("TN", 0))
    m3.metric("FP", matrix.get("FP", 0))
    m4.metric("FN", matrix.get("FN", 0))
    m5.metric("F1", guard.get("f1", "—"))

    st.write(
        f"Precision **{guard.get('precision', '—')}** · "
        f"Recall **{guard.get('recall', '—')}** · "
        f"Accuracy **{guard.get('accuracy', '—')}**"
    )

    heat = _confusion_heatmap(matrix)
    st.dataframe(heat, width="stretch")

    rows = guard.get("rows") or []
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, width="stretch", hide_index=True)
        fn = df[df["label"] == "FN"] if "label" in df.columns else pd.DataFrame()
        if len(fn):
            st.error(
                f"{len(fn)} false negative(s) — unsafe/off-policy asks that slipped through."
            )
        fp = df[df["label"] == "FP"] if "label" in df.columns else pd.DataFrame()
        if len(fp):
            st.warning(f"{len(fp)} false positive(s) — over-blocked legitimate asks.")

    st.caption(
        "TP = correctly blocked · TN = correctly allowed · "
        "FP = over-blocked · FN = missed block"
    )


def render_cases(bundle: dict) -> None:
    st.subheader("Case explorer")
    cases = (bundle.get("run") or {}).get("cases") or []
    if not cases:
        st.info("No enriched cases.")
        return

    kinds = sorted({c.get("kind") or "?" for c in cases})
    pick_kind = st.multiselect(
        "Filter kind",
        kinds,
        default=kinds,
        key="eval_case_filter_kind",
    )
    filtered = [c for c in cases if (c.get("kind") or "?") in pick_kind]

    compact = []
    for c in filtered:
        compact.append(
            {
                "id": c.get("id"),
                "kind": c.get("kind"),
                "status": c.get("status"),
                "blocked": c.get("blocked"),
                "tools": ", ".join(c.get("actual_tools") or c.get("tools") or []),
                "latency_s": c.get("latency_s"),
                "n_contexts": len(c.get("contexts") or []),
                "question": (c.get("question") or "")[:100],
            }
        )
    st.dataframe(pd.DataFrame(compact), width="stretch", hide_index=True)

    ids = [c.get("id") for c in filtered if c.get("id")]
    if not ids:
        return
    chosen = st.selectbox("Inspect case", ids, key="eval_inspect_case_id")
    case = next(c for c in filtered if c.get("id") == chosen)
    st.markdown(f"**Question:** {case.get('question')}")
    if case.get("reference"):
        with st.expander("Reference (ground truth)"):
            st.write(case.get("reference"))
    st.markdown("**Actual response**")
    st.write(case.get("actual_response") or "_(empty)_")
    ctxs = case.get("contexts") or []
    if ctxs:
        with st.expander(f"Retrieved contexts ({len(ctxs)})"):
            for i, t in enumerate(ctxs, 1):
                st.markdown(f"**Chunk {i}**")
                st.text((t or "")[:2000])


def main() -> None:
    """Single entry for Streamlit — avoids double-render when file is also imported."""
    from evals_ui.services import (
        GUARDRAILS_REPORT,
        LATEST_RUN,
        METRICS_REPORT,
        load_saved_bundle,
        run_eval_suite,
    )

    st.set_page_config(
        page_title="Enterprise RAG · Eval Monitor",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
          .block-container { padding-top: 1.2rem; }
          div[data-testid="stMetricValue"] { font-size: 1.45rem; }
          .erag-banner {
            background: linear-gradient(120deg, #0f172a 0%, #1e3a5f 55%, #334155 100%);
            color: #f8fafc;
            padding: 1.1rem 1.35rem;
            border-radius: 10px;
            margin-bottom: 1rem;
          }
          .erag-banner h1 { margin: 0; font-size: 1.55rem; font-weight: 650; }
          .erag-banner p { margin: 0.35rem 0 0; opacity: 0.88; font-size: 0.95rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Eval controls")
        st.caption("Local monitor only — not the product chat UI (Next.js).")

        base_url = st.text_input(
            "API base URL",
            value=_default_base_url(),
            key="eval_api_base_url",
        )
        limit = st.number_input(
            "Limit cases (0 = all)",
            min_value=0,
            max_value=100,
            value=0,
            key="eval_limit_cases",
        )
        kinds_rag = st.checkbox("Include RAG cases", value=True, key="eval_kinds_rag")
        kinds_gr = st.checkbox(
            "Include guardrail cases", value=True, key="eval_kinds_gr"
        )
        skip_ragas = st.checkbox(
            "Skip RAGAS (faster)", value=False, key="eval_skip_ragas"
        )

        kinds: set[str] = set()
        if kinds_rag:
            kinds.add("rag")
        if kinds_gr:
            kinds.add("guardrail")

        run_clicked = st.button(
            "Run full eval suite",
            type="primary",
            width="stretch",
            key="eval_run_suite",
        )
        load_clicked = st.button(
            "Load saved results",
            width="stretch",
            key="eval_load_saved",
        )

        st.divider()
        st.caption("Reports on disk")
        st.code(
            f"{LATEST_RUN.name}\n{METRICS_REPORT.name}\n{GUARDRAILS_REPORT.name}",
            language="text",
        )
        st.caption(str(LATEST_RUN.parent))

    st.markdown(
        """
        <div class="erag-banner">
          <h1>Enterprise RAG · Eval Monitor</h1>
          <p>RAGAS · tool correctness · guardrail TP/TN/FP/FN — improve retrieval from observations over time.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "bundle" not in st.session_state:
        st.session_state.bundle = load_saved_bundle()

    if run_clicked:
        if not kinds:
            st.error("Select at least one case kind (RAG and/or guardrail).")
        else:
            with st.spinner(
                "Running live /query pipeline + metrics… Keep the FastAPI server up."
            ):
                try:
                    st.session_state.bundle = run_eval_suite(
                        base_url=base_url.strip(),
                        kinds=kinds,
                        limit=int(limit) or None,
                        skip_ragas=skip_ragas,
                    )
                    st.success("Eval finished — reports saved under evals/results/.")
                except Exception as exc:
                    st.exception(exc)

    if load_clicked:
        bundle = load_saved_bundle()
        if bundle is None:
            st.warning("No saved reports found. Run an eval first.")
        else:
            st.session_state.bundle = bundle
            st.success("Loaded evals/results/*")

    bundle = st.session_state.get("bundle")
    if not bundle:
        st.info(
            "No results yet. Start the API (`uvicorn app.main:app --port 8000`), "
            "then click **Run full eval suite**, or **Load saved results** if you already ran "
            "`uv run python -m evals` in the terminal."
        )
        st.stop()

    render_overview(bundle)
    tab_ragas, tab_tools, tab_rails, tab_cases = st.tabs(
        ["RAGAS", "Tool correctness", "Guardrails", "Cases"]
    )
    with tab_ragas:
        render_ragas(bundle)
    with tab_tools:
        render_tools(bundle)
    with tab_rails:
        render_guardrails(bundle)
    with tab_cases:
        render_cases(bundle)


# Only render when this file is the Streamlit entrypoint — not when imported as evals_ui.app
if __name__ == "__main__":
    main()
