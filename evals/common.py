"""Shared paths and helpers for the eval suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EVALS_DIR = Path(__file__).resolve().parent
GOLDEN_PATH = EVALS_DIR / "golden_dataset.json"
RESULTS_DIR = EVALS_DIR / "results"
DEFAULT_RESULTS_PATH = RESULTS_DIR / "latest_run.json"


def load_golden(path: Path | None = None) -> dict[str, Any]:
    p = path or GOLDEN_PATH
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def load_results(path: Path | None = None) -> dict[str, Any]:
    p = path or DEFAULT_RESULTS_PATH
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def save_results(payload: dict[str, Any], path: Path | None = None) -> Path:
    p = path or DEFAULT_RESULTS_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return p


def infer_actual_tools(row: dict[str, Any]) -> list[str]:
    """Derive which agent tools ran from a /query response shape."""
    status = str(row.get("status") or "").lower()
    intent = str(row.get("intent") or "").lower()
    docs = row.get("documents") or row.get("contexts") or []
    if status in {"blocked", "guardrailed"}:
        return []
    if intent == "technical" or (isinstance(docs, list) and len(docs) > 0):
        return ["retriever"]
    return []


def was_blocked(row: dict[str, Any]) -> bool:
    status = str(row.get("status") or "").lower()
    if status in {"blocked", "guardrailed"}:
        return True
    # Heuristic: empty docs + refuse-like intent from rails
    intent = str(row.get("intent") or "").lower()
    return intent == "blocked"


def context_texts_from_documents(documents: list[Any]) -> list[str]:
    texts: list[str] = []
    for d in documents or []:
        if isinstance(d, dict):
            t = (d.get("text") or "").strip()
            if t:
                texts.append(t)
        elif isinstance(d, str) and d.strip():
            texts.append(d.strip())
    return texts


def filenames_from_documents(documents: list[Any]) -> list[str]:
    names: list[str] = []
    for d in documents or []:
        if not isinstance(d, dict):
            continue
        name = d.get("filename") or d.get("source") or ""
        name = str(name).strip()
        if name:
            # normalize to basename
            names.append(Path(name).name)
    return names
