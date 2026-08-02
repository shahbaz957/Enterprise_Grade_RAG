"""Ingestion processor — load → chunk → embed → upsert.

CLI
---
    # one file
    uv run python -m app.ingestion.processor --file DATA/true_data/cronjobs.docx --corpus true

    # one directory
    uv run python -m app.ingestion.processor --dir DATA/true_data --corpus true

    # both true_data + noisy_data (universal)
    uv run python -m app.ingestion.processor --universal
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import logfire

from app.config import ROOT_DIR, settings
from app.ingestion.chunking import split_document
from app.ingestion.loaders import SUPPORTED_EXTENSIONS, load_document
from app.ingestion.loaders.base import ensure_logfire, resolve_path
from app.services.retrieval.embedding import embed_chunks, probe
from app.services.retrieval.qdrant_service import upsert_embedded_chunks


@dataclass(slots=True)
class ProcessResult:
    """Outcome for a single file ingest."""

    source: str
    doc_type: str
    corpus: str | None
    chars: int
    chunks: int
    upserted: int
    skipped: bool = False
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and not self.skipped


def _iter_supported_files(directory: Path, *, recursive: bool = True) -> list[Path]:
    pattern_iter: Iterable[Path]
    if recursive:
        pattern_iter = directory.rglob("*")
    else:
        pattern_iter = directory.glob("*")

    files = [
        p
        for p in pattern_iter
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files)


def _write_processed_manifest(result: ProcessResult) -> Path | None:
    """Drop a small JSON receipt under processed_data/ for debugging."""
    try:
        out_dir = settings.processed_data_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = Path(result.source).stem[:80] or "document"
        path = out_dir / f"{name}_{stamp}.json"
        path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
        return path
    except OSError as exc:
        logfire.warn("Could not write processed manifest", error=str(exc))
        return None


def process_file(
    path: str | Path,
    *,
    corpus: str | None = None,
    collection: str | None = None,
    upsert: bool = True,
    allow_embedding_fallback: bool = False,
    write_manifest: bool = True,
) -> ProcessResult:
    """Load → split → embed → (optional) Qdrant upsert for one file."""
    ensure_logfire()
    file_path = resolve_path(path)

    with logfire.span(
        "ingestion.process_file",
        source=str(file_path),
        corpus=corpus,
    ):
        try:
            doc = load_document(file_path)
            if doc.is_empty:
                result = ProcessResult(
                    source=str(file_path),
                    doc_type=doc.doc_type,
                    corpus=corpus,
                    chars=0,
                    chunks=0,
                    upserted=0,
                    skipped=True,
                    error="empty document after load",
                    metadata=doc.metadata,
                )
                logfire.warn("Skipping empty document", source=str(file_path))
                return result

            chunks = split_document(
                doc.text,
                source=doc.source,
                doc_type=doc.doc_type,
                extra_metadata={
                    **doc.metadata,
                    **({"corpus": corpus} if corpus else {}),
                },
            )
            if not chunks:
                result = ProcessResult(
                    source=str(file_path),
                    doc_type=doc.doc_type,
                    corpus=corpus,
                    chars=doc.char_count,
                    chunks=0,
                    upserted=0,
                    skipped=True,
                    error="no chunks produced",
                    metadata=doc.metadata,
                )
                logfire.warn("Skipping — no chunks", source=str(file_path))
                return result

            embedded = embed_chunks(
                chunks,
                allow_fallback=allow_embedding_fallback,
            )
            upserted = 0
            if upsert:
                upserted = upsert_embedded_chunks(
                    embedded,
                    collection=collection,
                    corpus=corpus,
                )

            result = ProcessResult(
                source=str(file_path),
                doc_type=doc.doc_type,
                corpus=corpus,
                chars=doc.char_count,
                chunks=len(chunks),
                upserted=upserted,
                metadata={
                    **doc.metadata,
                    "embedding_provider": embedded[0].provider,
                    "embedding_model": embedded[0].model,
                    "embedding_dimensions": embedded[0].dimensions,
                },
            )
            logfire.info(
                "Processed file",
                source=str(file_path),
                chunks=result.chunks,
                upserted=result.upserted,
                corpus=corpus,
            )
            if write_manifest:
                _write_processed_manifest(result)
            return result

        except Exception as exc:  # noqa: BLE001 — surfaced in ProcessResult
            logfire.error(
                "Failed to process file",
                source=str(file_path),
                error=str(exc),
            )
            return ProcessResult(
                source=str(file_path),
                doc_type="unknown",
                corpus=corpus,
                chars=0,
                chunks=0,
                upserted=0,
                error=str(exc),
            )


def process_directory(
    directory: str | Path,
    *,
    corpus: str | None = None,
    collection: str | None = None,
    recursive: bool = True,
    upsert: bool = True,
    allow_embedding_fallback: bool = False,
    write_manifest: bool = True,
) -> list[ProcessResult]:
    """Process every supported file under a directory."""
    ensure_logfire()
    root = Path(directory).expanduser()
    if not root.is_absolute():
        root = (ROOT_DIR / root).resolve()
    else:
        root = root.resolve()

    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    files = _iter_supported_files(root, recursive=recursive)
    with logfire.span(
        "ingestion.process_directory",
        directory=str(root),
        files=len(files),
        corpus=corpus,
    ):
        logfire.info(
            "Scanning directory for ingestion",
            directory=str(root),
            files=len(files),
            corpus=corpus,
        )
        results = [
            process_file(
                path,
                corpus=corpus,
                collection=collection,
                upsert=upsert,
                allow_embedding_fallback=allow_embedding_fallback,
                write_manifest=write_manifest,
            )
            for path in files
        ]
        ok = sum(1 for r in results if r.ok)
        logfire.info(
            "Directory ingestion finished",
            directory=str(root),
            ok=ok,
            failed=len(results) - ok,
            upserted=sum(r.upserted for r in results),
        )
        return results


def run_universal_ingestion(
    *,
    true_dir: str | Path | None = None,
    noisy_dir: str | Path | None = None,
    collection: str | None = None,
    upsert: bool = True,
    allow_embedding_fallback: bool = False,
    probe_embeddings: bool = True,
) -> dict[str, list[ProcessResult]]:
    """Ingest both corpora: `DATA/true_data` and `DATA/noisy_data`."""
    ensure_logfire()

    with logfire.span("ingestion.universal"):
        if probe_embeddings:
            probe(use_fallback_on_failure=allow_embedding_fallback)

        true_path = Path(true_dir) if true_dir else settings.true_data_dir
        noisy_path = Path(noisy_dir) if noisy_dir else settings.noisy_data_dir

        true_results = process_directory(
            true_path,
            corpus="true",
            collection=collection,
            upsert=upsert,
            allow_embedding_fallback=allow_embedding_fallback,
        )
        noisy_results = process_directory(
            noisy_path,
            corpus="noisy",
            collection=collection,
            upsert=upsert,
            allow_embedding_fallback=allow_embedding_fallback,
        )

        summary = {
            "true_ok": sum(1 for r in true_results if r.ok),
            "true_failed": sum(1 for r in true_results if not r.ok),
            "noisy_ok": sum(1 for r in noisy_results if r.ok),
            "noisy_failed": sum(1 for r in noisy_results if not r.ok),
            "upserted": sum(r.upserted for r in true_results + noisy_results),
        }
        logfire.info("Universal ingestion finished", **summary)
        return {"true": true_results, "noisy": noisy_results}


def _print_results(results: Sequence[ProcessResult]) -> None:
    for r in results:
        status = "OK" if r.ok else ("SKIP" if r.skipped else "ERR")
        msg = r.error or f"chunks={r.chunks} upserted={r.upserted}"
        print(f"[{status}] {r.source} ({r.doc_type}) — {msg}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="app.ingestion.processor",
        description="Enterprise RAG ingestion: load → chunk → embed → Qdrant",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="Process a single file")
    group.add_argument("--dir", type=str, help="Process a directory of documents")
    group.add_argument(
        "--universal",
        action="store_true",
        help="Process DATA/true_data and DATA/noisy_data",
    )

    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Corpus label stored in Qdrant payload (e.g. true / noisy)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help=f"Qdrant collection (default: {settings.qdrant_collection})",
    )
    parser.add_argument(
        "--no-upsert",
        action="store_true",
        help="Load/chunk/embed only — do not write to Qdrant",
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Allow local sentence-transformers if OpenAI fails",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="With --dir, do not recurse into subfolders",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    ensure_logfire()

    upsert = not args.no_upsert
    allow_fallback = args.allow_fallback

    if args.file:
        result = process_file(
            args.file,
            corpus=args.corpus,
            collection=args.collection,
            upsert=upsert,
            allow_embedding_fallback=allow_fallback,
        )
        _print_results([result])
        return 0 if result.ok or result.skipped else 1

    if args.dir:
        results = process_directory(
            args.dir,
            corpus=args.corpus,
            collection=args.collection,
            recursive=not args.no_recursive,
            upsert=upsert,
            allow_embedding_fallback=allow_fallback,
        )
        _print_results(results)
        failed = sum(1 for r in results if not r.ok and not r.skipped)
        return 0 if failed == 0 else 1

    # --universal
    bundled = run_universal_ingestion(
        collection=args.collection,
        upsert=upsert,
        allow_embedding_fallback=allow_fallback,
    )
    print("=== true_data ===")
    _print_results(bundled["true"])
    print("=== noisy_data ===")
    _print_results(bundled["noisy"])
    all_results = bundled["true"] + bundled["noisy"]
    failed = sum(1 for r in all_results if not r.ok and not r.skipped)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
