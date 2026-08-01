"""Shared types and Logfire setup for document loaders."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import logfire

from app.config import ROOT_DIR, settings

_logfire_ready = False


def ensure_logfire() -> None:
    """Configure Logfire once (credentials file and/or LOGFIRE_TOKEN)."""
    global _logfire_ready
    if _logfire_ready:
        return

    kwargs: dict[str, Any] = {
        "service_name": settings.app_name,
        "service_version": settings.app_version,
        "send_to_logfire": "if-token-present",
        "config_dir": ROOT_DIR / ".logfire",
    }
    if settings.logfire_token:
        kwargs["token"] = settings.logfire_token

    logfire.configure(**kwargs)
    _logfire_ready = True


@dataclass(slots=True)
class LoadedDocument:
    """Normalized extraction result from any loader."""

    text: str
    source: str
    doc_type: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


def resolve_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (ROOT_DIR / p).resolve()
    else:
        p = p.resolve()
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    return p
