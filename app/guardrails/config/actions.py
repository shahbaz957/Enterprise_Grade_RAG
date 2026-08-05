"""Custom NeMo actions for Enterprise RAG guardrails."""

from __future__ import annotations

import re
from typing import Optional

from nemoguardrails.actions import action

# Covers US-style and international-ish numbers used in acceptance tests.
_PHONE_RE = re.compile(
    r"(?<!\w)(?:"
    r"\+?\d{1,3}[\s.\-]?)?"
    r"(?:\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}"
    r"|\d{3}[\s.\-]\d{3}[\s.\-]\d{4}"
    r"|\d{10,15}"
    r")(?!\w)"
)

_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")


def _has_pii(text: str) -> bool:
    if not text:
        return False
    if _PHONE_RE.search(text):
        return True
    if _EMAIL_RE.search(text):
        return True
    return False


def detect_phone_pii_mapping(result: bool) -> bool:
    """Block when the action returns True (PII found)."""
    return result


@action(is_system_action=True, output_mapping=detect_phone_pii_mapping)
async def detect_phone_pii(
    context: Optional[dict] = None,
    **kwargs,
) -> bool:
    """Return True when the latest user message looks like it contains a phone/email."""
    text = ""
    if context:
        text = str(context.get("user_message") or context.get("bot_message") or "")
    if not text and kwargs.get("text"):
        text = str(kwargs["text"])
    return _has_pii(text)
