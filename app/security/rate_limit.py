"""Per-IP / per-key rate limiting via Upstash Redis REST or in-memory fallback."""

from __future__ import annotations

import hashlib
import threading
import time
from collections import defaultdict, deque
from typing import Annotated, Deque

import httpx
from fastapi import Depends, HTTPException, Request, status

from app.config import settings
from app.security.auth import AuthPrincipal

_memory_lock = threading.Lock()
_memory_hits: dict[str, Deque[float]] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    forwarded = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
    if forwarded:
        return forwarded
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _bucket_key(ip: str, principal: str | None) -> str:
    who = principal or "anon"
    digest = hashlib.sha256(f"{ip}:{who}".encode()).hexdigest()[:24]
    return f"rl:{digest}"


def _memory_allow(key: str, limit: int, window: int) -> tuple[bool, int]:
    now = time.time()
    cutoff = now - window
    with _memory_lock:
        q = _memory_hits[key]
        while q and q[0] < cutoff:
            q.popleft()
        if len(q) >= limit:
            retry = max(1, int(window - (now - q[0])) + 1)
            return False, retry
        q.append(now)
        return True, 0


def _upstash_allow(key: str, limit: int, window: int) -> tuple[bool, int]:
    """Sliding window using Redis ZSET via Upstash REST pipeline."""
    url = settings.upstash_redis_rest_url.rstrip("/")
    token = settings.upstash_redis_rest_token
    now = time.time()
    member = f"{now}:{hashlib.md5(str(now).encode()).hexdigest()[:8]}"
    cutoff = now - window

    # Pipeline: ZREMRANGEBYSCORE, ZADD, ZCARD, EXPIRE
    commands = [
        ["ZREMRANGEBYSCORE", key, "-inf", str(cutoff)],
        ["ZADD", key, str(now), member],
        ["ZCARD", key],
        ["EXPIRE", key, str(window + 5)],
    ]
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.post(
                f"{url}/pipeline",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json=commands,
            )
            resp.raise_for_status()
            results = resp.json()
    except Exception:
        # Fail open if Upstash is unreachable so the product stays usable.
        return _memory_allow(key, limit, window)

    # results is a list of {result: ...} objects from Upstash
    try:
        card = int(results[2].get("result", 0))
    except (IndexError, TypeError, AttributeError, ValueError):
        return _memory_allow(key, limit, window)

    if card > limit:
        return False, window
    return True, 0


async def enforce_rate_limit(
    request: Request,
    principal: AuthPrincipal,
) -> None:
    """Reject with 429 when the IP/key pair exceeds the configured budget."""
    limit = max(1, settings.rate_limit_per_minute)
    window = max(1, settings.rate_limit_window_seconds)
    ip = _client_ip(request)
    key = _bucket_key(ip, principal)

    if settings.has_upstash:
        allowed, retry_after = _upstash_allow(key, limit, window)
    else:
        allowed, retry_after = _memory_allow(key, limit, window)

    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again shortly.",
            headers={"Retry-After": str(max(1, retry_after))},
        )


RateLimited = Annotated[None, Depends(enforce_rate_limit)]
