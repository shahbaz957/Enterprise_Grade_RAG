"""Smoke-test Portkey gateway: config print, optional live call + cache timing.

Usage:
  uv run python -m app.scripts.test_portkey
  uv run python -m app.scripts.test_portkey --live
  uv run python -m app.scripts.test_portkey --live --cache

Without --live: only prints whether Portkey is configured and the inline Config JSON
(paste into Portkey dashboard if you prefer a Config ID).

With --live: sends one chat turn through the gateway (needs PORTKEY_* in .env).
With --cache: sends the same prompt twice and prints latencies (2nd should be faster
when PORTKEY_CACHE_MODE=simple|semantic and Portkey logs show a cache hit).
"""

from __future__ import annotations

import argparse
import sys
import time

from app.config import settings
from app.gateway.portkey_client import (
    describe_inline_config,
    gateway_status,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Portkey gateway smoke test")
    parser.add_argument("--live", action="store_true", help="Invoke a real chat completion")
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Invoke twice with identical prompt to observe cache latency",
    )
    args = parser.parse_args()

    status = gateway_status()
    print("Portkey status:", status)
    print("\nInline gateway config (also usable in Portkey dashboard):\n")
    print(describe_inline_config() or "(empty — set virtual key slugs or PORTKEY_CONFIG_ID)")

    if not args.live:
        print(
            "\nTip: create Virtual Keys in https://app.portkey.ai, set "
            "PORTKEY_VIRTUAL_KEY_PRIMARY / PORTKEY_VIRTUAL_KEY_FALLBACK, then re-run with --live."
        )
        return 0 if status.get("enabled") or not settings.portkey_enabled else 1

    if not settings.has_portkey:
        print("FAIL: Portkey not configured (PORTKEY_API_KEY + virtual keys / config id)")
        return 1

    from langchain_core.messages import HumanMessage

    from app.agents.llm import invoke_chat

    prompt = "Reply with exactly: portkey-ok"
    messages = [HumanMessage(content=prompt)]

    def _one(label: str) -> float:
        t0 = time.perf_counter()
        resp = invoke_chat(
            messages,
            temperature=0,
            run_name="test.portkey",
            route="script.test_portkey",
            feature="portkey_smoke",
            user_id="smoke-test-user",
        )
        dt = time.perf_counter() - t0
        text = getattr(resp, "content", str(resp))
        print(f"[{label}] {dt:.3f}s → {text!r}")
        return dt

    t1 = _one("call-1")
    if args.cache:
        t2 = _one("call-2-same-prompt")
        print(
            f"\nLatency delta: call2-call1 = {t2 - t1:.3f}s "
            f"(expect call-2 faster on cache hit; confirm in Portkey Logs)"
        )
        if t2 < t1:
            print("PASS: second call was faster (likely cache / warm path)")
        else:
            print(
                "NOTE: second call was not faster — check PORTKEY_CACHE_MODE, "
                "cache max_age, and Portkey log 'cache status' column"
            )

    print(
        "\nVerify in Portkey dashboard → Logs: metadata (user_id, route, environment), "
        "fallback path if primary VK is broken, and cost."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
