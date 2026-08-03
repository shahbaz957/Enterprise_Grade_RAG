"""Portkey AI Gateway — virtual keys, fallback/loadbalance, cache, retries, metadata.

When ``PORTKEY_API_KEY`` is set, agent chat calls route through Portkey instead of
talking to Groq/OpenAI directly. Fallbacks, retries, timeouts, and caching are
expressed as a Portkey Config (inline JSON or a dashboard Config ID).

Dashboard setup (once):
  1. Create Virtual Keys for Groq (primary) and OpenAI (fallback) in Portkey.
  2. Put their slugs in ``PORTKEY_VIRTUAL_KEY_PRIMARY`` / ``PORTKEY_VIRTUAL_KEY_FALLBACK``.
  3. Optionally save the same JSON as a Config and set ``PORTKEY_CONFIG_ID``.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import logfire

from app.config import settings

PORTKEY_GATEWAY_URL = "https://api.portkey.ai/v1"


def build_gateway_config() -> dict[str, Any]:
    """Inline Portkey Config: routing + retry + timeout + cache."""
    primary_vk = settings.portkey_virtual_key_primary or settings.portkey_virtual_key
    fallback_vk = settings.portkey_virtual_key_fallback

    primary_model = settings.groq_chat_model
    fallback_model = settings.openai_chat_model

    targets: list[dict[str, Any]] = []

    if primary_vk:
        primary: dict[str, Any] = {
            "virtual_key": primary_vk,
            "override_params": {"model": primary_model},
            "retry": {
                "attempts": settings.portkey_retry_attempts,
                "on_status_codes": [408, 429, 500, 502, 503, 504],
            },
            "request_timeout": settings.portkey_timeout_ms,
        }
        if settings.portkey_strategy == "loadbalance":
            primary["weight"] = settings.portkey_primary_weight
        targets.append(primary)

    if fallback_vk:
        secondary: dict[str, Any] = {
            "virtual_key": fallback_vk,
            "override_params": {"model": fallback_model},
            "retry": {
                "attempts": max(1, settings.portkey_retry_attempts - 1),
                "on_status_codes": [408, 429, 500, 502, 503, 504],
            },
            "request_timeout": settings.portkey_timeout_ms,
        }
        if settings.portkey_strategy == "loadbalance":
            secondary["weight"] = settings.portkey_fallback_weight
        targets.append(secondary)

    if not targets:
        # No VKs: still allow gateway with provider + real keys via headers only.
        # Config stays empty; caller passes virtual_key/provider in createHeaders.
        return {}

    mode = settings.portkey_strategy if settings.portkey_strategy in (
        "fallback",
        "loadbalance",
    ) else "fallback"

    config: dict[str, Any] = {
        "strategy": {"mode": mode},
        "targets": targets,
    }

    cache_mode = (settings.portkey_cache_mode or "off").lower().strip()
    if cache_mode in ("simple", "semantic"):
        config["cache"] = {
            "mode": cache_mode,
            "max_age": settings.portkey_cache_max_age,
        }

    return config


def build_metadata(
    *,
    user_id: str | None = None,
    route: str | None = None,
    feature: str | None = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Attach request metadata for Portkey logs / cost dashboards."""
    meta: dict[str, Any] = {
        "_environment": settings.portkey_environment or ("dev" if settings.debug else "prod"),
        "environment": settings.portkey_environment or ("dev" if settings.debug else "prod"),
        "app": settings.app_name,
        "version": settings.app_version,
        "feature": feature or "agent_chat",
        "route": route or "llm",
    }
    if user_id:
        meta["user"] = user_id
        meta["user_id"] = user_id
    if extra:
        meta.update({k: v for k, v in extra.items() if v is not None})
    # Portkey metadata values must be strings
    return {k: str(v) for k, v in meta.items()}


def create_portkey_headers(
    *,
    user_id: str | None = None,
    route: str | None = None,
    feature: str | None = None,
    extra_metadata: Optional[dict[str, Any]] = None,
) -> dict[str, str]:
    """Build Portkey gateway headers (API key + config + metadata)."""
    from portkey_ai import createHeaders

    kwargs: dict[str, Any] = {
        "api_key": settings.portkey_api_key,
        "metadata": build_metadata(
            user_id=user_id,
            route=route,
            feature=feature,
            extra=extra_metadata,
        ),
    }

    if settings.portkey_config_id:
        kwargs["config"] = settings.portkey_config_id
    else:
        cfg = build_gateway_config()
        if cfg:
            kwargs["config"] = cfg
        else:
            # Single virtual key without multi-target config
            vk = settings.portkey_virtual_key_primary or settings.portkey_virtual_key
            if vk:
                kwargs["virtual_key"] = vk

    headers = createHeaders(**kwargs)
    # createHeaders may return a dict-like Headers object
    if hasattr(headers, "items"):
        return {str(k): str(v) for k, v in headers.items()}
    return dict(headers)


def get_portkey_chat_model(
    *,
    temperature: float = 0.2,
    user_id: str | None = None,
    route: str | None = None,
    feature: str | None = None,
    extra_metadata: Optional[dict[str, Any]] = None,
) -> Any:
    """LangChain ChatOpenAI client pointed at the Portkey gateway."""
    from langchain_openai import ChatOpenAI

    headers = create_portkey_headers(
        user_id=user_id,
        route=route,
        feature=feature,
        extra_metadata=extra_metadata,
    )

    # Model is usually overridden by Portkey config targets; set primary as default.
    model = settings.groq_chat_model if (
        settings.portkey_virtual_key_primary or settings.portkey_virtual_key or settings.has_groq
    ) else settings.openai_chat_model

    logfire.info(
        "Portkey chat client",
        strategy=settings.portkey_strategy,
        cache=settings.portkey_cache_mode,
        config_id=settings.portkey_config_id or None,
        has_inline_config=not bool(settings.portkey_config_id),
        route=route or "llm",
        feature=feature or "agent_chat",
    )

    return ChatOpenAI(
        model=model,
        api_key=settings.portkey_api_key or "not-needed",
        base_url=PORTKEY_GATEWAY_URL,
        default_headers=headers,
        temperature=temperature,
        # Portkey Config owns retries; avoid double-retry storms.
        max_retries=0,
        timeout=max(1.0, settings.portkey_timeout_ms / 1000.0),
    )


def gateway_status() -> dict[str, Any]:
    """Small diagnostic payload for /ready and scripts."""
    cfg = build_gateway_config() if settings.has_portkey else {}
    return {
        "enabled": settings.has_portkey,
        "strategy": settings.portkey_strategy if settings.has_portkey else None,
        "cache_mode": settings.portkey_cache_mode if settings.has_portkey else None,
        "config_id": settings.portkey_config_id or None,
        "has_primary_vk": bool(
            settings.portkey_virtual_key_primary or settings.portkey_virtual_key
        ),
        "has_fallback_vk": bool(settings.portkey_virtual_key_fallback),
        "inline_targets": len(cfg.get("targets") or []),
        "environment": settings.portkey_environment,
    }


def describe_inline_config() -> str:
    """Pretty JSON of the inline config (for debugging / Portkey dashboard paste)."""
    return json.dumps(build_gateway_config(), indent=2)
