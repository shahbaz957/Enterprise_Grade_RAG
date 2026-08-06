"""Portkey AI Gateway — virtual keys, fallback/loadbalance, cache, retries, metadata.

Many Portkey orgs enable ``block_inline_config``. In that case you cannot send a
JSON config in ``x-portkey-config`` — you must reference a saved Config ID
(``pc-...``) via ``PORTKEY_CONFIG_ID``, or use a single virtual key only.

Recommended setup:
  1. Create Virtual Keys for OpenAI (primary) and Groq (fallback) in Portkey.
  2. Paste the JSON from ``uv run python -m app.scripts.print_portkey_config``
     into a Portkey Config and copy its ID into ``PORTKEY_CONFIG_ID``.
  3. Set ``PORTKEY_API_KEY`` (+ optional VKs if the Config does not embed them).
"""

from __future__ import annotations

import json
from typing import Any, Optional

import logfire

from app.config import settings

PORTKEY_GATEWAY_URL = "https://api.portkey.ai/v1"


def build_gateway_config() -> dict[str, Any]:
    """Portkey Config JSON for the dashboard (fallback / loadbalance / cache / retry).

    Do **not** send this inline unless ``PORTKEY_ALLOW_INLINE_CONFIG=true`` and your
    org allows it. Prefer saving it in Portkey and setting ``PORTKEY_CONFIG_ID``.

    Primary = OpenAI VK + model; fallback = Groq VK + model.
    """
    primary_vk = settings.portkey_virtual_key_primary or settings.portkey_virtual_key
    fallback_vk = settings.portkey_virtual_key_fallback

    primary_model = settings.openai_chat_model
    fallback_model = settings.groq_chat_model

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
    return {k: str(v) for k, v in meta.items()}


def create_portkey_headers(
    *,
    user_id: str | None = None,
    route: str | None = None,
    feature: str | None = None,
    extra_metadata: Optional[dict[str, Any]] = None,
) -> dict[str, str]:
    """Build Portkey gateway headers.

    Priority:
      1. ``PORTKEY_CONFIG_ID`` (``pc-...``) — required for multi-target / cache when
         the org blocks inline configs.
      2. Inline JSON config — only if ``PORTKEY_ALLOW_INLINE_CONFIG=true``.
      3. Single ``virtual_key`` header (primary VK only; no automatic fallback).
    """
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

    config_id = (settings.portkey_config_id or "").strip()
    allow_inline = bool(settings.portkey_allow_inline_config)

    if config_id:
        kwargs["config"] = config_id
        # Some dashboard configs still need a default VK; harmless if unused.
        vk = settings.portkey_virtual_key_primary or settings.portkey_virtual_key
        if vk:
            kwargs["virtual_key"] = vk
    elif allow_inline:
        cfg = build_gateway_config()
        if cfg:
            kwargs["config"] = cfg
            logfire.warn(
                "Portkey using inline config — org must allow inline configs "
                "(set PORTKEY_CONFIG_ID to avoid this)"
            )
        else:
            vk = settings.portkey_virtual_key_primary or settings.portkey_virtual_key
            if vk:
                kwargs["virtual_key"] = vk
    else:
        # Safe default for orgs with block_inline_config.
        vk = settings.portkey_virtual_key_primary or settings.portkey_virtual_key
        if vk:
            kwargs["virtual_key"] = vk
        if settings.portkey_virtual_key_fallback:
            logfire.warn(
                "Portkey fallback VK ignored without PORTKEY_CONFIG_ID "
                "(inline config disabled). Save a Config in Portkey dashboard "
                "and set PORTKEY_CONFIG_ID=pc-..."
            )

    headers = createHeaders(**kwargs)
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

    model = settings.openai_chat_model

    mode = (
        "config_id"
        if (settings.portkey_config_id or "").strip()
        else ("inline" if settings.portkey_allow_inline_config else "virtual_key")
    )

    logfire.info(
        "Portkey chat client",
        strategy=settings.portkey_strategy,
        cache=settings.portkey_cache_mode,
        config_id=settings.portkey_config_id or None,
        config_mode=mode,
        route=route or "llm",
        feature=feature or "agent_chat",
    )

    return ChatOpenAI(
        model=model,
        api_key=settings.portkey_api_key or "not-needed",
        base_url=PORTKEY_GATEWAY_URL,
        default_headers=headers,
        temperature=temperature,
        max_retries=0,
        timeout=max(1.0, settings.portkey_timeout_ms / 1000.0),
    )


def gateway_status() -> dict[str, Any]:
    """Small diagnostic payload for /ready and scripts."""
    cfg = build_gateway_config() if settings.has_portkey else {}
    config_id = (settings.portkey_config_id or "").strip()
    return {
        "enabled": settings.has_portkey,
        "strategy": settings.portkey_strategy if settings.has_portkey else None,
        "cache_mode": settings.portkey_cache_mode if settings.has_portkey else None,
        "config_id": config_id or None,
        "allow_inline_config": settings.portkey_allow_inline_config,
        "config_mode": (
            "config_id"
            if config_id
            else ("inline" if settings.portkey_allow_inline_config else "virtual_key")
        ),
        "has_primary_vk": bool(
            settings.portkey_virtual_key_primary or settings.portkey_virtual_key
        ),
        "has_fallback_vk": bool(settings.portkey_virtual_key_fallback),
        "dashboard_config_targets": len(cfg.get("targets") or []),
        "environment": settings.portkey_environment,
    }


def describe_inline_config() -> str:
    """Pretty JSON to paste into the Portkey Config dashboard."""
    return json.dumps(build_gateway_config(), indent=2)
