"""API key / JWT gate for non-local deployments.

When ``settings.auth_required`` is true, protected routes expect
``Authorization: Bearer <token>`` where token is either ``RAG_API_KEY``
or an HS256 JWT signed with ``RAG_JWT_SECRET`` (must include ``sub``).
"""

from __future__ import annotations

from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings

_bearer = HTTPBearer(auto_error=False)


def _verify_jwt(token: str) -> dict:
    secret = (settings.rag_jwt_secret or "").strip()
    if not secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="JWT auth is not configured (set RAG_JWT_SECRET)",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid JWT: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    if not payload.get("sub"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="JWT missing required claim: sub",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload


async def require_auth(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Security(_bearer),
    ] = None,
) -> str | None:
    """Return principal id (api_key / jwt sub) or None when auth is off."""
    if not settings.auth_required:
        return None

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = (credentials.credentials or "").strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Empty Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    api_key = (settings.rag_api_key or "").strip()
    if api_key and token == api_key:
        return "api_key"

    if (settings.rag_jwt_secret or "").strip():
        payload = _verify_jwt(token)
        return str(payload["sub"])

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Auth required but RAG_API_KEY / RAG_JWT_SECRET not configured",
        )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key or JWT",
        headers={"WWW-Authenticate": "Bearer"},
    )


AuthPrincipal = Annotated[str | None, Depends(require_auth)]
