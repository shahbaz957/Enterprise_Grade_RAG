"""Compatibility helpers for RAGAS under current langchain-community versions."""

from __future__ import annotations


def ensure_ragas_imports() -> None:
    """Stub removed ChatVertexAI so ragas can import on langchain-community>=0.4."""
    import sys
    import types

    name = "langchain_community.chat_models.vertexai"
    if name in sys.modules:
        return
    mod = types.ModuleType(name)

    class ChatVertexAI:  # noqa: N801 - match upstream symbol
        """Placeholder; ragas only imports the symbol at module load."""

    mod.ChatVertexAI = ChatVertexAI
    sys.modules[name] = mod
