from functools import lru_cache
from pathlib import Path

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Backend settings loaded from environment / `.env` at the repo root."""

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- App ---
    app_name: str = "Enterprise RAG Backend"
    app_version: str = "0.1.0"
    debug: bool = False
    api_prefix: str = "/api/v1"
    backend_url: str = "http://localhost:8000"
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
    )

    # --- OpenAI (embeddings + optional chat) ---
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"
    embedding_dimensions: int = 1536

    # --- Groq (inference) ---
    groq_api_key: str = ""
    groq_fallback_api_key: str = ""
    groq_chat_model: str = "llama-3.3-70b-versatile"
    judge_groq_api_key: str = ""

    # --- Portkey gateway ---
    portkey_api_key: str = ""
    portkey_virtual_key: str = ""

    # --- Jina (reranker) ---
    jina_api_key: str = ""
    jina_rerank_model: str = "jina-reranker-v2-base-multilingual"
    jina_rerank_top_n: int = 15

    # --- Qdrant ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_cluster_endpoint: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "enterprise_rag"

    # --- Observability ---
    logfire_token: str = ""
    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""
    langfuse_base_url: str = "https://cloud.langfuse.com"

    # --- Paths ---
    data_dir: Path = ROOT_DIR / "DATA"
    true_data_dir: Path = ROOT_DIR / "DATA" / "true_data"
    noisy_data_dir: Path = ROOT_DIR / "DATA" / "noisy_data"
    processed_data_dir: Path = ROOT_DIR / "processed_data"

    @computed_field 
    @property
    def qdrant_endpoint(self) -> str:
        """Prefer cloud cluster URL when set; otherwise local URL."""
        return (self.qdrant_cluster_endpoint or self.qdrant_url).rstrip("/")

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def has_groq(self) -> bool:
        return bool(self.groq_api_key or self.groq_fallback_api_key)

    @property
    def active_groq_api_key(self) -> str:
        return self.groq_api_key or self.groq_fallback_api_key

    @property
    def has_qdrant(self) -> bool:
        return bool(self.qdrant_endpoint)

    @property
    def has_portkey(self) -> bool:
        return bool(self.portkey_api_key)

    @property
    def has_jina(self) -> bool:
        return bool(self.jina_api_key)

    @property
    def has_langfuse(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
