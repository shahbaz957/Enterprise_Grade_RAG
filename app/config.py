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
    backend_url: str = "http://localhost:8000"
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3001",
        ]
    )

    # --- Auth (prod): Bearer RAG_API_KEY or HS256 JWT ---
    rag_api_key: str = ""
    rag_jwt_secret: str = ""
    # When true, always require auth. Otherwise: require when DEBUG=false and RAG_API_KEY set.
    rag_auth_required: bool = False

    # --- Rate limit (Upstash Redis REST in cloud; in-memory locally) ---
    upstash_redis_rest_url: str = ""
    upstash_redis_rest_token: str = ""
    rate_limit_per_minute: int = 60
    rate_limit_window_seconds: int = 60

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

    # --- Portkey gateway (LLM routing / fallback / cache / cost logs) ---
    portkey_api_key: str = ""
    # Legacy single VK; prefer primary/fallback below.
    portkey_virtual_key: str = ""
    portkey_virtual_key_primary: str = ""  # OpenAI virtual key slug (primary)
    portkey_virtual_key_fallback: str = ""  # Groq virtual key slug (fallback)
    # Saved Config ID (pc-...). Required for fallback/cache when org blocks inline config.
    portkey_config_id: str = ""
    # Most orgs block inline JSON in x-portkey-config — keep False unless Portkey allows it.
    portkey_allow_inline_config: bool = False
    portkey_strategy: str = "fallback"  # fallback | loadbalance
    portkey_primary_weight: float = 0.7
    portkey_fallback_weight: float = 0.3
    portkey_cache_mode: str = "simple"  # off | simple | semantic
    portkey_cache_max_age: int = 3600  # seconds
    portkey_retry_attempts: int = 3
    portkey_timeout_ms: int = 30_000
    portkey_environment: str = "dev"
    portkey_enabled: bool = True

    # True = OpenAI for guardrails / judge / direct LLM. False = Groq (old default).
    # Does not change Portkey routing — Portkey uses PRIMARY/FALLBACK virtual keys.
    use_openai_llm: bool = True

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

    # --- Neon / Postgres (chat session memory) ---
    database_url: str = ""

    # --- NeMo Guardrails (local Colang config; reuses Groq/OpenAI — no NVIDIA key) ---
    guardrails_enabled: bool = True
    guardrails_config_path: str = "app/guardrails/config"
    # If True, skip rails when NeMo init/check fails; default fail-closed.
    guardrails_fail_open: bool = False

    # --- Paths ---
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
    def has_portkey(self) -> bool:
        """Portkey gateway ready: API key + enabled + at least one virtual key (or config id)."""
        if not self.portkey_enabled:
            return False
        if not self.portkey_api_key:
            return False
        if self.portkey_config_id:
            return True
        return bool(
            self.portkey_virtual_key_primary
            or self.portkey_virtual_key
            or self.portkey_virtual_key_fallback
        )

    @property
    def has_jina(self) -> bool:
        return bool(self.jina_api_key)

    @property
    def has_langfuse(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @property
    def has_database(self) -> bool:
        return bool(self.database_url)

    @property
    def root_dir(self) -> Path:
        return ROOT_DIR

    @property
    def has_guardrails(self) -> bool:
        """Enabled flag + checker LLM + config directory present."""
        if not self.guardrails_enabled:
            return False
        if not (self.has_groq or self.has_openai):
            return False
        raw = (self.guardrails_config_path or "").strip()
        path = Path(raw) if raw else ROOT_DIR / "app" / "guardrails" / "config"
        if not path.is_absolute():
            path = ROOT_DIR / path
        return path.is_dir()

    @property
    def auth_required(self) -> bool:
        """Protect /query and /sessions* in non-local setups."""
        if self.rag_auth_required:
            return True
        if self.debug:
            return False
        return bool(self.rag_api_key)

    @property
    def has_upstash(self) -> bool:
        return bool(self.upstash_redis_rest_url and self.upstash_redis_rest_token)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
