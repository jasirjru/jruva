"""Core Application Settings and Environment Validation."""
import sys

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings loaded from environment or .env file."""

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        str_strip_whitespace=True,
    )

    # General
    PROJECT_NAME: str = "Production AI Chatbot"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    GOOGLE_CLIENT_ID: str = Field(default="", description="Trusted Google OAuth application audience")
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    OPERATIONS_RESET_ENABLED: bool = Field(
        default=False,
        description="Permit authenticated admin reset only in development/test; always disabled elsewhere",
    )

    # CORS
    ALLOWED_ORIGINS: str = (
        "http://localhost:3121,http://127.0.0.1:3121,http://localhost:3000,http://127.0.0.1:3000,"
        "http://localhost:3001,http://127.0.0.1:3001,http://localhost:3005,http://127.0.0.1:3005,"
        "http://localhost:4000,http://127.0.0.1:4000,"
        "https://jruva.in,https://www.jruva.in,https://jru-chatbot.vercel.app"
    )

    # AI Model (Gemini Free Tier)
    GEMINI_API_KEY: str = Field(default="", description="Google AI Studio Gemini API Key")
    GEMINI_MODEL: str = "gemini-3.6-flash"
    DEFAULT_TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 2048

    # AI Model (Groq Cloud Free Tier)
    GROQ_API_KEY: str = Field(default="", description="Groq Cloud API Key (https://console.groq.com)")
    GROQ_MODEL: str = "openai/gpt-oss-120b"

    # Groq limits are per model, so every listed model adds its own free daily bucket.
    GROQ_FALLBACK_MODELS: str = Field(
        default="openai/gpt-oss-120b,openai/gpt-oss-20b,qwen/qwen3.8-27b",
        description="Comma-separated Groq models tried in order after the requested one",
    )
    GEMINI_FALLBACK_MODELS: str = Field(
        default="gemini-3.6-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite",
        description="Comma-separated Gemini models tried in order after the requested one",
    )

    # AI Model (OpenRouter Free Tier)
    OPENROUTER_API_KEY: str = Field(default="", description="OpenRouter API Key (https://openrouter.ai)")

    # Extra OpenAI-compatible free tiers (app/services/llm/openai_compat_provider.py).
    # Each vendor is enabled only when its API key is set; BASE_URL and MODELS are optional overrides.
    MISTRAL_API_KEY: str = ""
    MISTRAL_BASE_URL: str = ""
    MISTRAL_MODELS: str = ""
    CLOUDFLARE_API_KEY: str = Field(default="", description="Workers AI token for chat fallback (separate from image CLOUDFLARE_API_TOKEN)")
    CLOUDFLARE_BASE_URL: str = ""
    CLOUDFLARE_MODELS: str = ""
    CEREBRAS_API_KEY: str = ""
    CEREBRAS_BASE_URL: str = ""
    CEREBRAS_MODELS: str = ""
    NVIDIA_API_KEY: str = ""
    NVIDIA_BASE_URL: str = ""
    NVIDIA_MODELS: str = ""
    SAMBANOVA_API_KEY: str = ""
    SAMBANOVA_BASE_URL: str = ""
    SAMBANOVA_MODELS: str = ""

    # Hugging Face Inference API
    HUGGINGFACE_API_KEY: str = Field(default="", description="Hugging Face User Access Token")

    # Web Search (Tavily Search API)
    TAVILY_API_KEY: str = Field(default="", description="Tavily Search API Key (read from server-side environment)")
    TAVILY_SEARCH_DEPTH: str = Field(default="basic", description="Tavily search depth: 'basic' or 'advanced'")
    TAVILY_MAX_RESULTS: int = Field(default=5, ge=1, le=10, description="Max results from Tavily per query")
    TAVILY_TIMEOUT_SECONDS: float = Field(default=5.0, ge=1.0, le=15.0, description="Timeout in seconds for Tavily API")
    TAVILY_CACHE_TTL_SECONDS: int = Field(default=300, ge=0, le=3600, description="TTL in seconds for cached search results")
    TAVILY_CIRCUIT_BREAKER_SECONDS: int = Field(default=60, ge=10, le=600, description="Cooldown period when Tavily consecutive failures occur")
    TAVILY_MAX_REQUESTS_PER_MINUTE: int = Field(default=20, ge=1, le=120, description="Search rate admission limit to prevent free-tier exhaustion")
    # Second search provider (Jina s.jina.ai, free key grant). Works from datacenter IPs where DuckDuckGo blocks.
    JINA_API_KEY: str = Field(default="", description="Jina Search API key; empty disables the provider")
    JINA_TIMEOUT_SECONDS: float = Field(default=6.0, ge=1.0, le=15.0, description="Timeout in seconds for Jina search")

    # Image inference is opt-in. Free-plan attestation must be confirmed by account owner.
    IMAGE_GENERATION_ENABLED: bool = False
    CLOUDFLARE_ACCOUNT_ID: str = ""
    CLOUDFLARE_API_TOKEN: str = ""
    IMAGE_CF_FREE_PLAN_VERIFIED: bool = False
    IMAGE_CF_KLEIN_ENABLED: bool = False
    IMAGE_CF_KLEIN_FREE_VERIFIED: bool = False
    IMAGE_DAILY_NEURON_LIMIT: int = Field(default=1000, ge=1, le=10000)
    IMAGE_USER_DAILY_ATTEMPTS: int = Field(default=5, ge=1, le=100)
    IMAGE_MAX_USER_COUNT: int = Field(default=25, ge=1, le=1000, description="Max retained images per user")
    IMAGE_MAX_USER_BYTES: int = Field(default=50 * 1024 * 1024, ge=1024 * 1024, description="Max retained image bytes per user (50 MB)")
    IMAGE_MAX_PROJECT_BYTES: int = Field(default=500 * 1024 * 1024, ge=10 * 1024 * 1024, description="Max retained image bytes project-wide (500 MB)")
    IMAGE_MAX_OBJECT_BYTES: int = Field(default=10 * 1024 * 1024, ge=1024 * 1024, description="Max single image file size in bytes (10 MB)")

    # Document Storage Quotas (Phase 2)
    DOCUMENT_MAX_USER_COUNT: int = Field(default=25, ge=1, le=500, description="Max documents per user")
    DOCUMENT_MAX_USER_BYTES: int = Field(default=50 * 1024 * 1024, ge=1024 * 1024, description="Max document bytes per user (50 MB)")
    DOCUMENT_MAX_PROJECT_BYTES: int = Field(default=500 * 1024 * 1024, ge=10 * 1024 * 1024, description="Max document bytes project-wide (500 MB)")
    DOCUMENT_MAX_OBJECT_BYTES: int = Field(default=15 * 1024 * 1024, ge=1024 * 1024, description="Max single document file size (15 MB)")
    STORAGE_CAPACITY_WARN_PCT: float = Field(default=0.80, ge=0.5, le=0.99, description="Storage warning threshold (80%)")
    STORAGE_CAPACITY_THROTTLE_PCT: float = Field(default=0.90, ge=0.5, le=0.99, description="Storage throttle threshold (90%)")

    # Daily User Allowances (Phase 3 - Cloud Free-Tier Protection)
    ENABLE_DAILY_PROMPT_LIMIT: bool = Field(
        default=False,
        description="Enable fixed application-level daily chat prompt-count limit (disabled for public beta)",
    )
    DAILY_PROMPTS_PER_USER: int = Field(default=20, ge=0, le=100000, description="Daily chat prompts per user (enforced only when ENABLE_DAILY_PROMPT_LIMIT=True)")
    DAILY_SEARCHES_PER_USER: int = Field(default=3, ge=0, le=100, description="Daily web searches per user")
    # All guests share the "anonymous" bucket, so it needs a pool-sized limit (Tavily free = 1,000/month).
    DAILY_GUEST_SEARCH_POOL: int = Field(default=20, ge=0, le=500, description="Daily paid web searches shared by all guests")
    DAILY_UPLOADS_PER_USER: int = Field(default=5, ge=0, le=50, description="Daily document uploads per user")
    DAILY_AUDIO_SECONDS_PER_USER: int = Field(default=120, ge=0, le=600, description="Daily audio seconds per user (2 min)")
    GUEST_DAILY_AUDIO_SECONDS: int = Field(
        default=7200, ge=0, le=28800,
        description="Daily audio seconds shared by all guests (one quota row); Groq Whisper free tier is 28,800 s/day",
    )
    DAILY_GLOBAL_PROMPT_BUDGET: int = Field(default=500, ge=0, le=100000, description="Project-wide daily prompt cap (enforced only when ENABLE_DAILY_PROMPT_LIMIT=True)")

    # Cloudflare R2 Expansion (Phase 2F - Disabled Feature Flag)
    ENABLE_R2_STORAGE: bool = Field(default=False, description="Cloudflare R2 storage adapter (flagged off)")
    R2_ACCOUNT_ID: str = Field(default="", description="Cloudflare Account ID for R2")
    R2_ACCESS_KEY_ID: str = Field(default="", description="Cloudflare R2 Access Key ID")
    R2_SECRET_ACCESS_KEY: str = Field(default="", description="Cloudflare R2 Secret Access Key")
    R2_BUCKET_NAME: str = Field(default="jruva-documents-r2", description="Cloudflare R2 Bucket Name")
    R2_ENDPOINT_URL: str = Field(default="", description="Custom S3 endpoint URL for R2")

    # Voice Input (Phase 5 - Disabled Feature Flag)
    ENABLE_VOICE_TRANSCRIPTION: bool = Field(default=False, description="Groq Whisper hosted voice transcription (flagged off)")
    VOICE_MAX_FILE_BYTES: int = Field(default=10 * 1024 * 1024, description="Max audio upload size (10 MB)")
    VOICE_MAX_DURATION_SECONDS: int = Field(default=120, description="Max audio duration in seconds (2 min)")

    # Video Understanding (Phase 6 - Disabled Feature Flag)
    ENABLE_VIDEO_UNDERSTANDING: bool = Field(default=False, description="Cloud video and lecture understanding (flagged off)")

    # Multi-Provider Routing
    DEFAULT_PROVIDER: str = "auto"  # "auto", "gemini", "groq", or "openrouter"
    ENABLE_AUTO_FAILOVER: bool = True
    LOCAL_LLM_DIRECT_TIMEOUT_SECONDS: float = Field(
        default=120.0, ge=1.0, le=300.0,
        description="Bounded inference budget for explicitly selected local models, including cold start",
    )
    ALLOW_LOCAL_DEV_OLLAMA: bool = False

    # Free Tier Safeguards & Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    BENCHMARK_MODE: bool = False
    BENCHMARK_BYPASS_SECRET: str = ""

    # Phase 3: Fine-Tuned Model Configuration (Flagged OFF by default for cloud operation)
    ENABLE_FINETUNED_TIER: bool = False
    FINETUNED_MODEL_ID: str = "openai/gpt-oss-120b"
    ACTIVE_FINETUNED_VERSION: str = "cloud-v1"

    # Phase 4: Shadow-Mode Routing & Graduated Rollout
    ENABLE_SHADOW_MODE: bool = False
    SHADOW_TRAFFIC_PCT: float = 0.0
    SHADOW_MODEL_ID: str = "openai/gpt-oss-120b"
    TRAFFIC_ROLLOUT_PCT: float = 0.0  # 0% to 100% of real traffic to fine-tuned model

    # Write-Quiescence & Planned Maintenance Gate
    MAINTENANCE_MODE: bool = Field(
        default=False,
        description="When true, write operations, mutations, and external dispatch are suspended for safe maintenance.",
    )

    # Database (SQLite default for local dev, PostgreSQL for production)
    DATABASE_BACKEND: str = Field(
        default="sqlite",
        description="Active persistent backend: 'sqlite' or 'postgres'",
    )
    DATABASE_URL: str = Field(
        default="",
        description="Primary PostgreSQL connection string (Supabase, Neon, Cloud SQL, or local)",
    )
    SUPABASE_DATABASE_URL: str = Field(
        default="",
        description="Legacy alias for Supabase connection string",
    )
    SUPABASE_URL: str = Field(default="", description="Supabase project URL (Storage REST API)")
    SUPABASE_SERVICE_ROLE_KEY: str = ""
    DB_POOL_SIZE: int = Field(default=5, description="Connection pool size for Cloud Run serverless instances")
    DB_MAX_OVERFLOW: int = Field(default=5, description="Max overflow connections beyond pool size")
    DB_POOL_TIMEOUT: float = Field(default=30.0, description="Pool checkout timeout in seconds")
    DB_POOL_RECYCLE: int = Field(default=300, description="Recycle connections every N seconds")
    DB_POOL_PRE_PING: bool = Field(default=True, description="Test connection liveness on checkout")
    DB_STATEMENT_CACHE_SIZE: int = Field(
        default=0,
        description="Prepared statement cache size (must be 0 for transaction-mode poolers)",
    )

    # Blob and Media Storage
    STORAGE_BACKEND: str = Field(
        default="local",
        description="Active persistent blob storage backend: 'local', 'supabase', 's3', 'gcs'",
    )
    STORAGE_BUCKET_NAME: str = Field(
        default="jruva-documents",
        description="Active durable storage bucket name",
    )
    ENABLE_DOCUMENT_UPLOADS: bool = Field(
        default=True,
        description="Whether document uploading is enabled. Automatically disabled in production if STORAGE_BACKEND is local.",
    )

    @property
    def effective_enable_document_uploads(self) -> bool:
        """Resolve document uploads enabled status with fail-closed production check."""
        import os
        env_val = os.getenv("ENABLE_DOCUMENT_UPLOADS")
        if env_val is not None:
            return env_val.lower().strip() in ("1", "true", "yes", "on")
        if (os.getenv("ENVIRONMENT") or self.ENVIRONMENT).lower().strip() == "production" and self.effective_storage_backend == "local":
            return False
        return bool(self.ENABLE_DOCUMENT_UPLOADS)

    @property
    def effective_storage_backend(self) -> str:
        """Resolve active storage backend respecting process environment override."""
        import os
        return (os.getenv("STORAGE_BACKEND") or self.STORAGE_BACKEND).lower().strip()

    @property
    def effective_database_url(self) -> str:
        """Resolve and normalize database connection string for asyncpg or aiosqlite."""
        url = (self.DATABASE_URL or self.SUPABASE_DATABASE_URL or "").strip()
        if not url:
            return ""
        # Normalize protocol for SQLAlchemy asyncpg driver
        if url.startswith("postgres://"):
            url = "postgresql+asyncpg://" + url[len("postgres://"):]
        elif url.startswith("postgresql://") and not url.startswith("postgresql+"):
            url = "postgresql+asyncpg://" + url[len("postgresql://"):]
        elif url.startswith("sqlite://") and not url.startswith("sqlite+"):
            url = "sqlite+aiosqlite://" + url[len("sqlite://"):]
        return url

    # Vector Database (Qdrant Standalone Server or Cloud)
    QDRANT_URL: str = Field(
        default="",
        description="Standalone Qdrant server URL (e.g., http://qdrant:6333 or cloud URL)",
    )
    QDRANT_API_KEY: str = Field(
        default="",
        description="Qdrant Cloud API Key if using managed cluster",
    )

    # Phase 5: RAG Resource & Memory Management (512MB Guard)
    LOW_MEMORY_RAG_MODE: bool = Field(
        default=True,
        description="Enable low-memory RAG mode for constrained environments (e.g. Render 512MB Free Tier)",
    )
    ENABLE_FLASHRANK: bool = Field(
        default=False,
        description="Enable FlashRank cross-encoder reranker (requires >512MB RAM)",
    )
    ENABLE_LOCAL_OCR: bool = Field(
        default=False,
        description="Enable local ONNX RapidOCR engine (requires >512MB RAM; disabled by default to protect container memory ceilings)",
    )
    ENABLE_LOCAL_EMBEDDINGS: bool = Field(
        default=True,
        description="Enable local FastEmbed dense embeddings (guarded by MAX_PROCESS_RSS_MB 380MB)",
    )
    ENABLE_LOCAL_VISUAL_ANALYSIS: bool = Field(
        default=False,
        description="Opt in to local image classification/enhancement in production; requires a measured memory budget",
    )

    @property
    def local_visual_analysis_enabled(self) -> bool:
        """Development retains CV; production delegates pixels to cloud vision by default."""
        return self.ENVIRONMENT.lower().strip() != "production" or self.ENABLE_LOCAL_VISUAL_ANALYSIS

    MAX_PROCESS_RSS_MB: float = Field(
        default=380.0,
        description="Process RSS memory safety ceiling in MiB before degrading to lightweight retrieval",
    )

    @property
    def cors_origins(self) -> list[str]:
        """Parse comma-separated origins string into a trimmed list."""
        if not self.ALLOWED_ORIGINS:
            # Never fall back to "*" with credentialed CORS outside local development.
            if self.ENVIRONMENT.strip().lower() == "development":
                return ["http://localhost:3000"]
            return []
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin.strip()]


_existing = getattr(sys, "_jruva_global_settings", None)
if _existing is not None and type(_existing).__name__ == "Settings":
    fresh = Settings()
    _existing.__dict__.clear()
    _existing.__dict__.update(fresh.__dict__)
    if hasattr(_existing, "__pydantic_fields_set__") and hasattr(fresh, "__pydantic_fields_set__"):
        _existing.__pydantic_fields_set__.clear()
        _existing.__pydantic_fields_set__.update(fresh.__pydantic_fields_set__)
    settings = _existing
else:
    settings = Settings()
    sys._jruva_global_settings = settings

