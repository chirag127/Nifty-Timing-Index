"""Central configuration — loads all multi-provider settings from environment variables.

Every provider category (LLM, search, news, email) supports:
- Multiple providers with ENABLE toggle
- Fallback chain: tries providers in order, skips disabled/failed ones
- Feature toggles to disable entire subsystems

MULTI-MODEL SUPPORT:
- Each provider's LLM_<PROVIDER>_MODEL variable supports comma-separated models
- Single model: LLM_GROQ_MODEL=llama-3.3-70b-versatile
- Multiple models: LLM_GROQ_MODEL=llama-3.3-70b-versatile,mixtral-8x7b-32768,llama-3.1-8b-instant
- Each comma-separated model becomes a separate entry in the fusion workflow
- FUSION_SYNTHESIZER can specify provider name (e.g., "groq") or provider/model (e.g., "groq/llama-3.3-70b-versatile")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load .env file if present
load_dotenv()


def _env(key: str, default: str = "") -> str:
    """Get env var with default."""
    return os.environ.get(key, default).strip()


def _env_bool(key: str, default: bool = False) -> bool:
    """Get env var as boolean."""
    val = _env(key).lower()
    if val in ("true", "1", "yes", "on"):
        return True
    if val in ("false", "0", "no", "off"):
        return False
    return default


def _env_float(key: str, default: float = 0.0) -> float:
    """Get env var as float."""
    val = _env(key)
    try:
        return float(val)
    except ValueError:
        return default


def _env_int(key: str, default: int = 0) -> int:
    """Get env var as int."""
    val = _env(key)
    try:
        return int(val)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# LLM Provider Config
# ---------------------------------------------------------------------------


@dataclass
class LLMProviderConfig:
    """Configuration for a single LLM provider (OpenAI-compatible).

    Attributes:
        name: Unique identifier like "groq" or "groq/llama-3.3-70b-versatile" for multi-model
        enabled: Whether this provider is active
        api_key: API key for authentication
        base_url: OpenAI-compatible chat/completions endpoint URL
        model: Model identifier to use
        max_tokens: Maximum tokens in response
        provider_type: "openai_compatible" (default), "cohere", or "huggingface"
            Determines which LangChain class to use for LLM instantiation
    """

    name: str
    enabled: bool
    api_key: str
    base_url: str
    model: str
    max_tokens: int = 2000
    provider_type: str = "openai_compatible"


def _parse_models(provider_prefix: str, default_model: str) -> list[str]:
    """Parse models for a provider from LLM_<PREFIX>_MODEL.

    Supports both single and comma-separated models in the same variable:
      Single:  LLM_GROQ_MODEL=llama-3.3-70b-versatile
      Multi:   LLM_GROQ_MODEL=llama-3.3-70b-versatile,mixtral-8x7b-32768

    Args:
        provider_prefix: Uppercase provider prefix (e.g., "GROQ", "GEMINI")
        default_model: Default model if the env var is not set

    Returns:
        List of model identifier strings (always at least one)
    """
    models_str = _env(f"LLM_{provider_prefix}_MODEL", default_model)
    return [m.strip() for m in models_str.split(",") if m.strip()]


def _build_provider_entries(
    provider_name: str,
    provider_prefix: str,
    default_model: str,
    default_base_url: str,
    provider_type: str = "openai_compatible",
    default_max_tokens: int = 2000,
) -> list[LLMProviderConfig]:
    """Build LLMProviderConfig entries for a provider, supporting multi-model.

    If LLM_<PREFIX>_MODEL contains comma-separated models, one config entry
    is created per model with name like "groq/llama-3.3-70b-versatile".
    If LLM_<PREFIX>_MODEL is a single model, name is just the provider name.

    Args:
        provider_name: Short name for the provider (e.g., "groq", "gemini")
        provider_prefix: Uppercase env var prefix (e.g., "GROQ", "GEMINI")
        default_model: Default model identifier
        default_base_url: Default OpenAI-compatible base URL
        provider_type: "openai_compatible", "cohere", or "huggingface"
        default_max_tokens: Default max tokens for responses

    Returns:
        List of LLMProviderConfig entries (one per model)
    """
    enabled = _env_bool(f"LLM_{provider_prefix}_ENABLED")
    api_key = _env(f"LLM_{provider_prefix}_API_KEY")
    base_url = _env(f"LLM_{provider_prefix}_BASE_URL", default_base_url)
    max_tokens = _env_int(f"LLM_{provider_prefix}_MAX_TOKENS", default_max_tokens)
    models = _parse_models(provider_prefix, default_model)

    entries = []
    for model in models:
        # Use compound name for multi-model, simple name for single
        if len(models) > 1:
            name = f"{provider_name}/{model}"
        else:
            name = provider_name
        entries.append(
            LLMProviderConfig(
                name=name,
                enabled=enabled,
                api_key=api_key,
                base_url=base_url,
                model=model,
                max_tokens=max_tokens,
                provider_type=provider_type,
            )
        )
    return entries


def _get_llm_providers() -> list[LLMProviderConfig]:
    """Load all configured LLM providers from env vars.

    Supports multi-model per provider via comma-separated LLM_<PROVIDER>_MODEL.
    Each model becomes its own entry in the fusion workflow.
    """
    providers: list[LLMProviderConfig] = []

    # --- OpenAI-compatible providers ---
    for name, prefix, default_model, default_url, ptype in [
        ("groq", "GROQ", "llama-3.3-70b-versatile", "https://api.groq.com/openai/v1", "openai_compatible"),
        ("gemini", "GEMINI", "gemini-2.0-flash", "https://generativelanguage.googleapis.com/v1beta/openai/", "openai_compatible"),
        ("cerebras", "CEREBRAS", "llama-3.3-70b", "https://api.cerebras.ai/v1", "openai_compatible"),
        ("openrouter", "OPENROUTER", "meta-llama/llama-3.3-70b-instruct:free", "https://openrouter.ai/api/v1", "openai_compatible"),
        ("together", "TOGETHER", "meta-llama/Llama-3-70b-chat-hf", "https://api.together.xyz/v1", "openai_compatible"),
        ("nvidia", "NVIDIA", "meta/llama-3.1-405b-instruct", "https://integrate.api.nvidia.com/v1", "openai_compatible"),
        ("sambanova", "SAMBANOVA", "Meta-Llama-3.1-70B-Instruct", "https://api.sambanova.ai/v1", "openai_compatible"),
        ("mistral", "MISTRAL", "mistral-large-latest", "https://api.mistral.ai/v1", "openai_compatible"),
        ("cloudflare", "CLOUDFLARE", "@cf/meta/llama-3.1-8b-instruct", "", "openai_compatible"),
    ]:
        providers.extend(
            _build_provider_entries(name, prefix, default_model, default_url, ptype)
        )

    # --- Cohere (uses native SDK via langchain_cohere.ChatCohere) ---
    providers.extend(
        _build_provider_entries(
            "cohere", "COHERE", "command-r-plus", "https://api.cohere.com/v2", "cohere"
        )
    )

    # --- HuggingFace (uses native SDK via langchain_huggingface.ChatHuggingFace) ---
    providers.extend(
        _build_provider_entries(
            "huggingface", "HUGGINGFACE", "meta-llama/Meta-Llama-3-8B-Instruct", "https://api-inference.huggingface.co/v1", "huggingface"
        )
    )

    # Legacy single-provider fallback
    legacy_key = _env("LLM_API_KEY")
    if legacy_key:
        providers.append(
            LLMProviderConfig(
                name="legacy",
                enabled=True,
                api_key=legacy_key,
                base_url=_env("LLM_BASE_URL", "https://api.openai.com/v1"),
                model=_env("LLM_MODEL", "gpt-4o-mini"),
                max_tokens=_env_int("LLM_MAX_TOKENS", 2000),
                provider_type="openai_compatible",
            )
        )
    return providers


# ---------------------------------------------------------------------------
# Search Provider Config
# ---------------------------------------------------------------------------


@dataclass
class SearchProviderConfig:
    """Configuration for a single search API provider."""

    name: str
    enabled: bool
    api_key: str = ""
    base_url: str = ""
    extra: dict[str, str] = field(default_factory=dict)


def _get_search_providers() -> list[SearchProviderConfig]:
    """Load all configured search providers from env vars."""
    return [
        SearchProviderConfig(
            name="serper",
            enabled=_env_bool("SEARCH_SERPER_ENABLED"),
            api_key=_env("SEARCH_SERPER_API_KEY"),
            base_url=_env("SEARCH_SERPER_BASE_URL", "https://google.serper.dev"),
        ),
        SearchProviderConfig(
            name="google_cse",
            enabled=_env_bool("SEARCH_GOOGLE_CSE_ENABLED"),
            api_key=_env("SEARCH_GOOGLE_CSE_API_KEY"),
            base_url=_env("SEARCH_GOOGLE_CSE_BASE_URL", "https://www.googleapis.com/customsearch/v1"),
            extra={"cx": _env("SEARCH_GOOGLE_CSE_CX")},
        ),
        SearchProviderConfig(
            name="tavily",
            enabled=_env_bool("SEARCH_TAVILY_ENABLED"),
            api_key=_env("SEARCH_TAVILY_API_KEY"),
            base_url=_env("SEARCH_TAVILY_BASE_URL", "https://api.tavily.com"),
        ),
        SearchProviderConfig(
            name="brave",
            enabled=_env_bool("SEARCH_BRAVE_ENABLED"),
            api_key=_env("SEARCH_BRAVE_API_KEY"),
            base_url=_env("SEARCH_BRAVE_BASE_URL", "https://api.search.brave.com"),
        ),
        SearchProviderConfig(
            name="duckduckgo",
            enabled=_env_bool("SEARCH_DUCKDUCKGO_ENABLED", True),
            base_url=_env("SEARCH_DUCKDUCKGO_BASE_URL", "https://api.duckduckgo.com"),
        ),
        SearchProviderConfig(
            name="searxng",
            enabled=_env_bool("SEARCH_SEARXNG_ENABLED"),
            base_url=_env("SEARCH_SEARXNG_BASE_URL", "https://searxng.example.com"),
        ),
    ]


# ---------------------------------------------------------------------------
# News Provider Config
# ---------------------------------------------------------------------------


@dataclass
class NewsProviderConfig:
    """Configuration for a single news API provider."""

    name: str
    enabled: bool
    api_key: str = ""
    base_url: str = ""


def _get_news_providers() -> list[NewsProviderConfig]:
    """Load all configured news providers from env vars."""
    return [
        NewsProviderConfig(
            name="rss",
            enabled=_env_bool("NEWS_RSS_ENABLED", True),
        ),
        NewsProviderConfig(
            name="newsapi",
            enabled=_env_bool("NEWS_NEWSAPI_ENABLED"),
            api_key=_env("NEWS_NEWSAPI_API_KEY"),
            base_url=_env("NEWS_NEWSAPI_BASE_URL", "https://newsapi.org/v2"),
        ),
        NewsProviderConfig(
            name="gnews",
            enabled=_env_bool("NEWS_GNEWS_ENABLED"),
            api_key=_env("NEWS_GNEWS_API_KEY"),
            base_url=_env("NEWS_GNEWS_BASE_URL", "https://gnews.io/api/v4"),
        ),
        NewsProviderConfig(
            name="currents",
            enabled=_env_bool("NEWS_CURRENTS_ENABLED"),
            api_key=_env("NEWS_CURRENTS_API_KEY"),
            base_url=_env("NEWS_CURRENTS_BASE_URL", "https://api.currentsapi.services/v1"),
        ),
        NewsProviderConfig(
            name="mediastack",
            enabled=_env_bool("NEWS_MEDIASTACK_ENABLED"),
            api_key=_env("NEWS_MEDIASTACK_API_KEY"),
            base_url=_env("NEWS_MEDIASTACK_BASE_URL", "http://api.mediastack.com/v1"),
        ),
        NewsProviderConfig(
            name="thenewsapi",
            enabled=_env_bool("NEWS_THENEWSAPI_ENABLED"),
            api_key=_env("NEWS_THENEWSAPI_API_KEY"),
            base_url=_env("NEWS_THENEWSAPI_BASE_URL", "https://api.thenewsapi.com/v1"),
        ),
        NewsProviderConfig(
            name="worldnews",
            enabled=_env_bool("NEWS_WORLDNEWS_ENABLED"),
            api_key=_env("NEWS_WORLDNEWS_API_KEY"),
            base_url=_env("NEWS_WORLDNEWS_BASE_URL", "https://worldnewsapi.com/api/v1"),
        ),
    ]


# ---------------------------------------------------------------------------
# Email Provider Config
# ---------------------------------------------------------------------------


@dataclass
class EmailProviderConfig:
    """Configuration for a single email provider."""

    name: str
    enabled: bool
    api_key: str = ""
    base_url: str = ""
    from_address: str = ""
    smtp_host: str = ""
    smtp_port: int = 0


def _get_email_providers() -> list[EmailProviderConfig]:
    """Load all configured email providers from env vars."""
    return [
        EmailProviderConfig(
            name="gmail",
            enabled=_env_bool("EMAIL_GMAIL_ENABLED", True),
            from_address=_env("EMAIL_GMAIL_ADDRESS"),
            smtp_host=_env("EMAIL_GMAIL_SMTP_HOST", "smtp.gmail.com"),
            smtp_port=_env_int("EMAIL_GMAIL_SMTP_PORT", 587),
            api_key=_env("EMAIL_GMAIL_APP_PASSWORD"),
        ),
        EmailProviderConfig(
            name="resend",
            enabled=_env_bool("EMAIL_RESEND_ENABLED"),
            api_key=_env("EMAIL_RESEND_API_KEY"),
            base_url=_env("EMAIL_RESEND_BASE_URL", "https://api.resend.com"),
            from_address=_env("EMAIL_RESEND_FROM"),
        ),
        EmailProviderConfig(
            name="brevo",
            enabled=_env_bool("EMAIL_BREVO_ENABLED"),
            api_key=_env("EMAIL_BREVO_API_KEY"),
            base_url=_env("EMAIL_BREVO_BASE_URL", "https://api.brevo.com/v3"),
            from_address=_env("EMAIL_BREVO_FROM"),
        ),
        EmailProviderConfig(
            name="sendgrid",
            enabled=_env_bool("EMAIL_SENDGRID_ENABLED"),
            api_key=_env("EMAIL_SENDGRID_API_KEY"),
            base_url=_env("EMAIL_SENDGRID_BASE_URL", "https://api.sendgrid.com/v3"),
            from_address=_env("EMAIL_SENDGRID_FROM"),
        ),
    ]


# ---------------------------------------------------------------------------
# Main Settings Class
# ---------------------------------------------------------------------------


class Settings:
    """Central settings — all configuration loaded from environment variables.

    Every provider category supports multiple providers with enable/disable
    toggles and automatic fallback chains.
    """

    def __init__(self) -> None:
        # --- Project ---
        self.project_root = Path(_env("PROJECT_ROOT", str(Path.cwd())))
        self.github_username = _env("GITHUB_USERNAME", "chirag127")
        self.github_token = _env("GITHUB_TOKEN")
        self.github_repo_name = _env("GITHUB_REPO_NAME", "nifty-timing-index")

        # --- Feature Toggles ---
        self.enable_email = _env_bool("NTI_ENABLE_EMAIL", True)
        self.enable_llm = _env_bool("NTI_ENABLE_LLM", True)
        self.enable_screener = _env_bool("NTI_ENABLE_SCREENER", True)
        self.enable_model = _env_bool("NTI_ENABLE_MODEL", True)
        self.enable_search = _env_bool("NTI_ENABLE_SEARCH", True)
        self.enable_news = _env_bool("NTI_ENABLE_NEWS", True)
        self.enable_blog = _env_bool("NTI_ENABLE_BLOG", True)
        self.enable_chromium_scraping = _env_bool("NTI_ENABLE_CHROMIUM_SCRAPING", True)

        # --- LLM Providers ---
        self.llm_providers = _get_llm_providers()
        self.fusion_mode = _env("FUSION_MODE", "all")  # "all" or "sequential"
        self.fusion_synthesizer = _env("FUSION_SYNTHESIZER", "groq")
        # Specific model for the synthesizer (overrides the provider's default model)
        # Format: provider name (e.g., "groq") or provider/model (e.g., "groq/llama-3.3-70b-versatile")
        self.fusion_synthesizer_model = _env("FUSION_SYNTHESIZER_MODEL", "")
        self.fusion_max_concurrent = _env_int("NTI_FUSION_MAX_CONCURRENT_LLM", 5)
        self.fusion_critique_rounds = _env_int("NTI_FUSION_CRITIQUE_ROUNDS", 1)
        self.fusion_timeout_seconds = _env_int("NTI_FUSION_TIMEOUT_SECONDS", 60)

        # --- Search Providers ---
        self.search_providers = _get_search_providers()

        # --- News Providers ---
        self.news_providers = _get_news_providers()

        # --- Email Providers ---
        self.email_providers = _get_email_providers()
        self.alert_email_to = _env("ALERT_EMAIL_TO")
        self.alert_on_zone_change = _env_bool("ALERT_ON_ZONE_CHANGE", True)
        self.alert_on_big_move = _env_bool("ALERT_ON_BIG_MOVE", True)
        self.alert_big_move_threshold = _env_int("ALERT_BIG_MOVE_THRESHOLD", 10)

        # --- Data APIs ---
        self.fred_api_key = _env("FRED_API_KEY")
        self.finnhub_api_key = _env("FINNHUB_API_KEY")

        # --- Cloudflare ---
        self.cloudflare_api_token = _env("CLOUDFLARE_API_TOKEN")
        self.cloudflare_account_id = _env("CLOUDFLARE_ACCOUNT_ID")
        self.cloudflare_email = _env("CLOUDFLARE_EMAIL")

        # --- Langfuse Observability ---
        self.langfuse_enabled = _env_bool("LANGFUSE_ENABLED")
        self.langfuse_public_key = _env("LANGFUSE_PUBLIC_KEY")
        self.langfuse_secret_key = _env("LANGFUSE_SECRET_KEY")
        self.langfuse_base_url = _env("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

        # --- Firebase ---
        self.firebase_project_id = _env("FIREBASE_PROJECT_ID", "nifty-timing-index")
        self.firebase_api_key = _env("FIREBASE_API_KEY")
        self.firebase_auth_domain = _env("FIREBASE_AUTH_DOMAIN")
        self.firebase_app_id = _env("FIREBASE_APP_ID")
        self.firebase_service_account_b64 = _env("FIREBASE_SERVICE_ACCOUNT_BASE64")

        # --- Screener Limits ---
        self.max_pe = _env_float("NTI_MAX_PE", 20.0)
        self.max_pb = _env_float("NTI_MAX_PB", 3.0)
        self.min_market_cap_cr = _env_float("NTI_MIN_MARKET_CAP_CR", 500.0)
        self.psu_boost_score = _env_float("NTI_PSU_BOOST_SCORE", 10.0)

        # --- MTF Risk ---
        self.mtf_leverage = _env_float("NTI_MTF_LEVERAGE", 3.0)
        self.mtf_annual_interest_rate = _env_float("NTI_MTF_ANNUAL_INTEREST_RATE", 0.12)

        # --- Blog ---
        self.blog_word_target_mid = _env_int("NTI_BLOG_WORD_TARGET_MID", 500)
        self.blog_word_target_full = _env_int("NTI_BLOG_WORD_TARGET_FULL", 900)
        self.blog_news_hours = _env_int("NTI_BLOG_NEWS_HOURS", 4)

        # --- Optional ---
        self.sentry_dsn = _env("SENTRY_DSN")

        # --- GIFT Nifty ---
        self.gift_nifty_enabled = _env_bool("GIFT_NIFTY_ENABLED", True)

        # --- MMI Alternative Source ---
        self.mmi_rapidapi_key = _env("MMI_RAPIDAPI_KEY")

    # --- Convenience Methods ---

    def get_enabled_llm_providers(self) -> list[LLMProviderConfig]:
        """Return all enabled LLM providers that have valid API keys.

        For OpenAI-compatible providers, base_url is required.
        For native SDK providers (cohere, huggingface), base_url is optional
        since they use their own SDKs with built-in endpoints.
        """
        result = []
        for p in self.llm_providers:
            if not (p.enabled and p.api_key):
                continue
            provider_type = p.provider_type
            if provider_type == "openai_compatible" and not p.base_url:
                continue
            result.append(p)
        return result

    def get_synthesizer_provider(self) -> LLMProviderConfig | None:
        """Return the configured synthesizer LLM provider.

        FUSION_SYNTHESIZER can be:
          - A provider name like "groq" (uses the first enabled model for that provider)
          - A compound name like "groq/llama-3.3-70b-versatile" (specific model)

        If FUSION_SYNTHESIZER_MODEL is set, it overrides the synthesizer's model
        while still using the FUSION_SYNTHESIZER provider's API key and base_url.
        """
        # Try exact match first (handles compound names like "groq/llama-3.3-70b-versatile")
        for p in self.llm_providers:
            if p.name == self.fusion_synthesizer and p.enabled and p.api_key:
                result = p
                break
        else:
            # Try matching by provider base name (e.g., "groq" matches "groq/llama-3.3-70b-versatile")
            for p in self.llm_providers:
                base_name = p.name.split("/")[0]
                if base_name == self.fusion_synthesizer and p.enabled and p.api_key:
                    result = p
                    break
            else:
                # Fallback to first enabled provider
                enabled = self.get_enabled_llm_providers()
                if not enabled:
                    return None
                result = enabled[0]

        # If FUSION_SYNTHESIZER_MODEL is set, clone the config with that model
        if self.fusion_synthesizer_model and result.model != self.fusion_synthesizer_model:
            result = replace(result, model=self.fusion_synthesizer_model)

        return result

    def get_enabled_search_providers(self) -> list[SearchProviderConfig]:
        """Return all enabled search providers."""
        return [p for p in self.search_providers if p.enabled]

    def get_enabled_news_providers(self) -> list[NewsProviderConfig]:
        """Return all enabled news providers."""
        return [p for p in self.news_providers if p.enabled]

    def get_enabled_email_providers(self) -> list[EmailProviderConfig]:
        """Return all enabled email providers."""
        return [p for p in self.email_providers if p.enabled]

    def has_any_llm(self) -> bool:
        """Check if at least one LLM provider is available."""
        return len(self.get_enabled_llm_providers()) > 0

    def has_any_search(self) -> bool:
        """Check if at least one search provider is available."""
        return len(self.get_enabled_search_providers()) > 0

    def has_any_email(self) -> bool:
        """Check if at least one email provider is available."""
        return len(self.get_enabled_email_providers()) > 0


# Singleton instance
settings = Settings()
