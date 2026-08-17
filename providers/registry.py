# providers/registry.py
"""Registry of LLM providers and builder functions."""

import os
import re
import requests
from typing import Optional
from functools import lru_cache
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# PROVIDER CONFIGS
# ---------------------------------------------------------------------------

OPENAI_COMPATIBLE_CONFIGS = {
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
        "website": "https://console.groq.com/keys",
        # Verified free-tier chat models only — https://console.groq.com/docs/models
        "models": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.6-27b",
        ],
    },
    "Cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "env_key": "CEREBRAS_API_KEY",
        "website": "https://cloud.cerebras.ai/",
        # Verified free-tier chat models only — https://inference-docs.cerebras.ai/introduction
        "models": [
            "gpt-oss-120b",
            "gemma-4-31b",
            "zai-glm-4.7",
        ],
    },
    "OpenRouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
        "website": "https://openrouter.ai/keys",
        # Only :free models — paid models are excluded by design
        "models": [
            "meta-llama/llama-3.3-70b-instruct:free",
            "meta-llama/llama-3.1-8b-instruct:free",
            "mistralai/mistral-7b-instruct:free",
            "google/gemma-3-27b-it:free",
            "qwen/qwen3-235b-a22b:free",
        ],
    },
}

NATIVE_PROVIDER_CONFIGS = {
    "Google Gemini": {
        "env_key": "GEMINI_API_KEY",
        "website": "https://aistudio.google.com/apikey",
        "models": [
            "gemini-flash-lite-latest",
            "gemini-flash-latest",
        ],
    },
}

# ---------------------------------------------------------------------------
# DYNAMIC MODEL FETCHING (with fallback to hardcoded lists)
# ---------------------------------------------------------------------------

_PLACEHOLDER_VALUES = {
    "your_key",
    "your_groq_api_key_here",
    "your_cerebras_api_key_here",
    "your_openrouter_api_key_here",
    "your_gemini_api_key_here",
    "your_api_key_here",
    "changeme",
    "change_me",
    "replace_me",
    "placeholder",
    "example",
    "dummy",
    "fake",
    "test_key",
    "not_a_real_key",
}

_NON_CHAT_MODEL_MARKERS = (
    "embedding",
    "reranker",
    "safety",
    "guard",
    "moderation",
    "whisper",
)


def _is_chat_model(model_id: str) -> bool:
    """Return True for models suitable for chat completions."""
    lowered = model_id.lower()
    return not any(marker in lowered for marker in _NON_CHAT_MODEL_MARKERS)


def refresh_env_from_file() -> None:
    """Reload environment variables from the project .env file."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path, override=True)


def _looks_like_placeholder_key(value: Optional[str]) -> bool:
    if value is None:
        return True

    cleaned = value.strip()
    if not cleaned:
        return True

    lowered = cleaned.lower()
    if lowered in _PLACEHOLDER_VALUES:
        return True

    if re.search(r"(your_|example|placeholder|changeme|replace_me|dummy|fake|test_key|not_a_real_key)", lowered):
        return True

    return False


def _has_real_api_key(value: Optional[str]) -> bool:
    return not _looks_like_placeholder_key(value)


def provider_env_key(provider_name: str) -> Optional[str]:
    """Return the configured environment variable name for a provider."""
    if provider_name in OPENAI_COMPATIBLE_CONFIGS:
        return OPENAI_COMPATIBLE_CONFIGS[provider_name].get("env_key")
    if provider_name in NATIVE_PROVIDER_CONFIGS:
        return NATIVE_PROVIDER_CONFIGS[provider_name].get("env_key")
    return None


def has_api_key(provider_name: str) -> bool:
    """Return True when the provider has a non-placeholder API key in the environment."""
    refresh_env_from_file()
    env_key = provider_env_key(provider_name)
    if not env_key:
        return False
    return _has_real_api_key(os.getenv(env_key, ""))


def clear_model_cache(provider_name: Optional[str] = None) -> None:
    """Clear cached dynamic model fetches (OpenRouter only)."""
    if provider_name in {None, "OpenRouter"}:
        _fetch_openrouter_models.cache_clear()


@lru_cache(maxsize=8)
def _fetch_openrouter_models() -> list[str]:
    """Fetch free-tier models from OpenRouter API (only :free models)."""
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            timeout=3,
        )
        response.raise_for_status()
        models = [
            m["id"] for m in response.json().get("data", [])
            if m["id"].endswith(":free") and _is_chat_model(m["id"])
        ]
        return models if models else OPENAI_COMPATIBLE_CONFIGS["OpenRouter"]["models"]
    except Exception:
        return OPENAI_COMPATIBLE_CONFIGS["OpenRouter"]["models"]


# ---------------------------------------------------------------------------
# BUILDER FUNCTIONS
# ---------------------------------------------------------------------------

def get_models(provider_name: str, refresh: bool = False) -> list[str]:
    """
    Get available models for a provider.

    Returns the curated free-tier model list for each provider.
    OpenRouter models are fetched live (filtered to :free only); all others use
    verified hardcoded lists.
    """
    refresh_env_from_file()
    if refresh:
        clear_model_cache(provider_name)

    if provider_name == "OpenRouter":
        return _fetch_openrouter_models()
    elif provider_name in OPENAI_COMPATIBLE_CONFIGS:
        return OPENAI_COMPATIBLE_CONFIGS[provider_name].get("models", [])
    elif provider_name in NATIVE_PROVIDER_CONFIGS:
        return NATIVE_PROVIDER_CONFIGS[provider_name].get("models", [])
    return []


def get_website(provider_name: str) -> Optional[str]:
    """Get the website/console URL for a provider."""
    if provider_name in OPENAI_COMPATIBLE_CONFIGS:
        return OPENAI_COMPATIBLE_CONFIGS[provider_name].get("website")
    elif provider_name in NATIVE_PROVIDER_CONFIGS:
        return NATIVE_PROVIDER_CONFIGS[provider_name].get("website")
    return None


def missing_key_for(provider_name: str) -> bool:
    """Check if API key is missing for a provider."""
    return not has_api_key(provider_name)


def build_provider(provider_name: str, model_name: str):
    """
    Factory function to build a provider instance.
    
    Args:
        provider_name: Name of an active provider
        model_name: Model to use
    
    Returns:
        LLMProvider instance or raises ValueError
    """
    if provider_name in OPENAI_COMPATIBLE_CONFIGS:
        from providers.openai_compatible import OpenAICompatibleProvider
        config = OPENAI_COMPATIBLE_CONFIGS[provider_name]
        base_url = config["base_url"]
        if config.get("account_id_env"):
            base_url = base_url.format(
                account_id=os.getenv(config["account_id_env"], "")
            )
        return OpenAICompatibleProvider(
            base_url=base_url,
            api_key=os.getenv(config["env_key"], ""),
            model=model_name,
        )
    elif provider_name in NATIVE_PROVIDER_CONFIGS:
        from providers.gemini import GeminiProvider
        config = NATIVE_PROVIDER_CONFIGS[provider_name]
        return GeminiProvider(
            api_key=os.getenv(config["env_key"], ""),
            model=model_name,
        )
    else:
        raise ValueError(f"Unknown provider: {provider_name}")


def list_all_providers() -> list[str]:
    """Return list of all available provider names."""
    return sorted(
        list(OPENAI_COMPATIBLE_CONFIGS.keys()) + list(NATIVE_PROVIDER_CONFIGS.keys())
    )
