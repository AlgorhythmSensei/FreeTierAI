# providers/registry.py
"""Registry of LLM providers and builder functions."""

import os
import re
import requests
from typing import Optional
from functools import lru_cache
import time
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# PROVIDER CONFIGS
# ---------------------------------------------------------------------------

OPENAI_COMPATIBLE_CONFIGS = {
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
        "website": "https://console.groq.com/keys",
        "models": [
            "mixtral-8x7b-32768",
            "mixtral-8x7b-32768-v0.1",
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "gemma2-9b-it",
        ],
    },
    "Cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "env_key": "CEREBRAS_API_KEY",
        "website": "https://cloud.cerebras.ai/",
        "models": [
            "llama-3.3-70b",
            "llama3.1-70b",
        ],
    },
    "OpenRouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
        "website": "https://openrouter.ai/keys",
        "models": [
            "meta-llama/llama-3.3-70b-instruct",
            "meta-llama/llama-3.1-70b-instruct",
            "mistralai/mistral-7b-instruct",
            "nousresearch/nous-hermes-2-mixtral-8x7b-dpo",
        ],
    },
    "Mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "env_key": "MISTRAL_API_KEY",
        "website": "https://console.mistral.ai/api-keys/",
        "models": [
            "mistral-large-latest",
            "mistral-medium-latest",
            "mistral-small-latest",
            "open-mistral-7b",
        ],
    },
    "NVIDIA NIM": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "env_key": "NVIDIA_API_KEY",
        "website": "https://build.nvidia.com/",
        "models": [
            "meta/llama3-70b-instruct",
            "mistralai/mixtral-8x7b-instruct-v0.1",
        ],
    },
    "Cloudflare Workers AI": {
        "base_url": "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
        "env_key": "CLOUDFLARE_API_TOKEN",
        "account_id_env": "CLOUDFLARE_ACCOUNT_ID",
        "website": "https://dash.cloudflare.com/",
        "models": [
            "@cf/meta/llama-3.1-8b-instruct",
            "@cf/meta/llama-3.2-1b-instruct",
            "@cf/qwen/qwen1.5-0.5b-chat",
        ],
    },
}

NATIVE_PROVIDER_CONFIGS = {
    "Google Gemini": {
        "env_key": "GEMINI_API_KEY",
        "website": "https://aistudio.google.com/apikey",
        "models": [
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
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
    "your_mistral_api_key_here",
    "your_nvidia_nim_api_key_here",
    "your_gemini_api_key_here",
    "your_cloudflare_api_token_here",
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
    """Clear cached dynamic model fetches for one or all providers."""
    if provider_name in {None, "Groq"}:
        _fetch_groq_models.cache_clear()
    if provider_name in {None, "OpenRouter"}:
        _fetch_openrouter_models.cache_clear()
    if provider_name in {None, "Mistral"}:
        _fetch_mistral_models.cache_clear()


@lru_cache(maxsize=8)
def _fetch_groq_models() -> list[str]:
    """Fetch available models from Groq API."""
    try:
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            return OPENAI_COMPATIBLE_CONFIGS["Groq"]["models"]
        
        response = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=3,
        )
        response.raise_for_status()
        models = [m["id"] for m in response.json().get("data", [])]
        return models if models else OPENAI_COMPATIBLE_CONFIGS["Groq"]["models"]
    except Exception:
        return OPENAI_COMPATIBLE_CONFIGS["Groq"]["models"]


@lru_cache(maxsize=8)
def _fetch_openrouter_models() -> list[str]:
    """Fetch available models from OpenRouter API."""
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            timeout=3,
        )
        response.raise_for_status()
        models = [m["id"] for m in response.json().get("data", [])]
        # Filter to more popular/relevant models only
        return models[:20] if models else OPENAI_COMPATIBLE_CONFIGS["OpenRouter"]["models"]
    except Exception:
        return OPENAI_COMPATIBLE_CONFIGS["OpenRouter"]["models"]


@lru_cache(maxsize=8)
def _fetch_mistral_models() -> list[str]:
    """Fetch available models from Mistral API."""
    try:
        api_key = os.getenv("MISTRAL_API_KEY", "")
        if not api_key:
            return OPENAI_COMPATIBLE_CONFIGS["Mistral"]["models"]
        
        response = requests.get(
            "https://api.mistral.ai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=3,
        )
        response.raise_for_status()
        models = [m["id"] for m in response.json().get("data", [])]
        return models if models else OPENAI_COMPATIBLE_CONFIGS["Mistral"]["models"]
    except Exception:
        return OPENAI_COMPATIBLE_CONFIGS["Mistral"]["models"]


# ---------------------------------------------------------------------------
# BUILDER FUNCTIONS
# ---------------------------------------------------------------------------

def get_models(provider_name: str, refresh: bool = False) -> list[str]:
    """
    Get available models for a provider.

    Attempts to fetch dynamically from provider API (Groq, OpenRouter, Mistral)
    when a valid API key is configured. Falls back to hardcoded lists if the API
    call fails or the provider does not support dynamic listing.
    """
    refresh_env_from_file()
    if refresh:
        clear_model_cache(provider_name)

    if provider_name == "Groq":
        return _fetch_groq_models()
    elif provider_name == "OpenRouter":
        return _fetch_openrouter_models()
    elif provider_name == "Mistral":
        return _fetch_mistral_models()
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
