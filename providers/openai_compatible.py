# providers/openai_compatible.py
"""Adapter for OpenAI-compatible providers (Groq, Cerebras, OpenRouter, Mistral, NVIDIA NIM)."""

import time
from typing import Optional

from providers.base import LLMProvider, LLMResponse


class OpenAICompatibleProvider(LLMProvider):
    """
    Adapter for providers that speak the OpenAI Chat Completions wire format.
    
    Covers: Groq, Cerebras, OpenRouter, Mistral, NVIDIA NIM
    """

    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self._client = None

    @property
    def client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def chat(self, messages: list[dict]) -> LLMResponse:
        """Send a chat request to the provider."""
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
            )
            
            elapsed = time.time() - start_time
            
            return LLMResponse(
                text=response.choices[0].message.content,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                elapsed_seconds=elapsed,
                model_used=response.model,
                error=None,
            )
        except Exception as e:
            elapsed = time.time() - start_time
            return LLMResponse(
                text="",
                input_tokens=0,
                output_tokens=0,
                elapsed_seconds=elapsed,
                model_used=self.model,
                error=str(e),
            )
