"""Adapter for the Google Gemini API."""

import time

from google import genai

from providers.base import LLMProvider, LLMResponse


class GeminiProvider(LLMProvider):
    """Adapter for Google Gemini using the google-genai SDK."""

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self._client = genai.Client(api_key=api_key)

    def chat(self, messages: list[dict]) -> LLMResponse:
        """Send a chat request to Gemini."""
        contents = []
        system_instruction = None

        for message in messages:
            if message["role"] == "system":
                system_instruction = message["content"]
                continue
            role = "model" if message["role"] == "assistant" else "user"
            contents.append({
                "role": role,
                "parts": [{"text": message["content"]}],
            })

        start = time.perf_counter()
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config={"system_instruction": system_instruction}
                if system_instruction
                else None,
            )
        except Exception as error:
            elapsed = time.perf_counter() - start
            return LLMResponse(
                text="",
                input_tokens=0,
                output_tokens=0,
                elapsed_seconds=elapsed,
                model_used=self.model,
                error=str(error),
            )

        elapsed = time.perf_counter() - start
        usage = getattr(response, "usage_metadata", None)
        return LLMResponse(
            text=response.text or "",
            input_tokens=getattr(usage, "prompt_token_count", 0) if usage else 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) if usage else 0,
            elapsed_seconds=elapsed,
            model_used=self.model,
            error=None,
        )