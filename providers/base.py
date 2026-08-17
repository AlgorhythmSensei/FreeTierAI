# providers/base.py
"""Abstract interface for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    text: str
    input_tokens: int
    output_tokens: int
    elapsed_seconds: float
    model_used: str
    error: Optional[str] = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def success(self) -> bool:
        return self.error is None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def chat(self, messages: list[dict]) -> LLMResponse:
        """
        Send a chat request to the provider.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
                     (standard OpenAI format)
        
        Returns:
            LLMResponse with text, tokens, timing, and error info
        """
        pass
