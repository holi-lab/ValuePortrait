from typing import Dict, Type
from .base import BaseApiClient
from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient
from .gemini_client import GeminiClient
from .openrouter_client import OpenRouterClient

class ApiClientFactory:
    _clients: Dict[str, Type[BaseApiClient]] = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
        "gemini": GeminiClient,
        "openrouter": OpenRouterClient
    }
    
    @classmethod
    def create_client(cls, provider: str) -> BaseApiClient:
        """Create and return appropriate API client based on provider"""
        provider = provider.lower()
        client_class = cls._clients.get(provider)
        if not client_class:
            raise ValueError(f"Unsupported provider: {provider}")
        return client_class()