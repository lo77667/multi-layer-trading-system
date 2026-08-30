"""LLM Provider abstraction layer for multiple LLM services."""

import asyncio
import logging
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import os

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Base class for LLM providers."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "", **kwargs):
        self.api_key = api_key or os.getenv(f"{self.__class__.__name__.upper()}_API_KEY")
        self.model_name = model_name
        self.kwargs = kwargs
    
    @abstractmethod
    async def generate(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.7) -> str:
        """Generate response from LLM."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI (GPT) provider."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4-turbo", **kwargs):
        super().__init__(api_key, model_name, **kwargs)
        self.api_base = kwargs.get('api_base', 'https://api.openai.com/v1')
    
    async def generate(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.7) -> str:
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_base}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                    timeout=30.0,
                )
                result = response.json()
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise


class ClaudeProvider(BaseLLMProvider):
    """Anthropic Claude provider."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "claude-3-opus", **kwargs):
        super().__init__(api_key, model_name, **kwargs)
    
    async def generate(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.7) -> str:
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                    },
                    json={
                        "model": self.model_name,
                        "max_tokens": max_tokens,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temperature,
                    },
                    timeout=30.0,
                )
                result = response.json()
                return result["content"][0]["text"]
        except Exception as e:
            logger.error(f"Claude generation failed: {e}")
            raise


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-pro", **kwargs):
        super().__init__(api_key, model_name, **kwargs)
    
    async def generate(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.7) -> str:
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent",
                    params={"key": self.api_key},
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "maxOutputTokens": max_tokens,
                            "temperature": temperature,
                        },
                    },
                    timeout=30.0,
                )
                result = response.json()
                return result["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek provider."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "deepseek-chat", **kwargs):
        super().__init__(api_key, model_name, **kwargs)
        self.api_base = kwargs.get('api_base', 'https://api.deepseek.com/v1')
    
    async def generate(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.7) -> str:
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_base}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                    timeout=30.0,
                )
                result = response.json()
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"DeepSeek generation failed: {e}")
            raise


class QwenProvider(BaseLLMProvider):
    """Alibaba Qwen provider."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "qwen-plus", **kwargs):
        super().__init__(api_key, model_name, **kwargs)
    
    async def generate(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.7) -> str:
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model_name,
                        "input": {"messages": [{"role": "user", "content": prompt}]},
                        "parameters": {
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                        },
                    },
                    timeout=30.0,
                )
                result = response.json()
                return result["output"]["text"]
        except Exception as e:
            logger.error(f"Qwen generation failed: {e}")
            raise


class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "llama2", **kwargs):
        super().__init__(api_key, model_name, **kwargs)
        self.api_base = kwargs.get('api_base', 'http://localhost:11434')
    
    async def generate(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.7) -> str:
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_base}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens,
                        },
                    },
                    timeout=60.0,
                )
                result = response.json()
                return result["response"]
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise


class vLLMProvider(BaseLLMProvider):
    """vLLM inference server provider."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "meta-llama/Llama-2-7b", **kwargs):
        super().__init__(api_key, model_name, **kwargs)
        self.api_base = kwargs.get('api_base', 'http://localhost:8000')
    
    async def generate(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.7) -> str:
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_base}/v1/completions",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                    timeout=60.0,
                )
                result = response.json()
                return result["choices"][0]["text"]
        except Exception as e:
            logger.error(f"vLLM generation failed: {e}")
            raise


class LLMProviderFactory:
    """Factory for creating LLM providers."""
    
    _providers = {
        'openai': OpenAIProvider,
        'claude': ClaudeProvider,
        'gemini': GeminiProvider,
        'deepseek': DeepSeekProvider,
        'qwen': QwenProvider,
        'ollama': OllamaProvider,
        'vllm': vLLMProvider,
    }
    
    @classmethod
    def create(cls, provider_name: str, **kwargs) -> BaseLLMProvider:
        """Create an LLM provider instance."""
        provider_class = cls._providers.get(provider_name.lower())
        if not provider_class:
            raise ValueError(f"Unknown provider: {provider_name}")
        return provider_class(**kwargs)
    
    @classmethod
    def get_available_providers(cls) -> list:
        """Get list of available providers."""
        return list(cls._providers.keys())
