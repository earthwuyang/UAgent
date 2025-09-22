"""LLM client for classification and other AI tasks"""

import json
import logging
import os
import asyncio
import random
import time
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from typing import AsyncIterator

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import dashscope
    from dashscope import Generation
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False


class LLMClient(ABC):
    """Abstract base class for LLM clients"""

    @abstractmethod
    async def classify(self, request: str, prompt: str) -> Dict[str, Any]:
        """Classify a request using the LLM"""
        pass

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using the LLM"""
        pass

    @abstractmethod
    async def stream_generate(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Stream generate text using the LLM"""
        pass


class AnthropicClient(LLMClient):
    """Anthropic Claude client for LLM operations"""

    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        """Initialize Anthropic client

        Args:
            api_key: Anthropic API key
            model: Model name to use
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package is required for AnthropicClient")

        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model
        self.logger = logging.getLogger(__name__)

    async def classify(self, request: str, prompt: str) -> Dict[str, Any]:
        """Classify a request using Claude

        Args:
            request: User request to classify
            prompt: Classification prompt

        Returns:
            Classification result as dictionary
        """
        try:
            full_prompt = f"{prompt}\n\nUser request: {request}\n\nClassification:"

            response = await self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.1,
                messages=[{
                    "role": "user",
                    "content": full_prompt
                }]
            )

            # Extract JSON from response
            content = response.content[0].text

            # Try to parse JSON response
            try:
                result = json.loads(content)
                return result
            except json.JSONDecodeError:
                # Fallback: extract JSON from text
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    raise ValueError(f"Could not parse JSON from response: {content}")

        except Exception as e:
            self.logger.error(f"Anthropic classification error: {e}")
            raise

    async def generate(self, prompt: str, max_tokens: int = 1000, **kwargs) -> str:
        """Generate text using Claude

        Args:
            prompt: Generation prompt
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=kwargs.get("temperature", 0.7),
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            return response.content[0].text

        except Exception as e:
            self.logger.error(f"Anthropic generation error: {e}")
            raise

    async def stream_generate(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Stream generate text using Claude (placeholder - not implemented)"""
        # For now, fall back to non-streaming
        result = await self.generate(prompt, **kwargs)
        yield result


class OpenAIClient(LLMClient):
    """OpenAI GPT client for LLM operations"""

    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
        """Initialize OpenAI client

        Args:
            api_key: OpenAI API key
            model: Model name to use
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package is required for OpenAIClient")

        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
        self.logger = logging.getLogger(__name__)

    async def classify(self, request: str, prompt: str) -> Dict[str, Any]:
        """Classify a request using GPT

        Args:
            request: User request to classify
            prompt: Classification prompt

        Returns:
            Classification result as dictionary
        """
        try:
            full_prompt = f"{prompt}\n\nUser request: {request}\n\nClassification:"

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": full_prompt
                }],
                temperature=0.1,
                max_tokens=1000
            )

            content = response.choices[0].message.content

            # Try to parse JSON response
            try:
                result = json.loads(content)
                return result
            except json.JSONDecodeError:
                # Fallback: extract JSON from text
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    raise ValueError(f"Could not parse JSON from response: {content}")

        except Exception as e:
            self.logger.error(f"OpenAI classification error: {e}")
            raise

    async def generate(self, prompt: str, max_tokens: int = 1000, **kwargs) -> str:
        """Generate text using GPT

        Args:
            prompt: Generation prompt
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=max_tokens
            )

            return response.choices[0].message.content

        except Exception as e:
            self.logger.error(f"OpenAI generation error: {e}")
            raise

    async def stream_generate(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Stream generate text using GPT (placeholder - not implemented)"""
        # For now, fall back to non-streaming
        result = await self.generate(prompt, **kwargs)
        yield result


class DashScopeClient(LLMClient):
    """DashScope Qwen client for LLM operations"""

    def __init__(self, api_key: str, model: str = "qwen-max-latest"):
        """Initialize DashScope client

        Args:
            api_key: DashScope API key
            model: Model name to use (default: qwen-max-latest)
        """
        if not DASHSCOPE_AVAILABLE:
            raise ImportError("dashscope package is required for DashScopeClient")

        dashscope.api_key = api_key
        self.model = model
        self.logger = logging.getLogger(__name__)
        # Global async concurrency limiter (per-process)
        max_concurrent = int(os.getenv("LLM_MAX_CONCURRENT", "4"))
        # Use a class-level semaphore shared across instances
        if not hasattr(DashScopeClient, "_sem"):
            DashScopeClient._sem = asyncio.Semaphore(max_concurrent)

    async def classify(self, request: str, prompt: str) -> Dict[str, Any]:
        """Classify a request using Qwen

        Args:
            request: User request to classify
            prompt: Classification prompt

        Returns:
            Classification result as dictionary
        """
        try:
            full_prompt = f"{prompt}\n\nUser request: {request}\n\nClassification (respond with valid JSON only):"

            # Use async generation with retry + rate limiting
            async with DashScopeClient._sem:
                response = await self._retry_async_call(self._async_generate, full_prompt)

            content = response

            # Try to parse JSON response
            try:
                result = json.loads(content)
                if "confidence_score" not in result and "confidence" in result:
                    result["confidence_score"] = result.pop("confidence")
                result.setdefault("sub_components", {})
                result.setdefault("reasoning", "")
                if "engine" in result and "confidence_score" in result:
                    return result
                raise ValueError(f"Missing required fields in response: {result}")
            except json.JSONDecodeError:
                # Fallback: extract JSON from text
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    if "confidence_score" not in result and "confidence" in result:
                        result["confidence_score"] = result.pop("confidence")
                    result.setdefault("sub_components", {})
                    result.setdefault("reasoning", "")
                    if "engine" in result and "confidence_score" in result:
                        return result
                    raise ValueError(f"Missing required fields in response: {result}")
                else:
                    self.logger.error(f"Could not parse JSON from DashScope response: {content}")
                    raise ValueError(f"Could not parse JSON from response: {content}")

        except Exception as e:
            self.logger.error(f"DashScope classification error: {e}")
            raise

    async def _async_generate(self, prompt: str) -> str:
        """Async generation (if supported by dashscope)"""
        # Check if async is available
        # Use asyncio.to_thread to run sync function in thread pool
        return await asyncio.to_thread(self._sync_generate, prompt)

    def _sync_generate(self, prompt: str) -> str:
        """Synchronous generation with DashScope"""
        response = Generation.call(
            model=self.model,
            prompt=prompt,
            temperature=0.1,
            max_tokens=1000,
            top_p=0.1,
        )

        if response.status_code == 200:
            return response.output.text
        else:
            raise Exception(f"DashScope API error: {response.status_code} - {response.message}")

    async def generate(self, prompt: str, max_tokens: int = 1000, **kwargs) -> str:
        """Generate text using Qwen

        Args:
            prompt: Generation prompt
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        try:
            # Use async generation with retry + rate limiting
            async with DashScopeClient._sem:
                return await self._retry_async_call(self._async_generate_full, prompt, max_tokens, **kwargs)

        except Exception as e:
            self.logger.error(f"DashScope generation error: {e}")
            raise

    async def _async_generate_full(self, prompt: str, max_tokens: int, **kwargs) -> str:
        """Async generation for full text"""
        return await asyncio.to_thread(self._sync_generate_full, prompt, max_tokens, **kwargs)

    def _sync_generate_full(self, prompt: str, max_tokens: int, **kwargs) -> str:
        """Synchronous generation for full text"""
        response = Generation.call(
            model=self.model,
            prompt=prompt,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=max_tokens,
            top_p=kwargs.get("top_p", 0.8),
        )

        if response.status_code == 200:
            return response.output.text
        else:
            raise Exception(f"DashScope API error: {response.status_code} - {response.message}")

    async def stream_generate(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Stream generate text using Qwen with live token streaming

        Args:
            prompt: Generation prompt
            **kwargs: Additional parameters

        Yields:
            Generated text tokens as they are produced
        """
        try:
            import asyncio

            # Use async streaming generation
            async for chunk in self._async_stream_generate(prompt, **kwargs):
                yield chunk

        except Exception as e:
            self.logger.error(f"DashScope streaming error: {e}")
            raise

    async def _async_stream_generate(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Async streaming generation for DashScope"""
        import asyncio
        import threading
        from queue import Queue, Empty

        def _sync_stream():
            """Synchronous streaming generator"""
            # Retry with backoff on throttling
            attempts = int(os.getenv("LLM_STREAM_RETRIES", "5"))
            base = float(os.getenv("LLM_BACKOFF_BASE", "1.5"))
            for attempt in range(1, attempts + 1):
                try:
                    response = Generation.call(
                        model=self.model,
                        prompt=prompt,
                        temperature=kwargs.get("temperature", 0.7),
                        max_tokens=kwargs.get("max_tokens", 1000),
                        top_p=kwargs.get("top_p", 0.8),
                        stream=True,  # Enable streaming
                    )
                    for chunk in response:
                        if chunk.status_code == 200:
                            if hasattr(chunk.output, 'text') and chunk.output.text:
                                yield chunk.output.text
                        else:
                            msg = f"DashScope streaming error: {chunk.status_code} - {chunk.message}"
                            raise Exception(msg)
                    return
                except Exception as e:
                    s = str(e)
                    if "Too many requests" in s or "429" in s or "503" in s:
                        delay = min(30.0, (base ** (attempt - 1)) + random.uniform(0, 0.5))
                        time.sleep(delay)
                        continue
                    self.logger.error(f"DashScope sync streaming error: {e}")
                    raise

        # Create a queue for thread communication
        queue = Queue()

        def worker():
            try:
                for chunk in _sync_stream():
                    queue.put(('data', chunk))
                queue.put(('done', None))
            except Exception as e:
                queue.put(('error', e))

        # Start the worker thread
        thread = threading.Thread(target=worker)
        thread.start()

        try:
            while True:
                # Wait for data with timeout
                try:
                    item_type, item_data = queue.get(timeout=1.0)
                except Empty:
                    if thread.is_alive():
                        continue
                    break

                if item_type == 'data':
                    yield item_data
                elif item_type == 'done':
                    break
                elif item_type == 'error':
                    raise item_data

        finally:
            # Ensure thread is cleaned up
            thread.join(timeout=1)

    async def _retry_async_call(self, func, *args, **kwargs):
        """Retry wrapper with exponential backoff and jitter for async calls."""
        attempts = int(os.getenv("LLM_MAX_RETRIES", "5"))
        base = float(os.getenv("LLM_BACKOFF_BASE", "1.5"))
        for attempt in range(1, attempts + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                s = str(e)
                if any(t in s for t in ("Too many requests", "429", "503")):
                    delay = min(30.0, (base ** (attempt - 1)) + random.uniform(0, 0.5))
                    await asyncio.sleep(delay)
                    continue
                raise





def create_llm_client(provider: str, api_key: Optional[str] = None, model: Optional[str] = None) -> LLMClient:
    """Factory function to create LLM client

    Args:
        provider: LLM provider ('anthropic', 'openai', 'dashscope', 'mock')
        api_key: API key for the provider
        model: Model name to use

    Returns:
        LLM client instance
    """
    if provider == "anthropic":
        if not api_key:
            raise ValueError("API key required for Anthropic client")
        return AnthropicClient(api_key, model or "claude-3-sonnet-20240229")

    elif provider == "openai":
        if not api_key:
            raise ValueError("API key required for OpenAI client")
        return OpenAIClient(api_key, model or "gpt-4-turbo-preview")

    elif provider == "dashscope":
        if not api_key:
            raise ValueError("API key required for DashScope client")
        return DashScopeClient(api_key, model or "qwen-max-latest")

    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Available providers: anthropic, openai, dashscope")
