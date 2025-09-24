"""LLM client for classification and other AI tasks"""

import json
import logging
import os
import asyncio
import random
import re
import time
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from typing import AsyncIterator

from ..utils.json_utils import JsonParseError, safe_json_loads

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

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    litellm = None  # type: ignore
    LITELLM_AVAILABLE = False


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
        if not hasattr(DashScopeClient, "_rate_lock"):
            DashScopeClient._rate_lock = asyncio.Lock()
            DashScopeClient._next_available = 0.0
            DashScopeClient._min_interval = float(
                os.getenv("DASHSCOPE_MIN_INTERVAL", "0.25")
            )

    @classmethod
    async def _acquire_rate_slot(cls) -> None:
        """Enforce minimum delay between outbound DashScope requests."""
        interval = getattr(cls, "_min_interval", 0.25)
        lock: asyncio.Lock = getattr(cls, "_rate_lock")
        async with lock:
            now = time.monotonic()
            wait_for = getattr(cls, "_next_available", 0.0) - now
            if wait_for > 0:
                await asyncio.sleep(wait_for)
                now = time.monotonic()
            setattr(cls, "_next_available", now + interval)

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
            await DashScopeClient._acquire_rate_slot()
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
            await DashScopeClient._acquire_rate_slot()
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

            await DashScopeClient._acquire_rate_slot()

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
                if any(t in s for t in ("Too many requests", "429", "503", "capacity limits", "InternalError.Algo")):
                    delay = min(30.0, (base ** (attempt - 1)) + random.uniform(0, 0.5))
                    await asyncio.sleep(delay)
                    continue
                raise


class LiteLLMClient(LLMClient):
    """LiteLLM-backed client that can proxy multiple LLM providers."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        provider: Optional[str] = None,
        api_base: Optional[str] = None,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not LITELLM_AVAILABLE:
            raise ImportError("litellm package is required for LiteLLMClient")

        self.logger = logging.getLogger(__name__)
        normalized_model = model.strip() if isinstance(model, str) else str(model)
        provider_prefix = (provider or "").strip().lower()
        if provider_prefix and "/" not in normalized_model:
            normalized_model = f"{provider_prefix}/{normalized_model}"
        self.model = normalized_model
        self.api_key = api_key or os.getenv("LITELLM_API_KEY")
        self.api_base = api_base or os.getenv("LITELLM_API_BASE")
        self.extra_options: Dict[str, Any] = {}
        if extra_options:
            self.extra_options.update(extra_options)
        env_extra = os.getenv("LITELLM_EXTRA_OPTIONS")
        if env_extra:
            try:
                parsed = json.loads(env_extra)
                if isinstance(parsed, dict):
                    self.extra_options.update(parsed)
            except json.JSONDecodeError:
                self.logger.warning("Invalid JSON in LITELLM_EXTRA_OPTIONS; ignoring.")

    def _build_request(self, **overrides: Any) -> Dict[str, Any]:
        payload: Dict[str, Any] = dict(self.extra_options)
        payload.update(overrides)
        payload.setdefault("model", self.model)
        if self.api_key:
            payload.setdefault("api_key", self.api_key)
        if self.api_base:
            payload.setdefault("api_base", self.api_base)
        return payload

    async def classify(self, request: str, prompt: str) -> Dict[str, Any]:
        env_tokens = int(os.getenv("LITELLM_CLASSIFICATION_MAX_TOKENS", "8192"))
        classification_tokens = max(1, min(env_tokens, 8192))
        full_prompt = (
            f"{prompt}\n\nUser request: {request}\n\n"
            "Respond with a raw JSON object only. Do not include markdown fences, code blocks, or extra commentary."
        )
        # Try strict JSON mode where provider supports it; fall back to plain text
        try:
            content = await self.generate(
                full_prompt,
                max_tokens=classification_tokens,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
        except Exception:
            content = await self.generate(full_prompt, max_tokens=classification_tokens, temperature=0.1)
        try:
            parsed = safe_json_loads(content)
        except JsonParseError as exc:
            snippet = (content or "").strip().replace("\n", " ")
            if len(snippet) > 300:
                snippet = snippet[:297] + "..."
            self.logger.error(
                "LiteLLM classification JSON decode error: %s | raw=%s",
                exc,
                snippet or "<empty>",
            )
            raise

        if not isinstance(parsed, dict):
            self.logger.error(
                "LiteLLM classification returned non-dict payload of type %s",
                type(parsed).__name__,
            )
            raise JsonParseError("Classification response was not a JSON object")

        result = dict(parsed)
        if "confidence_score" not in result and "confidence" in result:
            result["confidence_score"] = result.pop("confidence")
        result.setdefault("sub_components", {})
        result.setdefault("reasoning", "")
        return result

    async def generate(self, prompt: str, max_tokens: int = 1000, **kwargs) -> str:
        messages = [{"role": "user", "content": prompt}]
        extra: Dict[str, Any] = {}
        if "response_format" in kwargs:
            extra["response_format"] = kwargs["response_format"]
        options = self._build_request(
            messages=messages,
            max_tokens=max_tokens,
            temperature=kwargs.get("temperature", 0.7),
            **extra,
        )

        # Add timeout (default 120 seconds, configurable via env var)
        timeout = float(os.getenv("LITELLM_TIMEOUT", "120"))
        options["timeout"] = timeout

        try:
            response = await asyncio.wait_for(
                litellm.acompletion(**options),
                timeout=timeout + 5  # Give a bit more time for the underlying call
            )
        except asyncio.TimeoutError:
            self.logger.error("LiteLLM generation timeout after %s seconds", timeout)
            raise TimeoutError(f"LLM request timed out after {timeout} seconds")
        except Exception as exc:
            self.logger.error("LiteLLM generation error: %s", exc)
            raise

        if isinstance(response, str):
            return response

        choices = response.get("choices", []) if isinstance(response, dict) else getattr(response, "choices", [])
        if not choices:
            return ""
        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else getattr(first, "message", None)
        content_text = self._extract_message_content(message)
        return self._strip_reasoning_tags(content_text)

    @staticmethod
    def _extract_message_content(message: Any) -> str:
        if message is None:
            return ""

        def _from_content(content: Any) -> str:
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        text_val = item.get("text") or item.get("content")
                        if isinstance(text_val, str):
                            parts.append(text_val)
                return "\n".join(parts)
            return str(content)

        if isinstance(message, dict):
            return _from_content(message.get("content"))

        content_attr = getattr(message, "content", None)
        if content_attr is not None:
            return _from_content(content_attr)

        return str(message)

    @staticmethod
    def _strip_reasoning_tags(text: str) -> str:
        if not isinstance(text, str):
            return ""
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return cleaned.strip()

    async def stream_generate(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        messages = [{"role": "user", "content": prompt}]
        options = self._build_request(
            messages=messages,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 1000),
            stream=True,
        )
        try:
            stream = await litellm.acompletion(**options)
        except Exception as exc:
            self.logger.error("LiteLLM streaming error: %s", exc)
            raise

        async for chunk in stream:
            if not chunk:
                continue
            if isinstance(chunk, str):
                yield chunk
                continue
            choices = chunk.get("choices") if isinstance(chunk, dict) else getattr(chunk, "choices", None)
            if not choices:
                continue
            first = choices[0]
            delta = first.get("delta") if isinstance(first, dict) else getattr(first, "delta", None)
            delta = delta or {}
            text = delta.get("content") if isinstance(delta, dict) else getattr(delta, "content", None)
            if text:
                yield text





DEFAULT_LITELLM_MODELS: Dict[str, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-sonnet-20240229",
    "dashscope": "qwen-max-latest",
    "litellm": "gpt-4o-mini",
}


def create_llm_client(provider: str, api_key: Optional[str] = None, model: Optional[str] = None) -> LLMClient:
    """Factory function to create LLM client

    Args:
        provider: Target provider label (e.g., 'openai', 'anthropic', 'dashscope', 'litellm')
        api_key: API key for the provider
        model: Model name to use; if missing, sensible defaults are applied per provider

    Returns:
        LLM client instance
    """

    normalized_provider = (provider or "litellm").strip().lower()
    if normalized_provider not in DEFAULT_LITELLM_MODELS:
        raise ValueError(
            f"Unknown LLM provider: {provider}. Available providers: "
            "openai, anthropic, dashscope, litellm"
        )

    resolved_model = model or os.getenv("LLM_MODEL")
    if not resolved_model:
        env_key = f"{normalized_provider.upper()}_MODEL"
        resolved_model = os.getenv(env_key)
    if not resolved_model:
        resolved_model = os.getenv("LITELLM_MODEL")
    if not resolved_model:
        resolved_model = DEFAULT_LITELLM_MODELS[normalized_provider]

    resolved_api_key = api_key or os.getenv("LLM_API_KEY")
    if not resolved_api_key:
        resolved_api_key = os.getenv(f"{normalized_provider.upper()}_API_KEY")
    if not resolved_api_key:
        resolved_api_key = os.getenv("LITELLM_API_KEY")

    if not resolved_api_key:
        raise ValueError(
            f"API key required for provider '{normalized_provider}'. "
            "Set LITELLM_API_KEY or the provider-specific key."
        )

    extra_options: Dict[str, Any] = {}
    env_extra = os.getenv("LITELLM_EXTRA_OPTIONS")
    if env_extra:
        try:
            parsed = json.loads(env_extra)
            if isinstance(parsed, dict):
                extra_options.update(parsed)
        except json.JSONDecodeError:
            logging.getLogger(__name__).warning(
                "Invalid JSON in LITELLM_EXTRA_OPTIONS; ignoring.")

    return LiteLLMClient(
        api_key=resolved_api_key,
        model=resolved_model,
        provider=None if normalized_provider == "litellm" else normalized_provider,
        api_base=os.getenv("LITELLM_API_BASE"),
        extra_options=extra_options,
    )
