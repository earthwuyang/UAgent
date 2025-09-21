"""DashScope Qwen-VL helper for visual question answering."""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import List

try:
    import dashscope  # type: ignore
    from dashscope import MultiModalConversation  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    dashscope = None  # type: ignore
    MultiModalConversation = None  # type: ignore

LOGGER = logging.getLogger(__name__)


class QwenVisionAnalyzer:
    """Wrapper that sends images to Qwen-VL-Max for interpretation."""

    def __init__(self, api_key: str, model: str = "qwen-vl-max"):
        if dashscope is None or MultiModalConversation is None:
            raise RuntimeError("dashscope multimodal support not available; install dashscope>=1.12.0")
        dashscope.api_key = api_key
        self.model = model

    @property
    def available(self) -> bool:
        return True

    async def describe_image(self, image_bytes: bytes, question: str) -> str:
        """Generate a textual description/answer grounded in the provided image."""

        encoded = base64.b64encode(image_bytes).decode("utf-8")
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"text": question},
                        {"image": f"data:image/png;base64,{encoded}"},
                    ],
                }
            ],
        }

        def _call() -> dict:
            response = MultiModalConversation.call(**payload)
            raw_output = getattr(response, "output", {})
            if getattr(response, "status_code", 200) != 200:
                message = getattr(response, "message", "Unknown error")
                raise RuntimeError(f"Qwen-VL request failed: {message}")
            return raw_output

        output = await asyncio.to_thread(_call)
        return self._extract_text(output)

    async def batch_describe(self, images: List[bytes], question: str) -> List[str]:
        results = []
        for image in images:
            try:
                results.append(await self.describe_image(image, question))
            except Exception as exc:  # pragma: no cover - best effort interactions
                LOGGER.warning("Qwen-VL analysis failed: %s", exc)
                results.append("")
        return results

    @staticmethod
    def _extract_text(output: dict) -> str:
        choices = output.get("choices", []) if isinstance(output, dict) else []
        if not choices:
            return ""
        message = choices[0].get("message", {})
        content = message.get("content", [])
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                parts.append(item["text"])
        return " \n".join(parts).strip()


__all__ = ["QwenVisionAnalyzer"]
