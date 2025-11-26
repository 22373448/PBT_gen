from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Protocol

from .config import LLMConfig


class LLMClient(Protocol):
    async def complete(self, prompt: str) -> str:  # pragma: no cover - interface
        ...


@dataclass
class OpenAILLMClient:
    """
    Thin wrapper around OpenAI's Chat Completions API.
    This is a placeholder; you should fill in API key handling etc.
    """

    config: LLMConfig

    async def complete(self, prompt: str) -> str:
        # NOTE: This is a synchronous placeholder; replace with real OpenAI call.
        # Keep it simple so project stays framework-level.
        try:
            import openai  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover - runtime guard
            raise RuntimeError(
                "openai package is not installed. Install it or implement your own LLMClient."
            ) from exc

        client = openai.OpenAI()  # assumes env var OPENAI_API_KEY
        completion = client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        return completion.choices[0].message.content or ""


