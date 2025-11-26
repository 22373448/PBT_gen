from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Protocol

from .config import LLMConfig


# 设置一个专门的 logger，将 LLM 请求与响应写入 llm_log.log
logger = logging.getLogger("pbt_gen.llm")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler("llm_log.log", encoding="utf-8")
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


class LLMClient(Protocol):
    async def complete(self, prompt: str) -> str:  # pragma: no cover - interface
        ...


@dataclass
class OpenAILLMClient(LLMClient):
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

        # 记录输入
        logger.info("LLM REQUEST model=%s\nPROMPT:\n%s", self.config.model, prompt)

        # 支持自定义 api_base（如代理或自建网关），默认为 None 使用官方默认。
        client = openai.OpenAI(base_url=self.config.api_base) if getattr(
            self.config, "api_base", None
        ) else openai.OpenAI()  # assumes env var OPENAI_API_KEY
        completion = client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        content = completion.choices[0].message.content or ""

        # 记录输出
        logger.info("LLM RESPONSE model=%s\nRESPONSE:\n%s", self.config.model, content)

        return content


