from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from jinja2 import Environment, FileSystemLoader, Template

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore

_DOTENV_LOADED = False


@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = os.getenv("AUTOGEN_MODEL", "gpt-4.1-mini")
    # Prefer AUTOGEN_API_KEY, fall back to OPENAI_API_KEY from .env
    api_key: Optional[str] = os.getenv("AUTOGEN_API_KEY") or os.getenv("OPENAI_API_KEY")


class BaseAgent(ABC):
    name: str = "base"

    def __init__(self, template_dir: Path | None = None, llm_config: LLMConfig | None = None) -> None:
        global _DOTENV_LOADED

        # Load .env from course_material_service if available (only once)
        if not _DOTENV_LOADED and load_dotenv is not None:
            project_root = Path(__file__).resolve().parents[1]
            env_path = project_root / "course_material_service" / ".env"
            if env_path.exists():
                load_dotenv(dotenv_path=env_path)
                logger.info("Loaded environment variables from %s", env_path)
            _DOTENV_LOADED = True

        base_dir = template_dir or Path(__file__).resolve().parents[1] / "templates"
        self._template_env = Environment(loader=FileSystemLoader(str(base_dir)))
        self.llm_config = llm_config or LLMConfig()
        self._llm_client: Any | None = None

        if self.llm_config.provider == "openai" and OpenAI is not None and self.llm_config.api_key:
            self._llm_client = OpenAI(api_key=self.llm_config.api_key)

    @abstractmethod
    def generate(self, input_json: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def call_llm(self, prompt: str, **kwargs: Any) -> Dict[str, Any] | str:
        if self.llm_config.provider != "openai":
            raise NotImplementedError(f"LLM provider '{self.llm_config.provider}' is not supported.")

        if OpenAI is None or self._llm_client is None:
            raise RuntimeError(
                "OpenAI client is not available. Install 'openai' and set AUTOGEN_API_KEY."
            )

        logger.info("Calling LLM provider=%s model=%s", self.llm_config.provider, self.llm_config.model)

        completion = self._llm_client.chat.completions.create(
            model=self.llm_config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            **kwargs
        )
        content = completion.choices[0].message.content or ""
        return content

    def load_template(self, name: str) -> Template:
        return self._template_env.get_template(name)

    def validate_json(self, json_data: str | Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(json_data, dict):
            return json_data
        text = (json_data or "").strip()
        # Strip Markdown-style JSON fences if present
        if text.startswith("```"):
            lines = text.splitlines()
            # drop opening ``` / ```json line
            lines = lines[1:]
            # drop trailing ``` if present
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fallback: attempt to extract JSON object from surrounding text
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = text[start : end + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError as exc:
                    logger.debug("Failed to decode extracted LLM JSON snippet: %s", exc)
            logger.debug("Failed to decode LLM JSON. Raw text: %s", text[:500])
            raise ValueError("Invalid JSON returned from LLM") from None
