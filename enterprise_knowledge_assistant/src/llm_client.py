"""LLM client with Gemini as primary provider and OpenRouter as fallback.

Both providers expose a streaming interface, so the UI can render answers
token by token. The fallback kicks in automatically when Gemini raises an
exception (network, quota, etc.) which makes the app resilient.
"""

from __future__ import annotations

from typing import Iterable, Iterator, List, Optional

from google import genai
from google.genai import types
from openai import OpenAI


class LLMUnavailableError(RuntimeError):
    """Raised when no provider can serve the request."""


class LLMClient:
    def __init__(
        self,
        gemini_api_key: Optional[str],
        openrouter_api_key: Optional[str],
        provider: str = "gemini",
        gemini_model: str = "gemini-2.5-flash",
        openrouter_model: str = "google/gemini-2.5-flash",
    ) -> None:
        self._provider = provider
        self._gemini_model = gemini_model
        self._openrouter_model = openrouter_model

        self._gemini = genai.Client(api_key=gemini_api_key) if gemini_api_key else None
        self._openrouter = (
            OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_api_key,
            )
            if openrouter_api_key
            else None
        )

    # --- public API ---------------------------------------------------

    def stream_answer(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        order = self._provider_order()
        last_error: Optional[Exception] = None

        for provider in order:
            try:
                if provider == "gemini" and self._gemini is not None:
                    yield from self._stream_gemini(system_prompt, user_prompt)
                    return
                if provider == "openrouter" and self._openrouter is not None:
                    yield from self._stream_openrouter(system_prompt, user_prompt)
                    return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue

        message = "No LLM provider is configured."
        if last_error is not None:
            message = f"All LLM providers failed. Last error: {last_error}"
        raise LLMUnavailableError(message)

    # --- internals ----------------------------------------------------

    def _provider_order(self) -> List[str]:
        preferred = "gemini" if self._provider not in {"gemini", "openrouter"} else self._provider
        secondary = "openrouter" if preferred == "gemini" else "gemini"
        return [preferred, secondary]

    def _stream_gemini(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        assert self._gemini is not None
        stream = self._gemini.models.generate_content_stream(
            model=self._gemini_model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.2,
                max_output_tokens=2048,
            ),
        )
        for chunk in stream:
            text = getattr(chunk, "text", None)
            if text:
                yield text

    def _stream_openrouter(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        assert self._openrouter is not None
        stream = self._openrouter.chat.completions.create(
            model=self._openrouter_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
            temperature=0.2,
            max_tokens=2048,
            extra_headers={
                "HTTP-Referer": "https://enterprise-knowledge-assistant.streamlit.app",
                "X-Title": "Enterprise Knowledge Assistant",
            },
        )
        for part in stream:
            if not part.choices:
                continue
            delta = part.choices[0].delta.content
            if delta:
                yield delta
