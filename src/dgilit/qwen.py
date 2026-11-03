"""Optional in-process Qwen client for WAGS structured tasks."""

from __future__ import annotations

import json
import math
import platform
from time import perf_counter
from typing import Any

from wags_llm.client.base import InvokeJsonResponse
from wags_llm.client.exceptions import (
    LLMEmptyResponseError,
    LLMInvocationError,
    LLMJsonDecodeError,
    LLMResponseFormatError,
)


class Qwen3JsonClient:
    """Generate JSON with 4-bit Qwen3 on Apple Silicon Metal, without thinking.

    The schema is supplied as instructions, not constrained decoding. WAGS's
    StructuredTaskRunner validates the parsed response against its result model.
    Model weights are loaded once and reused for subsequent invocations.
    """

    def __init__(
        self,
        model_id: str = "mlx-community/Qwen3-14B-4bit",
        max_tokens: int = 1000,
        temperature: float = 0.0,
    ) -> None:
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("Qwen MLX requires native arm64 Python on Apple Silicon macOS")
        try:
            import mlx.core as mx
            from mlx_lm import load, stream_generate
            from mlx_lm.sample_utils import make_sampler
        except ImportError as exc:
            raise ImportError("Install MLX dependencies with pip install 'dgilit[qwen]'") from exc
        if not mx.metal.is_available():
            raise RuntimeError("Qwen MLX requires Metal acceleration; CPU fallback is disabled")
        mx.set_default_device(mx.gpu)
        self.model_id = model_id
        self.backend = "mlx"
        self.device = "metal"
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._mx = mx
        self._stream_generate = stream_generate
        self._make_sampler = make_sampler
        self.last_generation_stats: dict[str, Any] = {}
        started = perf_counter()
        self._model, self._tokenizer, config = load(model_id, return_config=True)
        self.load_seconds = perf_counter() - started
        quantization = config.get("quantization", config.get("quantization_config", {}))
        self.quantization_bits = quantization.get("bits")
        if self.quantization_bits != 4:
            raise ValueError("Qwen MLX requires a model with 4-bit quantized weights")
        self._context_window = config["max_position_embeddings"]

    def prompt_tokens(self, system_prompt, user_prompt, json_schema=None):
        """Count the exact rendered input, including JSON/schema instructions."""
        return len(self._prompt_tokens(system_prompt, user_prompt, json_schema))

    def _prompt_tokens(self, system_prompt, user_prompt, json_schema=None):
        instructions = "\nReturn only a JSON object, without Markdown or explanation."
        if json_schema is not None:
            instructions += "\nFollow this JSON schema:\n" + json.dumps(json_schema)
        messages = [
            {"role": "system", "content": system_prompt + instructions},
            {"role": "user", "content": user_prompt},
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        return self._tokenizer.encode(text, add_special_tokens=False)

    def context_windows(self, prompt, context, drug, gene, schema, overlap=256):
        """Return character bounds whose complete prompts fit the model window."""
        from .text_windows import text_windows
        def count(text):
            payload = prompt.build_payload(text, drug, gene)
            return self.prompt_tokens(prompt.build_system_prompt(), prompt.build_user_prompt(payload), schema)
        if count(context) + self.max_tokens <= self._context_window:
            return [(0, len(context))]
        budget = self._context_window - self.max_tokens - count("") - 64
        while budget > overlap:
            windows = list(text_windows(self._tokenizer, context, budget, overlap))
            if all(count(context[a:b]) + self.max_tokens <= self._context_window for a, b in windows):
                return windows
            budget -= max(64, budget // 10)
        raise ValueError("Prompt overhead leaves no room for an overlapping context window")

    def invoke_json(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any] | None = None,
    ) -> InvokeJsonResponse:
        self.last_generation_stats = {}
        try:
            tokens = self._prompt_tokens(system_prompt, user_prompt, json_schema)
            if len(tokens) + self.max_tokens > self._context_window:
                raise ValueError("Prompt plus generation budget exceeds the model context window")
            started = perf_counter()
            parts = []
            final = None
            for response in self._stream_generate(
                self._model, self._tokenizer, prompt=tokens,
                max_tokens=self.max_tokens,
                sampler=self._make_sampler(temp=self.temperature),
            ):
                parts.append(response.text)
                final = response
            self._mx.synchronize()
            elapsed = perf_counter() - started
            if final is not None:
                self.last_generation_stats = {
                    "prompt_tokens": final.prompt_tokens,
                    "generation_tokens": final.generation_tokens,
                    "elapsed_seconds": elapsed,
                    "generation_tokens_per_second": final.generation_tps,
                    "end_to_end_tokens_per_second": final.generation_tokens / elapsed,
                }
            raw_text = "".join(parts)
        except Exception as exc:
            raise LLMInvocationError(f"Qwen generation failed: {exc}") from exc

        if not raw_text.strip():
            raise LLMEmptyResponseError("Qwen returned empty output")
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise LLMJsonDecodeError(f"Qwen returned invalid JSON: {raw_text!r}") from exc
        if not isinstance(parsed, dict):
            raise LLMResponseFormatError("Qwen response must be a JSON object")
        return InvokeJsonResponse(parsed_json=parsed, raw_text=raw_text)
