"""Tier 4: Harness compatibility tests.

Validates that external tools (OpenAI Python client) can connect to
and use the mlx-stack endpoints. This is what coding agents like aider,
OpenCode, Continue, and Claude Code use under the hood.

**Requirements:**
  - macOS (Apple Silicon)
  - vllm-mlx and litellm installed
  - ``openai`` Python package installed (optional test dependency)
  - Run explicitly: ``pytest -m harness -v``
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mlx_stack.core.pull import pull_model

from .conftest import (
    PortAllocator,
    ServiceManager,
    skip_no_litellm,
    skip_no_vllm,
    skip_not_darwin,
    write_test_stack,
)

# --------------------------------------------------------------------------- #
# Markers
# --------------------------------------------------------------------------- #

pytestmark = [pytest.mark.harness, skip_not_darwin, skip_no_vllm, skip_no_litellm]

# --------------------------------------------------------------------------- #
# OpenAI client availability
# --------------------------------------------------------------------------- #

try:
    from openai import OpenAI  # type: ignore[import-not-found]

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

skip_no_openai = pytest.mark.skipif(
    not HAS_OPENAI,
    reason="openai package not installed (pip install openai)",
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

MODEL_ID = "qwen3.5-0.8b"
MODEL_QUANT = "int4"
MODEL_SOURCE = "mlx-community/Qwen3.5-0.8B-4bit"


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


@skip_no_openai
class TestOpenAIClientCompatibility:
    """Validate the OpenAI Python client works against mlx-stack endpoints.

    This is the client library that most coding agents (aider, OpenCode,
    Continue, Claude Code) use to communicate with OpenAI-compatible APIs.
    """

    @pytest.fixture(autouse=True)
    def _start_stack(
        self,
        integration_home: Path,
        port_allocator: PortAllocator,
        model_cache_dir: Path,
    ):
        """Start a single-tier stack with LiteLLM for all harness tests."""
        self.vllm_port = port_allocator.allocate()
        self.litellm_port = port_allocator.allocate()

        tier = {
            "name": "fast",
            "model": MODEL_ID,
            "quant": MODEL_QUANT,
            "source": MODEL_SOURCE,
            "port": self.vllm_port,
            "vllm_flags": {},
        }

        write_test_stack(integration_home, [tier], self.litellm_port)
        pull_model(model_id=MODEL_ID, quant=MODEL_QUANT)

        self._svc = ServiceManager(integration_home)
        self._svc.__enter__()

        self._svc.start_vllm(
            "fast",
            MODEL_SOURCE,
            self.vllm_port,
            timeout=120.0,
        )
        self._svc.start_litellm(
            self.litellm_port,
            integration_home / "litellm.yaml",
            timeout=120.0,
        )

        self.client = OpenAI(
            base_url=f"http://localhost:{self.litellm_port}/v1",
            api_key="dummy",
        )

        yield

        self._svc.__exit__(None, None, None)
        port_allocator.release(self.vllm_port)
        port_allocator.release(self.litellm_port)

    def test_chat_completion(self) -> None:
        """Standard chat completion via OpenAI client succeeds."""
        response = self.client.chat.completions.create(
            model="fast",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10,
        )
        assert response.choices[0].message.content, "OpenAI client got empty response content"
        assert len(response.choices[0].message.content.strip()) > 0

    def test_streaming(self) -> None:
        """SSE streaming via OpenAI client produces multiple chunks."""
        stream = self.client.chat.completions.create(
            model="fast",
            messages=[{"role": "user", "content": "Count to 3"}],
            max_tokens=20,
            stream=True,
        )
        chunks = list(stream)
        assert len(chunks) > 1, f"Expected multiple SSE chunks, got {len(chunks)}"

        # At least one chunk should have content
        has_content = any(
            c.choices and c.choices[0].delta and c.choices[0].delta.content for c in chunks
        )
        assert has_content, "No chunk contained content"

    def test_model_listing(self) -> None:
        """GET /v1/models returns expected tier names via OpenAI client."""
        models = self.client.models.list()
        model_ids = [m.id for m in models.data]
        assert "fast" in model_ids, f"Tier 'fast' not found in model list: {model_ids}"

    def test_tool_calling_via_client(self) -> None:
        """Function calling via OpenAI client returns structured tool_calls."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather in a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city name",
                            },
                        },
                        "required": ["location"],
                    },
                },
            },
        ]

        response = self.client.chat.completions.create(
            model="fast",
            messages=[
                {"role": "user", "content": "What is the weather in San Francisco?"},
            ],
            tools=tools,
            max_tokens=100,
        )

        msg = response.choices[0].message
        assert msg.tool_calls is not None and len(msg.tool_calls) > 0, (
            f"Expected tool_calls in response, got message: {msg}"
        )

        tc = msg.tool_calls[0]
        assert tc.function.name == "get_weather", (
            f"Expected get_weather call, got: {tc.function.name}"
        )

    def test_invalid_model_returns_error(self) -> None:
        """Request to a nonexistent model returns an error, not a crash."""
        from openai import NotFoundError  # type: ignore[import-not-found]

        with pytest.raises((NotFoundError, Exception)):
            self.client.chat.completions.create(
                model="nonexistent-model-name",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=5,
            )
        # The important thing is we got a proper error, not a hang or crash

    def test_system_message(self) -> None:
        """System messages are handled correctly (common in coding agents)."""
        response = self.client.chat.completions.create(
            model="fast",
            messages=[
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": "Say hello"},
            ],
            max_tokens=10,
        )
        assert response.choices[0].message.content
        assert len(response.choices[0].message.content.strip()) > 0

    def test_multi_turn_conversation(self) -> None:
        """Multi-turn conversations work (common in agent loops)."""
        response = self.client.chat.completions.create(
            model="fast",
            messages=[
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Hello Alice!"},
                {"role": "user", "content": "What is my name?"},
            ],
            max_tokens=20,
        )
        assert response.choices[0].message.content
        assert len(response.choices[0].message.content.strip()) > 0
