"""Outbound request shape: prompt_cache_key -> session_id header."""
from __future__ import annotations

import json

import httpx
import pytest

from codex_rosetta.config import Settings
from codex_rosetta.upstream.client import UpstreamClient


def make_client(handler):
    settings = Settings(
        UPSTREAM_BASE_URL="http://upstream.invalid/v1",
        UPSTREAM_API_KEY="sk-test",
        UPSTREAM_PROVIDER="openai",
    )
    client = UpstreamClient(settings)
    # Replace the transport only; the request-building code under test is the
    # same one used against the real upstream.
    client._client = httpx.AsyncClient(
        base_url=settings.UPSTREAM_BASE_URL,
        transport=httpx.MockTransport(handler),
    )
    return client


class TestPromptCacheKeyRelocation:
    @pytest.mark.asyncio
    async def test_non_streaming_moves_key_to_session_id_header(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content.decode())
            seen["headers"] = dict(request.headers)
            return httpx.Response(200, json={"id": "chatcmpl-1", "choices": []})

        client = make_client(handler)
        try:
            await client.chat_completions({
                "model": "deepseek-flash",
                "messages": [{"role": "user", "content": "hi"}],
                "prompt_cache_key": "01a0aad1-thread",
            })
        finally:
            await client.close()

        # strict chat-completions upstreams reject the field outright
        assert "prompt_cache_key" not in seen["body"]
        assert seen["headers"]["session_id"] == "01a0aad1-thread"
        # nothing else was disturbed
        assert seen["body"]["model"] == "deepseek-flash"
        assert seen["body"]["messages"] == [{"role": "user", "content": "hi"}]

    @pytest.mark.asyncio
    async def test_streaming_moves_key_to_session_id_header(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content.decode())
            seen["headers"] = dict(request.headers)
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=b"data: [DONE]\n\n",
            )

        client = make_client(handler)
        try:
            async for _ in client.chat_completions_stream({
                "model": "deepseek-flash",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
                "prompt_cache_key": "01a0aad1-thread",
            }):
                pass
        finally:
            await client.close()

        assert "prompt_cache_key" not in seen["body"]
        assert seen["headers"]["session_id"] == "01a0aad1-thread"

    @pytest.mark.asyncio
    async def test_absent_or_blank_key_sends_no_session_header(self):
        bodies = []

        def handler(request: httpx.Request) -> httpx.Response:
            bodies.append((json.loads(request.content.decode()), dict(request.headers)))
            return httpx.Response(200, json={"id": "chatcmpl-1", "choices": []})

        client = make_client(handler)
        try:
            await client.chat_completions({
                "model": "deepseek-flash",
                "messages": [{"role": "user", "content": "hi"}],
            })
            await client.chat_completions({
                "model": "deepseek-flash",
                "messages": [{"role": "user", "content": "hi"}],
                "prompt_cache_key": "   ",
            })
        finally:
            await client.close()

        for body, headers in bodies:
            assert "session_id" not in headers
            assert "prompt_cache_key" not in body

    @pytest.mark.asyncio
    async def test_caller_supplied_body_is_not_mutated(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"id": "chatcmpl-1", "choices": []})

        client = make_client(handler)
        original = {
            "model": "deepseek-flash",
            "messages": [{"role": "user", "content": "hi"}],
            "prompt_cache_key": "thread-1",
        }
        try:
            await client.chat_completions(original)
        finally:
            await client.close()

        assert original["prompt_cache_key"] == "thread-1"


class TestUpstreamErrorSurfacing:
    """Real failures must explain themselves; forced tool_choice must not kill a turn."""

    @pytest.mark.asyncio
    async def test_error_message_carries_upstream_detail(self):
        detail = "当前模型或上游不支持指定工具的强制选择方式，请改用 tool_choice=auto"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": {"message": detail}})

        client = make_client(handler)
        try:
            with pytest.raises(httpx.HTTPStatusError) as exc:
                await client.chat_completions({
                    "model": "deepseek-flash",
                    "messages": [{"role": "user", "content": "hi"}],
                    "tool_choice": {"type": "function", "function": {"name": "echo"}},
                })
        finally:
            await client.close()

        assert "tool_choice=auto" in str(exc.value)

    @pytest.mark.asyncio
    async def test_forced_tool_choice_retried_as_auto(self):
        seen: list[object] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode())
            seen.append(body.get("tool_choice"))
            if len(seen) == 1:
                return httpx.Response(400, json={"error": {"message": "请改用 tool_choice=auto"}})
            return httpx.Response(200, json={"id": "chatcmpl-1", "choices": []})

        client = make_client(handler)
        try:
            await client.chat_completions({
                "model": "deepseek-flash",
                "messages": [{"role": "user", "content": "hi"}],
                "tool_choice": {"type": "function", "function": {"name": "echo"}},
            })
        finally:
            await client.close()

        assert seen == [{"type": "function", "function": {"name": "echo"}}, "auto"]

    @pytest.mark.asyncio
    async def test_forced_tool_choice_retried_as_auto_streaming(self):
        seen: list[object] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode())
            seen.append(body.get("tool_choice"))
            if len(seen) == 1:
                return httpx.Response(400, json={"error": {"message": "请改用 tool_choice=auto"}})
            return httpx.Response(200, text="data: {}\n\n", headers={"content-type": "text/event-stream"})

        client = make_client(handler)
        try:
            chunks = [c async for c in client.chat_completions_stream({
                "model": "deepseek-flash",
                "messages": [{"role": "user", "content": "hi"}],
                "tool_choice": {"type": "function", "function": {"name": "echo"}},
            })]
        finally:
            await client.close()

        assert seen == [{"type": "function", "function": {"name": "echo"}}, "auto"]
        assert chunks
