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
