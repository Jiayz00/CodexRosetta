"""Regression tests for upstream error surfacing and forced tool_choice fallback."""
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
    client._client = httpx.AsyncClient(
        base_url=settings.UPSTREAM_BASE_URL,
        transport=httpx.MockTransport(handler),
    )
    return client


class TestUpstreamErrorSurfacing:
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
        seen = []

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
        seen = []

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
