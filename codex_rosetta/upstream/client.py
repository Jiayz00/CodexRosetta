from __future__ import annotations

import time
from typing import Any, AsyncIterator

import httpx

from codex_rosetta.config import Settings, get_settings
from codex_rosetta.upstream.provider_adapters import ProviderAdapter, get_provider_adapter
from codex_rosetta.utils.logging import get_logger

logger = get_logger("upstream")


def _raise_for_upstream_error(response: httpx.Response) -> None:
    """Raise an HTTPStatusError that carries the upstream's own error text.

    httpx's default message ("Client error '400 Bad Request' for url ...") hides
    the reason the upstream gave, which made real failures impossible to
    diagnose from the client. The exception type is unchanged so existing
    handlers keep working.
    """
    if response.status_code < 400:
        return
    detail = " ".join((response.text or "").split())[:600]
    message = (
        f"Client error '{response.status_code} {response.reason_phrase}' "
        f"for url '{response.request.url}'"
    )
    if detail:
        message = f"{message}: {detail}"
    raise httpx.HTTPStatusError(message, request=response.request, response=response)


class UpstreamClient:
    """Async HTTP client for forwarding requests to the upstream Chat Completions API."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._adapter: ProviderAdapter = get_provider_adapter(settings.UPSTREAM_PROVIDER)
        self._client = httpx.AsyncClient(
            base_url=settings.UPSTREAM_BASE_URL,
            timeout=httpx.Timeout(
                connect=settings.UPSTREAM_TIMEOUT_CONNECT,
                read=settings.UPSTREAM_TIMEOUT_READ,
                write=30.0,
                pool=30.0,
            ),
        )
        self._log = logger.bind(
            base_url=settings.UPSTREAM_BASE_URL,
            provider=settings.UPSTREAM_PROVIDER,
        )
        self._log_upstream_requests = settings.LOG_UPSTREAM_REQUESTS
        self._log_upstream_responses = settings.LOG_UPSTREAM_RESPONSES

    @property
    def adapter(self) -> ProviderAdapter:
        return self._adapter

    def _get_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        headers.update(self._adapter.get_auth_headers(self._settings.UPSTREAM_API_KEY))
        return headers

    @staticmethod
    def _finalize_request(
        adapted: dict[str, Any], headers: dict[str, str]
    ) -> dict[str, Any]:
        """Move `prompt_cache_key` out of the body and into a `session_id` header.

        The Responses API field is a cache-routing hint for chat-completions
        gateways (sub2api derives its sticky session seed from it), but many
        strict chat-completions upstreams reject the field outright with
        ``400 未知请求字段：prompt_cache_key``. Moving the value into the
        ``session_id`` header keeps the same sticky/cache identity for the
        gateway while the wire body stays within the chat-completions schema.
        """
        if "prompt_cache_key" not in adapted:
            return adapted

        body = dict(adapted)
        cache_key = str(body.pop("prompt_cache_key") or "").strip()
        if cache_key:
            headers.setdefault("session_id", cache_key)
        return body

    async def chat_completions(self, request_body: dict[str, Any]) -> dict[str, Any]:
        """Send a non-streaming request to upstream /v1/chat/completions."""
        headers = self._get_headers()
        adapted = self._finalize_request(
            self._adapter.adapt_request(request_body), headers
        )

        self._log.info(
            "upstream_request_sent",
            model=adapted.get("model"),
            message_count=len(adapted.get("messages", [])),
            stream=adapted.get("stream", False),
        )

        if self._log_upstream_requests:
            self._log.debug("upstream_request_body", body=adapted)

        start = time.monotonic()
        response = await self._client.post(
            "/chat/completions",
            json=adapted,
            headers=headers,
        )
        duration_ms = round((time.monotonic() - start) * 1000)

        self._log.info(
            "upstream_response_received",
            status_code=response.status_code,
            duration_ms=duration_ms,
        )

        response = await self._retry_without_forced_tool_choice(response, adapted, headers)
        if response.status_code >= 400:
            self._log.error(
                "upstream_error_response",
                status_code=response.status_code,
                error_body=response.text[:2000],
            )
        _raise_for_upstream_error(response)

        data = response.json()
        result = self._adapter.adapt_response(data)

        if self._log_upstream_responses:
            self._log.debug("upstream_response_body", body=result)

        return result

    async def chat_completions_stream(
        self, request_body: dict[str, Any]
    ) -> AsyncIterator[bytes]:
        """Stream from upstream /v1/chat/completions."""
        headers = self._get_headers()
        adapted = self._finalize_request(
            self._adapter.adapt_request(request_body), headers
        )

        self._log.info(
            "upstream_stream_started",
            model=adapted.get("model"),
            message_count=len(adapted.get("messages", [])),
        )

        if self._log_upstream_requests:
            self._log.debug("upstream_request_body", body=adapted)

        async for chunk in self._stream_once(adapted, headers):
            yield chunk

    async def _retry_without_forced_tool_choice(
        self, response: httpx.Response, body: dict[str, Any], headers: dict[str, str]
    ) -> httpx.Response:
        """Retry once with tool_choice=auto when the upstream rejects forcing.

        Strict upstreams answer 400 with "use tool_choice=auto" instead of
        degrading on their own; downgrading keeps the turn alive instead of
        killing the whole conversation.
        """
        if not self._should_downgrade_tool_choice(body, response):
            return response
        self._log.warning(
            "tool_choice_downgraded",
            upstream_status=response.status_code,
            tool_choice=body.get("tool_choice"),
        )
        return await self._client.post(
            "/chat/completions", json={**body, "tool_choice": "auto"}, headers=headers
        )

    @staticmethod
    def _should_downgrade_tool_choice(body: dict[str, Any], response: httpx.Response) -> bool:
        if response.status_code != 400:
            return False
        choice = body.get("tool_choice")
        if not choice or choice == "auto":
            return False
        detail = (response.text or "").lower()
        return "tool_choice" in detail or "强制选择" in detail

    async def _stream_once(
        self, adapted: dict[str, Any], headers: dict[str, str]
    ) -> AsyncIterator[bytes]:
        """Stream one upstream call, downgrading a rejected forced tool_choice once."""
        attempt = 0
        while True:
            async with self._client.stream(
                "POST",
                "/chat/completions",
                json=adapted,
                headers=headers,
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    self._log.error(
                        "upstream_error_response",
                        status_code=response.status_code,
                        error_body=response.text[:2000],
                    )
                    if attempt == 0 and self._should_downgrade_tool_choice(adapted, response):
                        self._log.warning(
                            "tool_choice_downgraded",
                            upstream_status=response.status_code,
                            tool_choice=adapted.get("tool_choice"),
                        )
                        adapted = {**adapted, "tool_choice": "auto"}
                        attempt += 1
                        continue
                    _raise_for_upstream_error(response)

                chunk_count = 0
                async for chunk in response.aiter_bytes():
                    chunk_count += 1
                    yield chunk

                self._log.info("upstream_stream_ended", chunk_count=chunk_count)
                return

    async def close(self) -> None:
        await self._client.aclose()
