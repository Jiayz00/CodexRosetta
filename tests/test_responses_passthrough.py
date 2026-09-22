"""UPSTREAM_API_MODE=responses: passthrough hop + native Tavily search loop."""
from __future__ import annotations

import copy
import json
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

from codex_rosetta.api import router as router_module
from codex_rosetta.api.router import (
    _responses_search_rounds,
    _stream_responses_search,
    _stream_responses_verbatim,
)
from codex_rosetta.audit.logger import NoOpAuditLogger
from codex_rosetta.config import Settings, get_settings
from codex_rosetta.converters.responses_mux import (
    INTERNAL_SEARCH_NAME,
    ResponsesStreamMux,
    normalize_responses_request,
    strip_search_function_tool,
)
from codex_rosetta.main import app
from codex_rosetta.search.base import (
    ERROR_QUOTA_EXCEEDED,
    SearchProvider,
    SearchResponse,
    SearchResult,
)
from codex_rosetta.upstream.client import UpstreamClient
from codex_rosetta.utils.logging import get_logger
from codex_rosetta.utils.sse import parse_sse_lines

LOG = get_logger("test_responses_passthrough")


# --- fakes --------------------------------------------------------------


class FakeResponsesUpstream:
    """Scripted Responses upstream: one script entry per call."""

    def __init__(self, responses=None, streams=None) -> None:
        self._responses = list(responses or [])
        self._streams = list(streams or [])
        self.requests: list[dict] = []

    def _record(self, body: dict) -> int:
        self.requests.append(copy.deepcopy(body))
        return len(self.requests) - 1

    async def responses(self, body: dict) -> dict:
        index = self._record(body)
        return copy.deepcopy(self._responses[min(index, len(self._responses) - 1)])

    async def responses_stream(self, body: dict):
        index = self._record(body)
        for chunk in self._streams[min(index, len(self._streams) - 1)]:
            yield chunk


class StubSearchProvider(SearchProvider):
    def __init__(self, responses: list[SearchResponse]) -> None:
        self._responses = responses
        self.calls: list[str] = []

    async def search(self, query: str, max_results: int = 5) -> SearchResponse:
        self.calls.append(query)
        return self._responses[min(len(self.calls) - 1, len(self._responses) - 1)]


def ok_response(count: int = 2) -> SearchResponse:
    return SearchResponse(
        results=[
            SearchResult(title=f"S{i}", url=f"https://example.com/{i}", snippet="s")
            for i in range(1, count + 1)
        ],
        query="AI news",
    )


def failed_response(kind: str = ERROR_QUOTA_EXCEEDED) -> SearchResponse:
    return SearchResponse(error=kind, error_detail="quota exhausted")


# --- payload builders ---------------------------------------------------


def sse(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def search_item(call_id: str, query: str, item_id: str = "fc_search") -> dict:
    return {
        "type": "function_call",
        "id": item_id,
        "call_id": call_id,
        "name": INTERNAL_SEARCH_NAME,
        "arguments": json.dumps({"query": query}),
    }


def search_response(call_id: str = "call_1", query: str = "AI news") -> dict:
    return {
        "id": "resp_up_1",
        "object": "response",
        "status": "completed",
        "model": "deepseek-flash",
        "output": [
            {"type": "reasoning", "id": "rs_1", "summary": []},
            search_item(call_id, query),
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def answer_response(text: str = "Final answer") -> dict:
    return {
        "id": "resp_up_2",
        "object": "response",
        "status": "completed",
        "model": "deepseek-flash",
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 44, "output_tokens": 9},
    }


def search_round_stream(call_id: str = "call_1", query: str = "AI news") -> str:
    item = search_item(call_id, query)
    return "".join([
        sse("response.created", {"type": "response.created", "response": {"id": "resp_up_1"}}),
        sse("response.in_progress", {"type": "response.in_progress", "response": {"id": "resp_up_1"}}),
        sse("response.output_item.added", {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {**item, "arguments": ""},
        }),
        sse("response.function_call_arguments.delta", {
            "type": "response.function_call_arguments.delta",
            "item_id": item["id"],
            "output_index": 0,
            "delta": item["arguments"],
        }),
        sse("response.function_call_arguments.done", {
            "type": "response.function_call_arguments.done",
            "item_id": item["id"],
            "output_index": 0,
            "arguments": item["arguments"],
        }),
        sse("response.output_item.done", {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": item,
        }),
        sse("response.completed", {
            "type": "response.completed",
            "response": {
                "id": "resp_up_1",
                "status": "completed",
                "output": [item],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        }),
    ])


def answer_round_stream(text: str = "Final answer") -> str:
    item = {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }
    return "".join([
        sse("response.created", {"type": "response.created", "response": {"id": "resp_up_2"}}),
        sse("response.output_item.added", {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {**item, "content": []},
        }),
        sse("response.output_text.delta", {
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": text,
        }),
        sse("response.output_text.done", {
            "type": "response.output_text.done",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "text": text,
        }),
        sse("response.output_item.done", {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": item,
        }),
        sse("response.completed", {
            "type": "response.completed",
            "response": {
                "id": "resp_up_2",
                "status": "completed",
                "output": [item],
                "usage": {"input_tokens": 44, "output_tokens": 9},
            },
        }),
    ])


def collect(events: list[tuple[str, dict]]) -> list[dict]:
    """Parse formatted SSE strings back into event dicts."""
    parsed: list[dict] = []
    for raw in events:
        for _event_type, data in parse_sse_lines(raw):
            if data is not None:
                parsed.append(data)
    return parsed


async def run_stream(generator) -> list[dict]:
    out: list[str] = []
    async for chunk in generator:
        out.append(chunk)
    return collect(out)


BODY = {
    "model": "deepseek-flash",
    "instructions": "You are Codex.",
    "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
    "tools": [{"type": "web_search"}],
    "stream": True,
}


# --- request normalization ---------------------------------------------


def test_normalize_replaces_builtin_search_with_internal_function():
    normalized, wants_search = normalize_responses_request(BODY, search_enabled=True)

    assert wants_search is True
    assert [tool["type"] for tool in normalized["tools"]] == ["function"]
    assert normalized["tools"][0]["name"] == INTERNAL_SEARCH_NAME
    assert normalized["input"] == BODY["input"]


def test_normalize_drops_search_tool_when_disabled():
    normalized, wants_search = normalize_responses_request(BODY, search_enabled=False)

    assert wants_search is True
    assert "tools" not in normalized


def test_normalize_maps_search_tool_choice():
    body = {**BODY, "tool_choice": {"type": "web_search_preview"}}
    normalized, _ = normalize_responses_request(body, search_enabled=True)

    assert normalized["tool_choice"] == {"type": "function", "name": INTERNAL_SEARCH_NAME}


def test_normalize_restores_replayed_search_history():
    body = {
        **BODY,
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            {
                "type": "web_search_call",
                "id": "ws_prev",
                "status": "completed",
                "action": {
                    "type": "search",
                    "queries": ["AI news"],
                    "sources": [{"type": "url", "url": "https://example.com/1"}],
                },
            },
        ],
    }
    normalized, _ = normalize_responses_request(body, search_enabled=True)

    kinds = [item["type"] for item in normalized["input"]]
    assert kinds == ["message", "function_call", "function_call_output"]
    call, output = normalized["input"][1], normalized["input"][2]
    assert call["name"] == INTERNAL_SEARCH_NAME
    assert call["call_id"] == output["call_id"] == "ws_prev"
    assert "https://example.com/1" in output["output"]


def test_strip_search_function_tool_forces_answer():
    normalized, _ = normalize_responses_request(BODY, search_enabled=True)
    forced = strip_search_function_tool(normalized, "STOP SEARCHING")

    assert "tools" not in forced
    assert forced["instructions"].endswith("STOP SEARCHING")


def normalized_body() -> dict:
    """Body as the router hands it to the search loop."""
    body, _ = normalize_responses_request(copy.deepcopy(BODY), search_enabled=True)
    return body


# --- stream mux ---------------------------------------------------------


def mux_events(raw_rounds: list[str]) -> list[dict]:
    mux = ResponsesStreamMux()
    events: list[dict] = []
    for round_index, raw in enumerate(raw_rounds):
        for event_type, data in parse_sse_lines(raw):
            if data is None:
                continue
            for _etype, event in mux.process_event(event_type, data, round_index):
                events.append(event)
        for _etype, event in mux.complete_search_calls({}, {}):
            events.append(event)
        mux.start_next_round()
    _etype, completed = mux.build_completed_event()
    events.append(completed)
    return events


def test_mux_keeps_single_round_shape():
    events = mux_events([answer_round_stream()])

    assert [e["type"] for e in events] == [
        "response.created",
        "response.output_item.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.output_item.done",
        "response.completed",
    ]
    sequences = [e["sequence_number"] for e in events]
    assert sequences == sorted(sequences)
    assert events[-1]["response"]["output"][0]["content"][0]["text"] == "Final answer"


def test_mux_merges_rounds_with_unique_output_indexes():
    events = mux_events([search_round_stream(), answer_round_stream()])

    created = [e for e in events if e["type"] == "response.created"]
    assert len(created) == 1
    completed = [e for e in events if e["type"] == "response.completed"]
    assert len(completed) == 1
    assert [item["type"] for item in completed[0]["response"]["output"]] == [
        "web_search_call",
        "message",
    ]
    sequences = [e["sequence_number"] for e in events]
    assert sequences == sorted(sequences)
    indexes = {
        e["output_index"] for e in events if e["type"] == "response.output_item.added"
    }
    assert indexes == {0, 1}


def test_mux_never_leaks_the_internal_search_name():
    events = mux_events([search_round_stream()])

    assert INTERNAL_SEARCH_NAME not in json.dumps(events)


def test_mux_reports_search_sources_on_completion():
    mux = ResponsesStreamMux()
    for event_type, data in parse_sse_lines(search_round_stream()):
        if data is not None:
            mux.process_event(event_type, data, 0)

    sources = [{"type": "url", "url": "https://example.com/1"}]
    events = [
        event
        for _etype, event in mux.complete_search_calls(
            {"call_1": "result text"}, {"call_1": sources}
        )
    ]
    _etype, completed = mux.build_completed_event()
    events.append(completed)

    assert [e["type"] for e in events] == [
        "response.web_search_call.completed",
        "response.output_item.done",
        "response.completed",
    ]
    item = completed["response"]["output"][0]
    assert item["type"] == "web_search_call"
    assert item["action"]["sources"] == sources
    assert item["action"]["queries"] == ["AI news"]


def test_mux_marks_incomplete_responses():
    raw = "".join([
        sse("response.created", {"type": "response.created", "response": {"id": "resp_x"}}),
        sse("response.incomplete", {
            "type": "response.incomplete",
            "response": {"id": "resp_x", "status": "incomplete"},
        }),
    ])
    mux = ResponsesStreamMux()
    for event_type, data in parse_sse_lines(raw):
        if data is not None:
            mux.process_event(event_type, data, 0)

    result = mux.finalize_round()
    assert result.terminal_event is not None
    assert result.terminal_event[0] == "response.incomplete"
    assert mux.build_completed_event()[0] == "response.incomplete"


# --- upstream client ----------------------------------------------------


def make_client(handler, **overrides) -> UpstreamClient:
    settings = Settings(
        UPSTREAM_BASE_URL="http://upstream.invalid/v1",
        UPSTREAM_API_KEY="sk-test",
        **overrides,
    )
    client = UpstreamClient(settings)
    client._client = httpx.AsyncClient(
        base_url=settings.UPSTREAM_BASE_URL,
        transport=httpx.MockTransport(handler),
    )
    return client


@pytest.mark.asyncio
async def test_responses_hop_keeps_cache_key_in_body():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content.decode())
        seen["headers"] = dict(request.headers)
        return httpx.Response(200, json={"id": "resp_1", "output": []})

    client = make_client(handler, UPSTREAM_API_MODE="responses")
    try:
        await client.responses({
            "model": "deepseek-flash",
            "input": "hi",
            "prompt_cache_key": "thread-1",
            "reasoning": {"effort": "medium"},
        })
    finally:
        await client.close()

    assert seen["url"].endswith("/v1/responses")
    assert seen["body"]["prompt_cache_key"] == "thread-1"
    assert seen["body"]["reasoning"] == {"effort": "medium"}
    assert seen["headers"]["session_id"] == "thread-1"
    assert seen["headers"]["x-opencode-session"] == "thread-1"


@pytest.mark.asyncio
async def test_responses_hop_drops_reasoning_when_configured():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={"id": "resp_1", "output": []})

    client = make_client(
        handler,
        UPSTREAM_API_MODE="responses",
        UPSTREAM_RESPONSES_DROP_REASONING_EFFORT=True,
    )
    try:
        await client.responses({
            "model": "deepseek-flash",
            "input": "hi",
            "reasoning": {"effort": "high", "summary": "auto"},
        })
    finally:
        await client.close()

    assert "reasoning" not in seen["body"]


@pytest.mark.asyncio
async def test_chat_mode_still_relocates_cache_key():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={"choices": []})

    client = make_client(handler)
    try:
        await client.chat_completions({
            "model": "deepseek-flash",
            "messages": [],
            "prompt_cache_key": "thread-1",
        })
    finally:
        await client.close()

    assert seen["url"].endswith("/v1/chat/completions")
    assert "prompt_cache_key" not in seen["body"]


# --- search loop --------------------------------------------------------


@pytest.mark.asyncio
async def test_non_streaming_loop_replays_search_output(monkeypatch):
    monkeypatch.setattr(router_module, "get_settings", lambda: Settings(
        UPSTREAM_API_MODE="responses", WEB_SEARCH_MAX_ROUNDS=3
    ))
    upstream = FakeResponsesUpstream([search_response(), answer_response()])
    provider = StubSearchProvider([ok_response()])

    result = await _responses_search_rounds(
        upstream, normalized_body(), LOG, NoOpAuditLogger(), provider
    )

    assert provider.calls == ["AI news"]
    assert [item["type"] for item in result["output"]] == [
        "reasoning",
        "web_search_call",
        "message",
    ]

    second = upstream.requests[1]
    kinds = [item["type"] for item in second["input"]]
    assert kinds[-2:] == ["function_call", "function_call_output"]
    assert second["input"][-2]["call_id"] == second["input"][-1]["call_id"] == "call_1"
    assert "S1" in second["input"][-1]["output"]
    assert INTERNAL_SEARCH_NAME not in json.dumps(result)


@pytest.mark.asyncio
async def test_non_streaming_loop_stops_after_two_failures(monkeypatch):
    monkeypatch.setattr(router_module, "get_settings", lambda: Settings(
        UPSTREAM_API_MODE="responses", WEB_SEARCH_MAX_ROUNDS=10
    ))
    upstream = FakeResponsesUpstream([
        search_response("call_1"),
        search_response("call_2"),
        search_response("call_3"),
        answer_response(),
    ])
    provider = StubSearchProvider([failed_response()])

    result = await _responses_search_rounds(
        upstream, normalized_body(), LOG, NoOpAuditLogger(), provider
    )

    # two failing searches, then the tool is stripped for the forced answer
    assert len(upstream.requests) == 4
    forced_request = upstream.requests[3]
    assert "tools" not in forced_request
    assert "直接" in forced_request["instructions"]
    assert result["output"][-1]["type"] == "message"
    assert all(
        item["action"]["sources"] == []
        for item in result["output"]
        if item["type"] == "web_search_call"
    )


@pytest.mark.asyncio
async def test_non_streaming_loop_returns_payload_without_search(monkeypatch):
    monkeypatch.setattr(router_module, "get_settings", lambda: Settings(
        UPSTREAM_API_MODE="responses"
    ))
    upstream = FakeResponsesUpstream([answer_response("plain")])

    result = await _responses_search_rounds(
        upstream, normalized_body(), LOG, NoOpAuditLogger(), StubSearchProvider([ok_response()])
    )

    assert len(upstream.requests) == 1
    assert result["output"][0]["content"][0]["text"] == "plain"


@pytest.mark.asyncio
async def test_streaming_loop_emits_lifecycle_and_single_created(monkeypatch):
    monkeypatch.setattr(router_module, "get_settings", lambda: Settings(
        UPSTREAM_API_MODE="responses", WEB_SEARCH_MAX_ROUNDS=3
    ))
    upstream = FakeResponsesUpstream(streams=[
        [search_round_stream()],
        [answer_round_stream()],
    ])
    provider = StubSearchProvider([ok_response()])

    events = await run_stream(_stream_responses_search(
        upstream, normalized_body(), LOG, NoOpAuditLogger(), provider
    ))

    types = [e["type"] for e in events]
    assert types.count("response.created") == 1
    assert types.count("response.completed") == 1
    assert "response.web_search_call.in_progress" in types
    assert "response.web_search_call.searching" in types
    assert "response.web_search_call.completed" in types
    assert types[-1] == "response.completed"

    sequences = [e["sequence_number"] for e in events]
    assert sequences == sorted(sequences)

    completed = events[-1]
    assert [item["type"] for item in completed["response"]["output"]] == [
        "web_search_call",
        "message",
    ]
    sources = completed["response"]["output"][0]["action"]["sources"]
    assert sources == [
        {"type": "url", "url": "https://example.com/1"},
        {"type": "url", "url": "https://example.com/2"},
    ]
    assert INTERNAL_SEARCH_NAME not in json.dumps(events)


@pytest.mark.asyncio
async def test_streaming_loop_mixed_round_forwards_client_tool(monkeypatch):
    monkeypatch.setattr(router_module, "get_settings", lambda: Settings(
        UPSTREAM_API_MODE="responses", WEB_SEARCH_MAX_ROUNDS=3
    ))
    shell = {
        "type": "function_call",
        "id": "fc_shell",
        "call_id": "call_shell",
        "name": "shell",
        "arguments": '{"command":"ls"}',
    }
    search = search_item("call_1", "AI news")
    raw = "".join([
        sse("response.created", {"type": "response.created", "response": {"id": "resp_up_1"}}),
        sse("response.output_item.added", {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {**search, "arguments": ""},
        }),
        sse("response.function_call_arguments.done", {
            "type": "response.function_call_arguments.done",
            "item_id": search["id"],
            "output_index": 0,
            "arguments": search["arguments"],
        }),
        sse("response.output_item.added", {
            "type": "response.output_item.added", "output_index": 1, "item": shell,
        }),
        sse("response.output_item.done", {
            "type": "response.output_item.done", "output_index": 1, "item": shell,
        }),
        sse("response.output_item.done", {
            "type": "response.output_item.done", "output_index": 0, "item": search,
        }),
        sse("response.completed", {
            "type": "response.completed",
            "response": {
                "id": "resp_up_1",
                "status": "completed",
                "output": [search, shell],
            },
        }),
    ])
    upstream = FakeResponsesUpstream(streams=[[raw]])

    events = await run_stream(_stream_responses_search(
        upstream, normalized_body(), LOG, NoOpAuditLogger(), StubSearchProvider([ok_response()])
    ))

    assert len(upstream.requests) == 1
    output = events[-1]["response"]["output"]
    assert [item["type"] for item in output] == ["web_search_call", "function_call"]
    assert output[0]["action"]["sources"] == []
    assert output[1]["name"] == "shell"
    assert INTERNAL_SEARCH_NAME not in json.dumps(events)


# --- wiring -------------------------------------------------------------


@pytest.fixture
def responses_mode(monkeypatch):
    monkeypatch.setattr(router_module, "get_settings", lambda: Settings(
        UPSTREAM_API_MODE="responses", WEB_SEARCH_ENABLED=False
    ))


def test_endpoint_forwards_without_search(responses_mode):
    upstream = FakeResponsesUpstream([answer_response("passthrough")])

    with patch("codex_rosetta.api.router.get_upstream_client", return_value=upstream), \
         TestClient(app) as client:
        response = client.post("/v1/responses", json={
            "model": "deepseek-flash",
            "input": "hi",
        })

    assert response.status_code == 200
    assert response.json()["output"][0]["content"][0]["text"] == "passthrough"
    assert len(upstream.requests) == 1


def test_endpoint_streams_upstream_bytes(responses_mode):
    upstream = FakeResponsesUpstream(streams=[[answer_round_stream("streamed")]])

    with patch("codex_rosetta.api.router.get_upstream_client", return_value=upstream), \
         TestClient(app) as client:
        response = client.post("/v1/responses", json={
            "model": "deepseek-flash",
            "input": "hi",
            "stream": True,
        })

    assert response.status_code == 200
    assert "streamed" in response.text


class ErroringUpstream:
    async def responses_stream(self, body):
        response = httpx.Response(400, text="unknown field", request=httpx.Request("POST", "http://x"))
        raise httpx.HTTPStatusError("Client error '400 Bad Request'", request=response.request, response=response)
        yield b""  # pragma: no cover - generator marker


@pytest.mark.asyncio
async def test_verbatim_stream_reports_upstream_errors():
    events = await run_stream(_stream_responses_verbatim(
        ErroringUpstream(), {"model": "m", "input": "hi"}, LOG
    ))

    assert events[-1]["type"] == "response.failed"
    assert "400" in events[-1]["response"]["error"]["message"]


def test_chat_mode_default_is_unchanged():
    assert Settings().UPSTREAM_API_MODE == "chat_completions"
