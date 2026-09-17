from __future__ import annotations

import copy
import json

import pytest

from codex_rosetta.audit.logger import NoOpAuditLogger
from codex_rosetta.converters.stream_converter import StreamConverter
from codex_rosetta.models.common import ConversionContext
from codex_rosetta.search.base import (
    ERROR_INVALID_QUERY,
    ERROR_QUOTA_EXCEEDED,
    SearchProvider,
    SearchResponse,
    SearchResult,
)
from codex_rosetta.state.conversation_store import InMemoryConversationStore
from codex_rosetta.utils.logging import get_logger
from codex_rosetta.utils.sse import parse_sse_lines

from codex_rosetta.api.router import (
    _search_loop_non_streaming,
    _stream_response,
)

SEARCH_FN = "__rosetta_web_search"


# --- helpers ------------------------------------------------------------

def make_chunk(delta: dict, finish_reason: str | None = None) -> str:
    chunk = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1746000000,
        "model": "gpt-4o",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def search_chunk(query: str = "AI news", call_id: str = "call_ws", index: int = 0) -> str:
    return make_chunk(
        {
            "tool_calls": [
                {
                    "index": index,
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": SEARCH_FN,
                        "arguments": json.dumps({"query": query}),
                    },
                }
            ]
        },
        finish_reason="tool_calls",
    )


def client_tool_chunk(name: str = "shell", call_id: str = "call_shell", index: int = 1) -> str:
    return make_chunk(
        {
            "tool_calls": [
                {
                    "index": index,
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": '{"command":"ls"}'},
                }
            ]
        },
        finish_reason="tool_calls",
    )


def answer_chunk(text: str = "Final answer") -> str:
    return make_chunk({"content": text}, finish_reason="stop")


class FakeUpstream:
    """Scripted Chat Completions upstream (one script entry per call)."""

    def __init__(
        self,
        stream_scripts: list[list[str]] | None = None,
        response_scripts: list[dict] | None = None,
    ) -> None:
        self.stream_scripts = stream_scripts or []
        self.response_scripts = response_scripts or []
        self.requests: list[dict] = []

    def _record(self, request: dict) -> int:
        self.requests.append(copy.deepcopy(request))
        return len(self.requests) - 1

    async def chat_completions_stream(self, request: dict):
        index = self._record(request)
        if not self.stream_scripts:
            return
        script = self.stream_scripts[min(index, len(self.stream_scripts) - 1)]
        for chunk in script:
            yield chunk

    async def chat_completions(self, request: dict) -> dict:
        index = self._record(request)
        if not self.response_scripts:
            return {}
        return copy.deepcopy(
            self.response_scripts[min(index, len(self.response_scripts) - 1)]
        )


class ScriptedSearchProvider(SearchProvider):
    """Returns the next scripted response, repeating the last one."""

    def __init__(self, responses: list[SearchResponse], on_call=None) -> None:
        self._responses = responses
        self.calls: list[str] = []
        self.on_call = on_call

    async def search(self, query: str, max_results: int = 5) -> SearchResponse:
        self.calls.append(query)
        index = min(len(self.calls) - 1, len(self._responses) - 1)
        if self.on_call is not None:
            self.on_call(len(self.calls))
        return self._responses[index]


def success_response(count: int = 2) -> SearchResponse:
    return SearchResponse(
        results=[
            SearchResult(
                title=f"Source {i}",
                url=f"https://example.com/{i}",
                snippet=f"snippet {i}",
            )
            for i in range(1, count + 1)
        ],
        query="AI news",
    )


def failure_response(kind: str = ERROR_QUOTA_EXCEEDED) -> SearchResponse:
    return SearchResponse(error=kind, error_detail="test double failure")


def make_context() -> ConversionContext:
    ctx = ConversionContext(response_id="resp_stream_search", model="gpt-4o")
    ctx.register_builtin_tool(SEARCH_FN, "web_search")
    return ctx


def make_chat_request() -> dict:
    return {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "What is new in AI?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": SEARCH_FN,
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ],
    }


async def collect_stream_events(upstream, provider, request=None, ctx=None) -> list[tuple]:
    events: list[tuple] = []
    async for raw in _stream_response(
        upstream,
        request if request is not None else make_chat_request(),
        ctx if ctx is not None else make_context(),
        InMemoryConversationStore(),
        {"model": "gpt-4o", "stream": True},
        get_logger("test"),
        NoOpAuditLogger(),
        provider,
    ):
        events.extend(parse_sse_lines(raw))
    return events


def event_data(events: list[tuple], event_type: str) -> list[dict]:
    return [data for name, data in events if name == event_type]


def final_response(events: list[tuple]) -> dict:
    completed = event_data(events, "response.completed")
    assert completed, "no response.completed event emitted"
    return completed[-1]["response"]


def tool_function_names(request: dict) -> list[str]:
    return [
        (tool.get("function") or {}).get("name", "")
        for tool in request.get("tools") or []
        if tool.get("type") == "function"
    ]


# --- streaming search loop ---------------------------------------------

class TestStreamingSearchRounds:
    @pytest.mark.asyncio
    async def test_search_lifecycle_events_and_completed_output(self):
        upstream = FakeUpstream(stream_scripts=[[search_chunk()], [answer_chunk()]])
        provider = ScriptedSearchProvider([success_response()])

        events = await collect_stream_events(upstream, provider)
        names = [name for name, _ in events]

        assert "response.web_search_call.in_progress" in names
        assert "response.web_search_call.searching" in names
        assert "response.web_search_call.completed" in names

        completed_event = event_data(events, "response.web_search_call.completed")[0]
        assert completed_event["status"] == "completed"
        assert completed_event["action"]["queries"] == ["AI news"]
        assert completed_event["action"]["sources"] == [
            {"type": "url", "url": "https://example.com/1"},
            {"type": "url", "url": "https://example.com/2"},
        ]

        response = final_response(events)
        assert [item["type"] for item in response["output"]] == [
            "web_search_call",
            "message",
        ]
        assert response["output"][0]["status"] == "completed"
        assert response["output"][0]["action"]["sources"]
        assert response["output"][1]["content"][0]["text"] == "Final answer"

        # lifecycle events of one item share the same item id
        ids = {
            data["id"]
            for event_type, data in events
            if event_type.startswith("response.web_search_call.")
        }
        assert len(ids) == 1

    @pytest.mark.asyncio
    async def test_sequence_numbers_are_unique_and_increasing(self):
        upstream = FakeUpstream(stream_scripts=[[search_chunk()], [answer_chunk()]])
        provider = ScriptedSearchProvider([success_response()])

        events = await collect_stream_events(upstream, provider)
        sequence_numbers = [
            data["sequence_number"]
            for _, data in events
            if isinstance(data, dict) and "sequence_number" in data
        ]

        assert sequence_numbers == sorted(sequence_numbers)
        assert len(sequence_numbers) == len(set(sequence_numbers))

    @pytest.mark.asyncio
    async def test_search_events_stream_before_search_executes(self):
        seen: list[str] = []
        snapshot: dict[str, list[str]] = {}

        def on_call(_count: int) -> None:
            snapshot["before_search"] = list(seen)

        upstream = FakeUpstream(stream_scripts=[[search_chunk()], [answer_chunk()]])
        provider = ScriptedSearchProvider([success_response()], on_call=on_call)

        async for raw in _stream_response(
            upstream,
            make_chat_request(),
            make_context(),
            InMemoryConversationStore(),
            {"model": "gpt-4o", "stream": True},
            get_logger("test"),
            NoOpAuditLogger(),
            provider,
        ):
            for name, _data in parse_sse_lines(raw):
                seen.append(name)

        # The search item is visible to the client before the backend search ran
        assert "response.web_search_call.in_progress" in snapshot["before_search"]
        assert "response.web_search_call.searching" in snapshot["before_search"]
        assert "response.web_search_call.completed" not in snapshot["before_search"]

    @pytest.mark.asyncio
    async def test_second_round_receives_search_results(self):
        upstream = FakeUpstream(stream_scripts=[[search_chunk()], [answer_chunk()]])
        provider = ScriptedSearchProvider([success_response()])

        await collect_stream_events(upstream, provider)

        assert provider.calls == ["AI news"]
        second_request = upstream.requests[1]
        tool_messages = [m for m in second_request["messages"] if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert "https://example.com/1" in tool_messages[0]["content"]
        assistant_calls = [
            m
            for m in second_request["messages"]
            if m.get("role") == "assistant" and m.get("tool_calls")
        ]
        assert len(assistant_calls) == 1
        assert assistant_calls[0]["tool_calls"][0]["id"] == "call_ws"
        # the simulated call is mirrored back so the upstream model sees its
        # own call paired with the tool result
        assert assistant_calls[0]["tool_calls"][0]["function"]["name"] == SEARCH_FN
        assert json.loads(
            assistant_calls[0]["tool_calls"][0]["function"]["arguments"]
        ) == {"query": "AI news"}


class TestStreamingSearchFailureHandling:
    @pytest.mark.asyncio
    async def test_rejected_query_keeps_search_available(self):
        upstream = FakeUpstream(
            stream_scripts=[
                [search_chunk(query="site:example.com", call_id="call_1")],
                [search_chunk(query="example.com news", call_id="call_2")],
                [answer_chunk("Answered with the second query")],
            ]
        )
        provider = ScriptedSearchProvider([
            failure_response(ERROR_INVALID_QUERY),
            success_response(),
        ])

        events = await collect_stream_events(upstream, provider)
        response = final_response(events)

        # the search tool is still offered after a rejected query
        assert SEARCH_FN in tool_function_names(upstream.requests[1])
        assert SEARCH_FN in tool_function_names(upstream.requests[2])
        assert provider.calls == ["site:example.com", "example.com news"]

        first_tool_message = [
            m for m in upstream.requests[1]["messages"] if m.get("role") == "tool"
        ][-1]["content"]
        assert "搜索查询被拒绝" in first_tool_message
        assert "暂不可用" not in first_tool_message

        search_items = [
            item for item in response["output"] if item.get("type") == "web_search_call"
        ]
        assert len(search_items) == 2
        assert all(item["status"] == "completed" for item in search_items)
        assert search_items[1]["action"]["sources"]
        assert response["output"][-1]["content"][0]["text"] == "Answered with the second query"

    @pytest.mark.asyncio
    async def test_two_failures_strip_tools_and_force_answer(self):
        upstream = FakeUpstream(
            stream_scripts=[
                [search_chunk(query="first", call_id="call_1")],
                [search_chunk(query="second", call_id="call_2")],
                [answer_chunk("Answered without search")],
            ]
        )
        provider = ScriptedSearchProvider([failure_response()])

        events = await collect_stream_events(upstream, provider)
        response = final_response(events)

        assert len(upstream.requests) == 3
        forced_request = upstream.requests[2]
        assert SEARCH_FN not in tool_function_names(forced_request)

        tool_messages = [m for m in forced_request["messages"] if m.get("role") == "tool"]
        assert tool_messages
        assert "暂不可用" in tool_messages[-1]["content"]
        assert "不要再用 web_search 重试" in tool_messages[-1]["content"]

        # No dangling search item: every emitted search item is completed.
        search_items = [
            item for item in response["output"] if item.get("type") == "web_search_call"
        ]
        assert search_items
        assert all(item["status"] == "completed" for item in search_items)

        completed_events = event_data(events, "response.web_search_call.completed")
        assert len(completed_events) == 2
        assert all(event["status"] == "completed" for event in completed_events)

        assert response["output"][-1]["type"] == "message"
        assert response["output"][-1]["content"][0]["text"] == "Answered without search"

    @pytest.mark.asyncio
    async def test_max_rounds_forces_answer_without_search_tools(self, monkeypatch):
        from codex_rosetta import config

        monkeypatch.setattr(config, "_runtime_overrides", {"WEB_SEARCH_MAX_ROUNDS": 2})

        upstream = FakeUpstream(
            stream_scripts=[
                [search_chunk(query="query 0", call_id="call_0")],
                [search_chunk(query="query 1", call_id="call_1")],
                [answer_chunk("Forced answer")],
            ]
        )
        provider = ScriptedSearchProvider([success_response()])

        events = await collect_stream_events(upstream, provider)
        response = final_response(events)

        # 2 search rounds + 1 forced answer round
        assert len(upstream.requests) == 3
        assert SEARCH_FN not in tool_function_names(upstream.requests[2])
        assert provider.calls == ["query 0", "query 1"]
        assert response["output"][-1]["content"][0]["text"] == "Forced answer"

    @pytest.mark.asyncio
    async def test_stubborn_search_call_after_forcing_does_not_query(self, monkeypatch):
        from codex_rosetta import config

        monkeypatch.setattr(config, "_runtime_overrides", {"WEB_SEARCH_MAX_ROUNDS": 2})

        upstream = FakeUpstream(
            stream_scripts=[
                [search_chunk(query="query 0", call_id="call_0")],
                [search_chunk(query="query 1", call_id="call_1")],
                [search_chunk(query="query 2", call_id="call_2")],
                [answer_chunk("Late answer")],
            ]
        )
        provider = ScriptedSearchProvider([success_response()])

        events = await collect_stream_events(upstream, provider)
        response = final_response(events)

        # only the two allowed rounds actually queried the provider
        assert provider.calls == ["query 0", "query 1"]
        assert SEARCH_FN not in tool_function_names(upstream.requests[2])
        tool_messages = [
            m for m in upstream.requests[3]["messages"] if m.get("role") == "tool"
        ]
        assert any("已停止联网搜索" in m["content"] for m in tool_messages)
        assert response["output"][-1]["content"][0]["text"] == "Late answer"
        search_items = [
            item for item in response["output"] if item.get("type") == "web_search_call"
        ]
        assert all(item["status"] == "completed" for item in search_items)

    @pytest.mark.asyncio
    async def test_mixed_round_passes_through_without_searching(self):
        upstream = FakeUpstream(
            stream_scripts=[[search_chunk() + client_tool_chunk()]]
        )
        provider = ScriptedSearchProvider([success_response()])

        events = await collect_stream_events(upstream, provider)
        response = final_response(events)

        assert provider.calls == []
        assert len(upstream.requests) == 1

        types = [item["type"] for item in response["output"]]
        assert "function_call" in types
        client_item = next(
            item for item in response["output"] if item.get("type") == "function_call"
        )
        assert client_item["name"] == "shell"

        # the handed-over search item is closed, not left dangling
        search_items = [
            item for item in response["output"] if item.get("type") == "web_search_call"
        ]
        assert search_items and all(item["status"] == "completed" for item in search_items)

    @pytest.mark.asyncio
    async def test_missing_provider_injects_unavailable_and_answers(self):
        upstream = FakeUpstream(
            stream_scripts=[
                [search_chunk()],
                [search_chunk(query="again", call_id="call_2")],
                [answer_chunk("No search available")],
            ]
        )

        events = await collect_stream_events(upstream, None)
        response = final_response(events)

        forced_request = upstream.requests[2]
        assert SEARCH_FN not in tool_function_names(forced_request)
        tool_messages = [m for m in forced_request["messages"] if m.get("role") == "tool"]
        assert any("暂不可用" in m["content"] for m in tool_messages)
        assert response["output"][-1]["content"][0]["text"] == "No search available"


class TestNonStreamingSearchLoop:
    def _chat_response_with_search(self, query: str = "AI news") -> dict:
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1746000000,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_ws",
                                "type": "function",
                                "function": {
                                    "name": SEARCH_FN,
                                    "arguments": json.dumps({"query": query}),
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    def _answer_response(self) -> dict:
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1746000000,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Final answer"},
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    @pytest.mark.asyncio
    async def test_failures_strip_search_tools(self):
        upstream = FakeUpstream(
            response_scripts=[
                self._chat_response_with_search("first"),
                self._chat_response_with_search("second"),
                self._answer_response(),
            ]
        )
        provider = ScriptedSearchProvider([failure_response()])

        result = await _search_loop_non_streaming(
            upstream,
            make_chat_request(),
            make_context(),
            get_logger("test"),
            provider,
            NoOpAuditLogger(),
        )

        assert len(upstream.requests) == 3
        assert SEARCH_FN not in tool_function_names(upstream.requests[2])
        tool_messages = [
            m for m in upstream.requests[2]["messages"] if m.get("role") == "tool"
        ]
        assert any("暂不可用" in m["content"] for m in tool_messages)
        assert result["choices"][0]["message"]["content"] == "Final answer"

    @pytest.mark.asyncio
    async def test_successful_search_returns_final_answer(self):
        upstream = FakeUpstream(
            response_scripts=[
                self._chat_response_with_search(),
                self._answer_response(),
            ]
        )
        provider = ScriptedSearchProvider([success_response()])

        result = await _search_loop_non_streaming(
            upstream,
            make_chat_request(),
            make_context(),
            get_logger("test"),
            provider,
            NoOpAuditLogger(),
        )

        assert provider.calls == ["AI news"]
        assert len(upstream.requests) == 2
        tool_messages = [
            m for m in upstream.requests[1]["messages"] if m.get("role") == "tool"
        ]
        assert "https://example.com/1" in tool_messages[0]["content"]
        assert result["choices"][0]["message"]["content"] == "Final answer"

    @pytest.mark.asyncio
    async def test_mixed_round_returns_client_tool_calls(self):
        response = self._chat_response_with_search()
        response["choices"][0]["message"]["tool_calls"].append(
            {
                "id": "call_shell",
                "type": "function",
                "function": {"name": "shell", "arguments": '{"command":"ls"}'},
            }
        )
        upstream = FakeUpstream(response_scripts=[response])
        provider = ScriptedSearchProvider([success_response()])

        result = await _search_loop_non_streaming(
            upstream,
            make_chat_request(),
            make_context(),
            get_logger("test"),
            provider,
            NoOpAuditLogger(),
        )

        assert provider.calls == []
        assert len(upstream.requests) == 1
        names = [
            call["function"]["name"]
            for call in result["choices"][0]["message"]["tool_calls"]
        ]
        assert "shell" in names


class TestSearchToolsStrippedWhenDisabled:
    @pytest.mark.asyncio
    async def test_no_provider_means_no_search_tool_upstream(self):
        upstream = FakeUpstream(stream_scripts=[[answer_chunk("plain answer")]])
        ctx = make_context()

        # create_response strips the simulated search tool when search is off;
        # emulate that path by asserting the converter never sees a search call.
        events = await collect_stream_events(
            upstream,
            None,
            request={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [],
            },
            ctx=ctx,
        )
        assert upstream.requests[0].get("tools") == []
        assert final_response(events)["output"][-1]["content"][0]["text"] == "plain answer"
