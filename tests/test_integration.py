import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from codex_rosetta.main import app
from codex_rosetta.search.base import SearchResponse, SearchResult
from codex_rosetta.state.conversation_store import InMemoryConversationStore


def parse_events(content: bytes) -> list[dict]:
    """Parse a Responses API SSE payload into [{event, data}, ...]."""
    events = []
    event_type = None
    for line in content.decode().split("\n"):
        if line.startswith("event:"):
            event_type = line[6:].strip()
        elif line.startswith("data:"):
            try:
                events.append({"event": event_type, "data": json.loads(line[5:].strip())})
            except json.JSONDecodeError:
                pass
    return events


@pytest.fixture
def mock_upstream():
    """Create a mock upstream client."""
    upstream = AsyncMock()
    upstream.chat_completions.return_value = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1746000000,
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "Hello from the model!"},
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    return upstream


@pytest.fixture
def mock_store():
    return InMemoryConversationStore()


class TestHealthEndpoint:
    def test_health(self):
        from fastapi.testclient import TestClient
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}


class TestNonStreamingProxy:
    def test_simple_text_request(self, mock_upstream, mock_store):
        with patch("codex_rosetta.api.router.get_upstream_client", return_value=mock_upstream), \
             patch("codex_rosetta.api.router.get_conversation_store", return_value=mock_store):
            from fastapi.testclient import TestClient
            with TestClient(app) as client:
                resp = client.post("/v1/responses", json={
                    "model": "gpt-4o",
                    "input": "Say hello",
                })

        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "response"
        assert data["status"] == "completed"
        assert data["model"] == "gpt-4o"

        msg_items = [o for o in data["output"] if o["type"] == "message"]
        assert len(msg_items) == 1
        assert msg_items[0]["role"] == "assistant"
        texts = [p["text"] for p in msg_items[0]["content"] if p.get("type") == "output_text"]
        assert "Hello from the model!" in texts

    def test_request_with_instructions(self, mock_upstream, mock_store):
        with patch("codex_rosetta.api.router.get_upstream_client", return_value=mock_upstream), \
             patch("codex_rosetta.api.router.get_conversation_store", return_value=mock_store):
            from fastapi.testclient import TestClient
            with TestClient(app) as client:
                resp = client.post("/v1/responses", json={
                    "model": "gpt-4o",
                    "instructions": "Be concise",
                    "input": "Hi",
                })

        assert resp.status_code == 200
        # Verify system message was sent upstream
        call_args = mock_upstream.chat_completions.call_args[0][0]
        system_msgs = [m for m in call_args["messages"] if m.get("role") == "system"]
        assert len(system_msgs) >= 1
        assert system_msgs[0]["content"] == "Be concise"

    def test_request_with_tool_calls_response(self, mock_upstream, mock_store):
        mock_upstream.chat_completions.return_value = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1746000000,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "Let me check.",
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"location":"SF"}'},
                    }],
                },
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        }

        with patch("codex_rosetta.api.router.get_upstream_client", return_value=mock_upstream), \
             patch("codex_rosetta.api.router.get_conversation_store", return_value=mock_store):
            from fastapi.testclient import TestClient
            with TestClient(app) as client:
                resp = client.post("/v1/responses", json={
                    "model": "gpt-4o",
                    "input": "Weather in SF?",
                    "tools": [
                        {"type": "function", "name": "get_weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}},
                    ],
                })

        data = resp.json()
        fc_items = [o for o in data["output"] if o["type"] == "function_call"]
        assert len(fc_items) == 1
        assert fc_items[0]["call_id"] == "call_abc"
        assert fc_items[0]["name"] == "get_weather"

    def test_usage_mapping(self, mock_upstream, mock_store):
        mock_upstream.chat_completions.return_value = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1746000000,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Hi"},
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "prompt_tokens_details": {"cached_tokens": 30},
                "completion_tokens_details": {"reasoning_tokens": 5},
            },
        }

        with patch("codex_rosetta.api.router.get_upstream_client", return_value=mock_upstream), \
             patch("codex_rosetta.api.router.get_conversation_store", return_value=mock_store):
            from fastapi.testclient import TestClient
            with TestClient(app) as client:
                resp = client.post("/v1/responses", json={"model": "gpt-4o", "input": "Hi"})

        data = resp.json()
        usage = data["usage"]
        assert usage["input_tokens"] == 100
        assert usage["output_tokens"] == 50
        assert usage["total_tokens"] == 150
        assert usage["input_tokens_details"]["cached_tokens"] == 30
        assert usage["output_tokens_details"]["reasoning_tokens"] == 5


class TestStreamingProxy:
    def test_streaming_text(self, mock_upstream, mock_store):
        async def mock_stream(*args, **kwargs):
            chunks = [
                b'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1746000000,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n',
                b'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1746000000,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n',
                b'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1746000000,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}\n\n',
                b'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1746000000,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
                b'data: [DONE]\n\n',
            ]
            for chunk in chunks:
                yield chunk

        mock_upstream.chat_completions_stream = mock_stream

        with patch("codex_rosetta.api.router.get_upstream_client", return_value=mock_upstream), \
             patch("codex_rosetta.api.router.get_conversation_store", return_value=mock_store):
            from fastapi.testclient import TestClient
            with TestClient(app) as client:
                resp = client.post("/v1/responses", json={
                    "model": "gpt-4o",
                    "input": "Hi",
                    "stream": True,
                })

        assert resp.status_code == 200

        content = resp.content.decode()
        events = []
        event_type = None
        for line in content.split("\n"):
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                data_str = line[5:].strip()
                try:
                    events.append({"event": event_type, "data": json.loads(data_str)})
                except json.JSONDecodeError:
                    pass

        event_types = [e["event"] for e in events]
        assert "response.created" in event_types
        assert "response.in_progress" in event_types
        assert "response.output_text.delta" in event_types
        assert "response.completed" in event_types

        deltas = [e for e in events if e["event"] == "response.output_text.delta"]
        texts = [d["data"]["delta"] for d in deltas]
        assert "Hello" in texts
        assert " world" in texts

    def test_streaming_search_streams_rounds_live(self, mock_upstream, mock_store):
        first_round = [
            b'data: {"id":"chatcmpl-search-1","object":"chat.completion.chunk","created":1746000000,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n',
            b'data: {"id":"chatcmpl-search-1","object":"chat.completion.chunk","created":1746000000,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Let me search for that."},"finish_reason":null}]}\n\n',
            b'data: {"id":"chatcmpl-search-1","object":"chat.completion.chunk","created":1746000000,"model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_ws","type":"function","function":{"name":"__rosetta_web_search","arguments":"{\\"query\\":\\"SearXNG project what is it\\"}"}}]},"finish_reason":null}]}\n\n',
            b'data: {"id":"chatcmpl-search-1","object":"chat.completion.chunk","created":1746000000,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n',
            b'data: [DONE]\n\n',
        ]
        final_round = [
            b'data: {"id":"chatcmpl-search-2","object":"chat.completion.chunk","created":1746000001,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n',
            b'data: {"id":"chatcmpl-search-2","object":"chat.completion.chunk","created":1746000001,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"SearXNG is an open-source metasearch engine."},"finish_reason":null}]}\n\n',
            b'data: {"id":"chatcmpl-search-2","object":"chat.completion.chunk","created":1746000001,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":" It aggregates results from multiple search providers."},"finish_reason":null}]}\n\n',
            b'data: {"id":"chatcmpl-search-2","object":"chat.completion.chunk","created":1746000001,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":12,"total_tokens":32}}\n\n',
            b'data: [DONE]\n\n',
        ]

        call_count = {"count": 0}

        async def mock_stream(*args, **kwargs):
            call_count["count"] += 1
            chunks = first_round if call_count["count"] == 1 else final_round
            for chunk in chunks:
                yield chunk

        mock_upstream.chat_completions_stream = mock_stream

        mock_search_provider = AsyncMock()
        mock_search_provider.search.return_value = SearchResponse(
            query="SearXNG project what is it",
            results=[
                SearchResult(
                    title="SearXNG",
                    url="https://example.com",
                    snippet="meta search",
                )
            ],
        )

        with patch("codex_rosetta.api.router.get_upstream_client", return_value=mock_upstream), \
             patch("codex_rosetta.api.router.get_conversation_store", return_value=mock_store), \
             patch("codex_rosetta.api.router._get_search_provider", return_value=mock_search_provider):
            from fastapi.testclient import TestClient
            with TestClient(app) as client:
                resp = client.post("/v1/responses", json={
                    "model": "gpt-4o",
                    "input": "What is SearXNG?",
                    "stream": True,
                    "tools": [{"type": "web_search"}],
                })

        assert resp.status_code == 200
        assert call_count["count"] == 2
        mock_search_provider.search.assert_awaited_once()

        events = parse_events(resp.content)
        event_types = [e["event"] for e in events]

        # the search lifecycle is visible while it happens
        assert "response.web_search_call.in_progress" in event_types
        assert "response.web_search_call.searching" in event_types
        assert "response.web_search_call.completed" in event_types

        deltas = [e["data"]["delta"] for e in events if e["event"] == "response.output_text.delta"]
        combined = "".join(deltas)
        # both rounds stream live now, including the short "let me search" text
        assert "Let me search for that." in combined
        assert "SearXNG is an open-source metasearch engine." in combined

        completed = [e for e in events if e["event"] == "response.completed"][0]["data"]
        output = completed["response"]["output"]
        assert [item["type"] for item in output] == ["web_search_call", "message"]
        assert output[0]["status"] == "completed"
        assert output[0]["action"]["sources"] == [
            {"type": "url", "url": "https://example.com"}
        ]
        output_text = output[1]["content"][0]["text"]
        # only the last round's assistant text survives into the final output
        assert "SearXNG is an open-source metasearch engine." in output_text
        assert "Let me search for that." not in output_text

    def test_streaming_search_answer_deltas_are_not_simulated(self, mock_upstream, mock_store):
        first_round = [
            b'data: {"id":"chatcmpl-search-1","object":"chat.completion.chunk","created":1746000000,"model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_ws","type":"function","function":{"name":"__rosetta_web_search","arguments":"{\\"query\\":\\"SearXNG project what is it\\"}"}}]},"finish_reason":null}]}\n\n',
            b'data: {"id":"chatcmpl-search-1","object":"chat.completion.chunk","created":1746000000,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n',
            b'data: [DONE]\n\n',
        ]
        answer = (
            "SearXNG is an open-source metasearch engine that aggregates results "
            "from multiple providers into a single interface."
        )
        final_round = [
            b'data: {"id":"chatcmpl-search-2","object":"chat.completion.chunk","created":1746000001,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n',
            (
                'data: {"id":"chatcmpl-search-2","object":"chat.completion.chunk","created":1746000001,'
                '"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"' + answer + '"},"finish_reason":null}]}\n\n'
            ).encode(),
            b'data: {"id":"chatcmpl-search-2","object":"chat.completion.chunk","created":1746000001,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":12,"total_tokens":32}}\n\n',
            b'data: [DONE]\n\n',
        ]

        requests: list[dict] = []

        async def mock_stream(request, *args, **kwargs):
            requests.append(request)
            chunks = first_round if len(requests) == 1 else final_round
            for chunk in chunks:
                yield chunk

        mock_upstream.chat_completions_stream = mock_stream

        mock_search_provider = AsyncMock()
        mock_search_provider.search.return_value = SearchResponse(
            query="SearXNG project what is it",
            results=[
                SearchResult(
                    title="SearXNG",
                    url="https://example.com",
                    snippet="meta search",
                )
            ],
        )

        with patch("codex_rosetta.api.router.get_upstream_client", return_value=mock_upstream), \
             patch("codex_rosetta.api.router.get_conversation_store", return_value=mock_store), \
             patch("codex_rosetta.api.router._get_search_provider", return_value=mock_search_provider):
            from fastapi.testclient import TestClient
            with TestClient(app) as client:
                resp = client.post("/v1/responses", json={
                    "model": "gpt-4o",
                    "input": "What is SearXNG?",
                    "stream": True,
                    "tools": [{"type": "web_search"}],
                })

        assert resp.status_code == 200
        assert len(requests) == 2

        events = parse_events(resp.content)
        deltas = [e["data"]["delta"] for e in events if e["event"] == "response.output_text.delta"]
        # no artificial chunk splitting: one upstream delta, one client delta
        assert "".join(deltas) == answer

        completed = [e for e in events if e["event"] == "response.completed"][0]["data"]
        assert completed["response"]["output"][1]["content"][0]["text"] == answer

        # the second round carries the real search digest for the model
        tool_messages = [m for m in requests[1]["messages"] if m.get("role") == "tool"]
        assert tool_messages and "https://example.com" in tool_messages[0]["content"]
