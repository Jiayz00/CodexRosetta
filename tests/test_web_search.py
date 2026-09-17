from __future__ import annotations

import json
import pytest

from codex_rosetta.search.base import SearchResult, SearchResponse, SearchProvider
from codex_rosetta.search.http_provider import HttpSearchProvider
from codex_rosetta.search.formatter import format_search_results
from codex_rosetta.api.router import (
    _extract_web_search_tool_calls,
    _parse_search_query,
    _inject_search_results,
    _chat_request_has_web_search_tool,
)


class MockSearchProvider(SearchProvider):
    def __init__(self, responses: dict[str, SearchResponse] | None = None) -> None:
        self._responses = responses or {}
        self._calls: list[str] = []

    async def search(self, query: str, max_results: int = 5) -> SearchResponse:
        self._calls.append(query)
        return self._responses.get(query, SearchResponse(
            results=[SearchResult(title=f"Result for {query}", url=f"https://example.com/{query}", snippet="Test snippet")],
            query=query,
        ))

    @property
    def calls(self) -> list[str]:
        return self._calls


class TestSearchResult:
    def test_creation(self):
        r = SearchResult(title="Test", url="https://example.com", snippet="Hello")
        assert r.title == "Test"
        assert r.url == "https://example.com"
        assert r.snippet == "Hello"

    def test_default_snippet(self):
        r = SearchResult(title="Test", url="https://example.com")
        assert r.snippet == ""


class TestSearchResponse:
    def test_creation(self):
        r = SearchResponse(results=[], query="test")
        assert r.results == []
        assert r.query == "test"

    def test_default_values(self):
        r = SearchResponse()
        assert r.results == []
        assert r.query == ""


class TestFormatter:
    def test_format_with_results(self):
        resp = SearchResponse(
            results=[
                SearchResult(title="Title 1", url="https://a.com", snippet="Snippet 1"),
                SearchResult(title="Title 2", url="https://b.com"),
            ],
            query="test query",
        )
        output = format_search_results(resp)
        assert "test query" in output
        assert "Title 1" in output
        assert "https://a.com" in output
        assert "Snippet 1" in output
        assert "Title 2" in output
        assert "2 条" in output

    def test_format_empty_results(self):
        resp = SearchResponse(results=[], query="nothing")
        output = format_search_results(resp)
        assert "未返回任何结果" in output


class TestExtractWebSearchToolCalls:
    def test_extracts_web_search_calls(self):
        response = {
            "choices": [{
                "message": {
                    "tool_calls": [
                        {"id": "tc1", "function": {"name": "__rosetta_web_search", "arguments": '{"query": "test"}'}},
                        {"id": "tc2", "function": {"name": "other_func", "arguments": '{}'}},
                    ]
                }
            }]
        }
        result = _extract_web_search_tool_calls(response)
        assert len(result) == 1
        assert result[0]["id"] == "tc1"

    def test_extracts_web_search_2025_calls(self):
        response = {
            "choices": [{
                "message": {
                    "tool_calls": [
                        {"id": "tc1", "function": {"name": "__rosetta_web_search_2025_08_26", "arguments": '{"query": "test"}'}},
                    ]
                }
            }]
        }
        result = _extract_web_search_tool_calls(response)
        assert len(result) == 1

    def test_no_tool_calls(self):
        response = {"choices": [{"message": {"content": "Hello"}}]}
        result = _extract_web_search_tool_calls(response)
        assert result == []

    def test_no_web_search_calls(self):
        response = {
            "choices": [{
                "message": {
                    "tool_calls": [
                        {"id": "tc1", "function": {"name": "other_func", "arguments": '{}'}},
                    ]
                }
            }]
        }
        result = _extract_web_search_tool_calls(response)
        assert result == []


class TestParseSearchQuery:
    def test_parse_query(self):
        tc = {"function": {"arguments": '{"query": "python async"}'}}
        assert _parse_search_query(tc) == "python async"

    def test_parse_empty_query(self):
        tc = {"function": {"arguments": '{}'}}
        assert _parse_search_query(tc) == ""

    def test_parse_invalid_json(self):
        tc = {"function": {"arguments": "not json"}}
        assert _parse_search_query(tc) == ""

    def test_parse_dict_arguments(self):
        tc = {"function": {"arguments": {"query": "test"}}}
        assert _parse_search_query(tc) == "test"


class TestInjectSearchResults:
    def test_inject_results(self):
        chat_request = {"messages": [{"role": "user", "content": "search for python"}], "model": "gpt-4"}
        chat_response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "tc1", "type": "function", "function": {"name": "__rosetta_web_search", "arguments": '{"query": "python"}'}},
                    ]
                }
            }]
        }
        search_results = {"tc1": "搜索结果：1. Python官网 https://python.org"}

        result = _inject_search_results(chat_request, chat_response, search_results)

        messages = result["messages"]
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["tool_calls"][0]["id"] == "tc1"
        assert messages[2]["role"] == "tool"
        assert messages[2]["tool_call_id"] == "tc1"
        assert "Python官网" in messages[2]["content"]


class TestMockSearchProvider:
    @pytest.mark.asyncio
    async def test_mock_search(self):
        provider = MockSearchProvider()
        result = await provider.search("test query")
        assert result.query == "test query"
        assert len(result.results) == 1
        assert result.results[0].title == "Result for test query"

    @pytest.mark.asyncio
    async def test_mock_search_custom_response(self):
        provider = MockSearchProvider(responses={
            "custom": SearchResponse(results=[SearchResult(title="Custom", url="https://custom.com")], query="custom")
        })
        result = await provider.search("custom")
        assert result.results[0].title == "Custom"

    @pytest.mark.asyncio
    async def test_mock_tracks_calls(self):
        provider = MockSearchProvider()
        await provider.search("query1")
        await provider.search("query2")
        assert provider.calls == ["query1", "query2"]


class TestSearchToolDetection:
    def test_detects_web_search_tool_in_chat_request(self):
        assert _chat_request_has_web_search_tool({
            "tools": [{
                "type": "function",
                "function": {"name": "__rosetta_web_search"},
            }]
        }) is True

    def test_ignores_non_search_tools(self):
        assert _chat_request_has_web_search_tool({
            "tools": [{
                "type": "function",
                "function": {"name": "get_weather"},
            }]
        }) is False


class TestErrorAwareFormatter:
    def test_error_response_renders_unavailable_instruction(self):
        from codex_rosetta.search.base import ERROR_QUOTA_EXCEEDED

        output = format_search_results(
            SearchResponse(error=ERROR_QUOTA_EXCEEDED, error_detail="HTTP 432: quota")
        )
        assert "搜索服务暂不可用" in output
        assert "搜索服务额度已用尽" in output
        assert "不要再用 web_search 重试" in output

    def test_rejected_query_asks_for_a_rephrase(self):
        from codex_rosetta.search.base import ERROR_INVALID_QUERY

        output = format_search_results(SearchResponse(
            error=ERROR_INVALID_QUERY,
            error_detail="HTTP 400: Query cannot consist only of site: operators.",
        ))
        assert "搜索查询被拒绝" in output
        assert "重新调用 web_search" in output
        assert "暂不可用" not in output

    def test_empty_response_is_not_an_error(self):
        output = format_search_results(SearchResponse(results=[], query="nothing"))
        assert "未返回任何结果" in output
        assert "暂不可用" not in output

    def test_search_response_error_flag(self):
        from codex_rosetta.search.base import ERROR_INVALID_KEY

        assert SearchResponse(results=[]).ok is True
        assert SearchResponse(error=ERROR_INVALID_KEY).ok is False


class TestProviderErrorClassification:
    @staticmethod
    def _tavily():
        from codex_rosetta.search.tavily_provider import TavilySearchProvider

        return TavilySearchProvider

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status,expected",
        [
            (400, "invalid_query"),
            (401, "invalid_key"),
            (403, "invalid_key"),
            (432, "quota_exceeded"),
            (429, "rate_limited"),
            (503, "upstream_error"),
        ],
    )
    async def test_tavily_maps_http_status(self, status, expected):
        import httpx

        provider = self._tavily()(api_key="test-key")
        await provider._client.aclose()
        provider._client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(status, text="boom"))
        )

        response = await provider.search("query")

        assert response.error == expected
        assert response.error_detail.startswith(f"HTTP {status}")
        assert response.results == []
        await provider.close()

    @pytest.mark.asyncio
    async def test_tavily_reports_network_error(self):
        import httpx

        def handler(request):
            raise httpx.ConnectError("dns failure", request=request)

        provider = self._tavily()(api_key="test-key")
        await provider._client.aclose()
        provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

        response = await provider.search("query")

        assert response.error == "network_error"
        assert "dns failure" in response.error_detail
        await provider.close()

    @pytest.mark.asyncio
    async def test_tavily_success_is_unchanged(self):
        import httpx

        payload = {
            "results": [
                {"title": "T", "url": "https://example.com", "content": "snippet"},
            ]
        }
        provider = self._tavily()(api_key="test-key")
        await provider._client.aclose()
        provider._client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
        )

        response = await provider.search("query")

        assert response.error is None
        assert response.results[0].title == "T"
        await provider.close()


class TestDuckDuckGoProvider:
    def test_module_imports(self):
        # Regression: _search_direct() used to be missing its `try:` block,
        # which made the whole module fail to import.
        import importlib

        module = importlib.import_module("codex_rosetta.search.duckduckgo_provider")
        assert hasattr(module, "DuckDuckGoSearchProvider")

    def test_html_result_parsing(self):
        from codex_rosetta.search.duckduckgo_provider import DuckDuckGoSearchProvider

        provider = DuckDuckGoSearchProvider()
        html = (
            '<a class="result__a" href="https://example.com/one">First <b>Title</b></a>'
            '<a class="result__snippet">First snippet</a>'
            '<a class="result__a" href="https://example.com/two">Second</a>'
            '<a class="result__snippet">Second snippet</a>'
        )

        results = provider._parse_html_results(html, max_results=5)

        assert [r.title for r in results] == ["First Title", "Second"]
        assert [r.url for r in results] == [
            "https://example.com/one",
            "https://example.com/two",
        ]
        assert results[0].snippet == "First snippet"

    def test_html_parsing_respects_max_results(self):
        from codex_rosetta.search.duckduckgo_provider import DuckDuckGoSearchProvider

        provider = DuckDuckGoSearchProvider()
        html = "".join(
            f'<a class="result__a" href="https://example.com/{i}">T{i}</a>'
            f'<a class="result__snippet">S{i}</a>'
            for i in range(5)
        )

        results = provider._parse_html_results(html, max_results=2)
        assert len(results) == 2
