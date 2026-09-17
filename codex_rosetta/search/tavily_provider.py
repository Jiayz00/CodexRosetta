from __future__ import annotations

import httpx

from codex_rosetta.search.base import (
    ERROR_NETWORK_ERROR,
    ERROR_UNKNOWN,
    SearchProvider,
    SearchResponse,
    SearchResult,
    classify_status_error,
    http_error_detail,
    response_text,
)
from codex_rosetta.utils.logging import get_logger

logger = get_logger("search")


class TavilySearchProvider(SearchProvider):
    def __init__(self, api_key: str, timeout: float = 10.0) -> None:
        self._api_key = api_key
        self._timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    async def search(self, query: str, max_results: int = 5) -> SearchResponse:
        try:
            response = await self._client.post(
                "https://api.tavily.com/search",
                json={
                    "query": query,
                    "max_results": max_results,
                    "api_key": self._api_key,
                },
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            detail = http_error_detail(e.response.status_code, response_text(e.response))
            kind = classify_status_error(e.response.status_code, response_text(e.response))
            logger.warning(
                "tavily_http_error",
                status=e.response.status_code,
                error_kind=kind,
                detail=detail,
            )
            return SearchResponse(query=query, error=kind, error_detail=detail)
        except httpx.RequestError as e:
            logger.warning("tavily_request_error", error=str(e))
            return SearchResponse(query=query, error=ERROR_NETWORK_ERROR, error_detail=str(e))
        except Exception as e:
            logger.warning("tavily_unexpected_error", error=str(e))
            return SearchResponse(query=query, error=ERROR_UNKNOWN, error_detail=str(e))

        results = []
        for item in data.get("results", []):
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                )
            )

        logger.debug("tavily_search_completed", query=query, result_count=len(results))
        return SearchResponse(results=results, query=query)

    async def close(self) -> None:
        await self._client.aclose()
