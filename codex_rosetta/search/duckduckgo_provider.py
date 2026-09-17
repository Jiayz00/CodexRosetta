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


class DuckDuckGoSearchProvider(SearchProvider):
    def __init__(self, base_url: str = "", api_key: str = "", timeout: float = 10.0) -> None:
        self._base_url = base_url.rstrip("/") if base_url else ""
        self._api_key = api_key
        self._timeout = timeout
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            },
        )

    async def search(self, query: str, max_results: int = 5) -> SearchResponse:
        if self._base_url:
            return await self._search_remote(query, max_results)
        return await self._search_direct(query, max_results)

    async def _search_remote(self, query: str, max_results: int) -> SearchResponse:
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            response = await self._client.get(
                self._base_url + "/search/text",
                params={"q": query, "max_results": max_results},
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            detail = http_error_detail(e.response.status_code, response_text(e.response))
            kind = classify_status_error(e.response.status_code, response_text(e.response))
            logger.warning(
                "duckduckgo_remote_http_error",
                status=e.response.status_code,
                error_kind=kind,
                detail=detail,
            )
            return SearchResponse(query=query, error=kind, error_detail=detail)
        except httpx.RequestError as e:
            logger.warning("duckduckgo_remote_request_error", error=str(e))
            return SearchResponse(query=query, error=ERROR_NETWORK_ERROR, error_detail=str(e))
        except Exception as e:
            logger.warning("duckduckgo_remote_unexpected_error", error=str(e))
            return SearchResponse(query=query, error=ERROR_UNKNOWN, error_detail=str(e))

        results = []
        for item in data if isinstance(data, list) else data.get("results", []):
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", item.get("href", "")),
                    snippet=item.get("snippet", item.get("body", "")),
                )
            )

        logger.debug("duckduckgo_remote_search_completed", query=query, result_count=len(results))
        return SearchResponse(results=results, query=query)

    async def _search_direct(self, query: str, max_results: int) -> SearchResponse:
        try:
            response = await self._client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
            )
            response.raise_for_status()
            results = self._parse_html_results(response.text, max_results)
        except httpx.HTTPStatusError as e:
            detail = http_error_detail(e.response.status_code, response_text(e.response))
            kind = classify_status_error(e.response.status_code, response_text(e.response))
            logger.warning(
                "duckduckgo_http_error",
                status=e.response.status_code,
                error_kind=kind,
                detail=detail,
            )
            return SearchResponse(query=query, error=kind, error_detail=detail)
        except httpx.RequestError as e:
            logger.warning("duckduckgo_request_error", error=str(e))
            return SearchResponse(query=query, error=ERROR_NETWORK_ERROR, error_detail=str(e))
        except Exception as e:
            logger.warning("duckduckgo_unexpected_error", error=str(e))
            return SearchResponse(query=query, error=ERROR_UNKNOWN, error_detail=str(e))

        logger.debug("duckduckgo_search_completed", query=query, result_count=len(results))
        return SearchResponse(results=results, query=query)

    def _parse_html_results(self, html: str, max_results: int) -> list[SearchResult]:
        results: list[SearchResult] = []
        import re

        pattern = re.compile(
            r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
            r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
            re.DOTALL,
        )

        for match in pattern.finditer(html):
            if len(results) >= max_results:
                break
            url = match.group(1)
            title = re.sub(r"<[^>]+>", "", match.group(2)).strip()
            snippet = re.sub(r"<[^>]+>", "", match.group(3)).strip()
            if url.startswith("//duckduckgo.com/l/"):
                url = url.split("uddg=", 1)
                if len(url) > 1:
                    from urllib.parse import unquote
                    url = unquote(url[1].split("&", 1)[0])
                else:
                    url = ""
            results.append(SearchResult(title=title, url=url, snippet=snippet))

        return results

    async def close(self) -> None:
        await self._client.aclose()
