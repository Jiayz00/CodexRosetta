from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

# Error kinds reported by search providers. Turning failures into typed
# errors (instead of an empty result list) is what allows the router to tell
# "nothing matched" apart from "the search backend is unusable".
ERROR_INVALID_KEY = "invalid_key"
ERROR_QUOTA_EXCEEDED = "quota_exceeded"
ERROR_RATE_LIMITED = "rate_limited"
ERROR_UPSTREAM_ERROR = "upstream_error"
ERROR_NETWORK_ERROR = "network_error"
ERROR_INVALID_QUERY = "invalid_query"
ERROR_UNKNOWN = "unknown_error"

PROVIDER_ERROR_KINDS = (
    ERROR_INVALID_KEY,
    ERROR_QUOTA_EXCEEDED,
    ERROR_RATE_LIMITED,
    ERROR_UPSTREAM_ERROR,
    ERROR_NETWORK_ERROR,
    ERROR_INVALID_QUERY,
    ERROR_UNKNOWN,
)

# The query itself was rejected (for example "site:..." with no search terms).
# The credential is fine, so it must not be cooled down or skipped.
QUERY_ERROR_KINDS = frozenset({ERROR_INVALID_QUERY})

# Errors that justify moving on to the next credential in the pool.
RETRYABLE_ERROR_KINDS = frozenset(kind for kind in PROVIDER_ERROR_KINDS if kind not in QUERY_ERROR_KINDS)

_QUOTA_HINTS = (
    "quota",
    "exceeded",
    "insufficient",
    "balance",
    "credit",
    "plan limit",
    "rate limit",
    "too many requests",
    "额度",
    "配额",
    "余额",
)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""


@dataclass
class SearchResponse:
    results: list[SearchResult] = field(default_factory=list)
    query: str = ""
    error: str | None = None
    error_detail: str = ""

    @property
    def ok(self) -> bool:
        return self.error is None


def classify_status_error(status_code: int, body: str = "") -> str:
    """Map an HTTP status (plus response body) to a provider error kind."""
    if status_code in (401, 403):
        return ERROR_INVALID_KEY
    if status_code in (400, 422):
        # The credential worked, the search engine just refused the query.
        return ERROR_INVALID_QUERY
    if status_code == 432:
        return ERROR_QUOTA_EXCEEDED
    if status_code == 429:
        lowered = (body or "").lower()
        if any(hint in lowered for hint in _QUOTA_HINTS):
            return ERROR_QUOTA_EXCEEDED
        return ERROR_RATE_LIMITED
    if status_code >= 500:
        return ERROR_UPSTREAM_ERROR
    return ERROR_UPSTREAM_ERROR


def http_error_detail(status_code: int, body: str = "", limit: int = 200) -> str:
    """Build a short, log friendly description of an HTTP error response."""
    snippet = " ".join((body or "").split())[:limit]
    if snippet:
        return f"HTTP {status_code}: {snippet}"
    return f"HTTP {status_code}"


def response_text(response: Any) -> str:
    """Best-effort text extraction from an httpx response."""
    try:
        return response.text or ""
    except Exception:  # pragma: no cover - defensive
        return ""


class SearchProvider(ABC):
    @abstractmethod
    async def search(self, query: str, max_results: int = 5) -> SearchResponse: ...

    async def close(self) -> None:
        """Release any resources held by the provider."""
        return None
