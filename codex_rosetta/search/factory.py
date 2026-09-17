from __future__ import annotations

from codex_rosetta.search.base import SearchProvider


def is_provider_configured(
    provider: str,
    api_key: str = "",
    base_url: str = "",
) -> bool:
    """Whether the given provider settings can run a search at all.

    Used to decide whether the legacy single-credential settings are worth
    migrating into the pool: migrating an unusable legacy config would only
    seed an entry that fails with ``invalid_key``.
    """
    name = (provider or "").strip().lower()
    if name in ("tavily", "brave"):
        return bool(api_key)
    if name == "searxng":
        return bool(base_url)
    if name == "duckduckgo":
        return True
    if name in ("", "custom", "http"):
        return bool(base_url)
    return False


def create_provider(
    provider: str,
    api_key: str = "",
    base_url: str = "",
) -> SearchProvider | None:
    """Build a search provider from a pool entry or legacy settings.

    Returns ``None`` when the configuration is incomplete (for example a
    Tavily entry without an API key), so callers can treat it as an
    unusable credential instead of silently searching with an empty key.
    """
    name = (provider or "").strip().lower()

    if name == "tavily":
        if not api_key:
            return None
        from codex_rosetta.search.tavily_provider import TavilySearchProvider

        return TavilySearchProvider(api_key=api_key)

    if name == "brave":
        if not api_key:
            return None
        from codex_rosetta.search.brave_provider import BraveSearchProvider

        return BraveSearchProvider(api_key=api_key)

    if name == "searxng":
        if not base_url:
            return None
        from codex_rosetta.search.searxng_provider import SearXNGSearchProvider

        return SearXNGSearchProvider(base_url=base_url, api_key=api_key)

    if name == "duckduckgo":
        from codex_rosetta.search.duckduckgo_provider import DuckDuckGoSearchProvider

        return DuckDuckGoSearchProvider(base_url=base_url, api_key=api_key)

    if name in ("", "custom", "http"):
        if not base_url:
            return None
        from codex_rosetta.search.http_provider import HttpSearchProvider

        return HttpSearchProvider(base_url=base_url, api_key=api_key)

    return None
