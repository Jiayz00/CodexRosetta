from __future__ import annotations

import json

import pytest

from codex_rosetta.search import pool as pool_module
from codex_rosetta.search.base import (
    ERROR_INVALID_KEY,
    ERROR_INVALID_QUERY,
    ERROR_NETWORK_ERROR,
    ERROR_QUOTA_EXCEEDED,
    ERROR_RATE_LIMITED,
    ERROR_UNKNOWN,
    ERROR_UPSTREAM_ERROR,
    SearchProvider,
    SearchResponse,
    SearchResult,
    QUERY_ERROR_KINDS,
    RETRYABLE_ERROR_KINDS,
    classify_status_error,
)
from codex_rosetta.search.pool import (
    COOLDOWN_SECONDS,
    SearchPoolEntry,
    SearchPoolStore,
    SearchProviderPool,
    get_search_pool,
    invalidate_search_pool,
    load_pool_entries,
    save_pool_entries,
)


class ScriptedProvider(SearchProvider):
    """Provider whose next result is scripted per API key."""

    def __init__(self, responses: dict[str, SearchResponse], key: str) -> None:
        self._responses = responses
        self._key = key
        self.calls = 0

    async def search(self, query: str, max_results: int = 5) -> SearchResponse:
        self.calls += 1
        return self._responses[self._key]

    async def close(self) -> None:
        return None


def ok_response(query: str = "q", count: int = 1) -> SearchResponse:
    return SearchResponse(
        results=[
            SearchResult(title=f"T{i}", url=f"https://example.com/{i}", snippet="s")
            for i in range(count)
        ],
        query=query,
    )


def err_response(kind: str) -> SearchResponse:
    return SearchResponse(error=kind, error_detail=f"{kind} from test double")


def patch_providers(monkeypatch, responses: dict[str, SearchResponse]):
    """Patch create_provider so each pool key maps to a scripted provider."""
    created: dict[str, ScriptedProvider] = {}

    def fake_create(provider: str, api_key: str = "", base_url: str = ""):
        if provider not in responses:
            return None
        if provider not in created:
            created[provider] = ScriptedProvider(responses, provider)
        return created[provider]

    monkeypatch.setattr(pool_module, "create_provider", fake_create)
    return created


def entry(entry_id: str, provider: str, **kwargs) -> SearchPoolEntry:
    return SearchPoolEntry(
        id=entry_id,
        provider=provider,
        api_key=kwargs.pop("api_key", f"{provider}-key"),
        **kwargs,
    )


class TestErrorClassification:
    @pytest.mark.parametrize(
        "status,expected",
        [
            (400, ERROR_INVALID_QUERY),
            (422, ERROR_INVALID_QUERY),
            (401, ERROR_INVALID_KEY),
            (403, ERROR_INVALID_KEY),
            (432, ERROR_QUOTA_EXCEEDED),
            (429, ERROR_RATE_LIMITED),
            (500, ERROR_UPSTREAM_ERROR),
            (503, ERROR_UPSTREAM_ERROR),
        ],
    )
    def test_status_mapping(self, status, expected):
        assert classify_status_error(status) == expected

    def test_429_with_quota_hint_is_quota_error(self):
        assert (
            classify_status_error(429, '{"error":"insufficient quota"}')
            == ERROR_QUOTA_EXCEEDED
        )

    def test_query_rejection_is_not_a_credential_error(self):
        assert ERROR_INVALID_QUERY not in COOLDOWN_SECONDS
        assert ERROR_INVALID_QUERY in QUERY_ERROR_KINDS
        assert ERROR_INVALID_QUERY not in RETRYABLE_ERROR_KINDS

    def test_cooldown_durations(self):
        assert COOLDOWN_SECONDS[ERROR_INVALID_KEY] == 24 * 3600
        assert COOLDOWN_SECONDS[ERROR_QUOTA_EXCEEDED] == 6 * 3600
        assert COOLDOWN_SECONDS[ERROR_RATE_LIMITED] == 60.0
        assert COOLDOWN_SECONDS[ERROR_UPSTREAM_ERROR] == 30.0
        assert COOLDOWN_SECONDS[ERROR_NETWORK_ERROR] == 30.0


class TestSearchPoolStore:
    def test_missing_file_loads_empty(self, tmp_path):
        store = SearchPoolStore(tmp_path / "missing.json")
        assert store.load() == []

    def test_round_trip_masks_keys(self, tmp_path):
        path = tmp_path / "pool.json"
        store = SearchPoolStore(path)
        entries = [entry("sp_a", "tavily"), entry("sp_b", "brave", api_key="")]
        store.save(entries)

        loaded = store.load()
        assert [item.id for item in loaded] == ["sp_a", "sp_b"]
        assert loaded[0].api_key == "tavily-key"

        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw["entries"][0]["api_key"] == "tavily-key"
        masked = loaded[0].to_dict(mask_secrets=True)
        assert masked["api_key"] == "***"

    def test_migrate_from_legacy_writes_file(self, tmp_path):
        path = tmp_path / "pool.json"
        store = SearchPoolStore(path)
        entries = store.migrate_from_legacy("tavily", api_key="tvly-legacy")

        assert path.exists()
        assert len(entries) == 1
        assert entries[0].provider == "tavily"
        assert entries[0].api_key == "tvly-legacy"
        assert SearchPoolStore(path).load()[0].api_key == "tvly-legacy"

    def test_migrate_without_provider_is_noop(self, tmp_path):
        path = tmp_path / "pool.json"
        assert SearchPoolStore(path).migrate_from_legacy("") == []
        assert not path.exists()

    def test_broken_file_loads_empty(self, tmp_path):
        path = tmp_path / "pool.json"
        path.write_text("{not json", encoding="utf-8")
        assert SearchPoolStore(path).load() == []

    def test_unusable_legacy_settings_are_not_migrated(self, tmp_path):
        path = tmp_path / "pool.json"
        invalidate_search_pool()
        pool = get_search_pool(str(path), "custom", "", "")

        assert pool.entries == []
        assert not path.exists()
        invalidate_search_pool()

    def test_get_search_pool_seeds_from_legacy_settings(self, tmp_path):
        path = tmp_path / "pool.json"
        invalidate_search_pool()
        pool = get_search_pool(str(path), "tavily", "tvly-legacy", "")

        assert [item.provider for item in pool.entries] == ["tavily"]
        assert path.exists()
        assert load_pool_entries(path)[0].api_key == "tvly-legacy"
        invalidate_search_pool()

    def test_save_pool_entries_invalidates_cache(self, tmp_path):
        path = tmp_path / "pool.json"
        invalidate_search_pool()
        get_search_pool(str(path), "tavily", "tvly-one", "")
        save_pool_entries([entry("sp_new", "brave")], path)
        pool = get_search_pool(str(path), "tavily", "tvly-one", "")
        assert [item.id for item in pool.entries] == ["sp_new"]
        invalidate_search_pool()


class TestSearchProviderPool:
    @pytest.mark.asyncio
    async def test_priority_falls_back_to_next_entry(self, monkeypatch):
        patch_providers(
            monkeypatch,
            {"tavily": err_response(ERROR_INVALID_KEY), "brave": ok_response()},
        )
        pool = SearchProviderPool([entry("sp_a", "tavily"), entry("sp_b", "brave")])

        response = await pool.search("hello")

        assert response.error is None
        assert response.results
        status = {item["id"]: item for item in pool.status()}
        assert status["sp_a"]["state"] == "cooldown"
        assert status["sp_a"]["cooldown_reason"] == ERROR_INVALID_KEY
        assert status["sp_b"]["state"] == "ok"

    @pytest.mark.asyncio
    async def test_cooldown_skips_entry_but_expires(self, monkeypatch):
        created = patch_providers(
            monkeypatch,
            {"tavily": err_response(ERROR_INVALID_KEY), "brave": ok_response()},
        )
        pool = SearchProviderPool([entry("sp_a", "tavily"), entry("sp_b", "brave")])

        first = await pool.search("hello")
        assert first.error is None
        assert created["tavily"].calls == 1

        second = await pool.search("hello again")
        assert second.error is None
        # The cooling entry is skipped entirely.
        assert created["tavily"].calls == 1
        assert created["brave"].calls == 2

        pool.reset_cooldowns("sp_a")
        await pool.search("third")
        assert created["tavily"].calls == 2

    @pytest.mark.asyncio
    async def test_all_cooling_returns_error_without_calling(self, monkeypatch):
        created = patch_providers(monkeypatch, {"tavily": err_response(ERROR_QUOTA_EXCEEDED)})
        pool = SearchProviderPool([entry("sp_a", "tavily")])

        first = await pool.search("hello")
        assert first.error == ERROR_QUOTA_EXCEEDED

        second = await pool.search("again")
        assert second.error == ERROR_QUOTA_EXCEEDED
        assert "冷却" in second.error_detail
        assert created["tavily"].calls == 1
        assert pool.all_cooling() is True

    @pytest.mark.asyncio
    async def test_rejected_query_does_not_cool_down_or_fail_over(self, monkeypatch):
        created = patch_providers(
            monkeypatch,
            {"tavily": err_response(ERROR_INVALID_QUERY), "brave": ok_response()},
        )
        pool = SearchProviderPool([entry("sp_a", "tavily"), entry("sp_b", "brave")])

        response = await pool.search("site:example.com")

        assert response.error == ERROR_INVALID_QUERY
        assert response.results == []
        # the healthy credential was not burnt, and no cooldown was applied
        assert "brave" not in created
        assert pool.status()[0]["state"] == "ok"

    @pytest.mark.asyncio
    async def test_disabled_entries_are_skipped(self, monkeypatch):
        created = patch_providers(
            monkeypatch,
            {"tavily": ok_response(), "brave": ok_response()},
        )
        pool = SearchProviderPool(
            [entry("sp_a", "tavily", enabled=False), entry("sp_b", "brave")]
        )

        response = await pool.search("hello")
        assert response.error is None
        assert "tavily" not in created

    @pytest.mark.asyncio
    async def test_incomplete_config_reports_invalid_key(self, monkeypatch):
        patch_providers(monkeypatch, {})
        pool = SearchProviderPool([entry("sp_a", "tavily")])

        response = await pool.search("hello")
        assert response.error == ERROR_INVALID_KEY
        assert pool.all_cooling() is True

    @pytest.mark.asyncio
    async def test_empty_pool_returns_error(self):
        pool = SearchProviderPool([])
        response = await pool.search("hello")
        assert response.error is not None
        assert response.results == []
        assert pool.all_cooling() is False

    @pytest.mark.asyncio
    async def test_exhausted_attempts_report_first_error(self, monkeypatch):
        patch_providers(
            monkeypatch,
            {
                "tavily": err_response(ERROR_INVALID_KEY),
                "brave": err_response(ERROR_UPSTREAM_ERROR),
            },
        )
        pool = SearchProviderPool([entry("sp_a", "tavily"), entry("sp_b", "brave")])
        response = await pool.search("hello")
        assert response.error == ERROR_INVALID_KEY
        assert response.error_detail == f"{ERROR_INVALID_KEY} from test double"

    @pytest.mark.asyncio
    async def test_test_entry_success_resets_cooldown(self, monkeypatch):
        patch_providers(monkeypatch, {"tavily": ok_response(count=2)})
        pool = SearchProviderPool([entry("sp_a", "tavily")])
        pool._cooldowns["sp_a"] = {"kind": ERROR_UNKNOWN, "detail": "x", "until": 9999999999}

        result = await pool.test_entry("sp_a")

        assert result["ok"] is True
        assert result["result_count"] == 2
        assert pool.status()[0]["state"] == "ok"

    @pytest.mark.asyncio
    async def test_test_entry_unknown_id(self):
        pool = SearchProviderPool([entry("sp_a", "tavily")])
        result = await pool.test_entry("sp_missing")
        assert result["ok"] is False
        assert result["error"] == "not_found"

    def test_status_masks_keys(self):
        pool = SearchProviderPool([entry("sp_a", "tavily", api_key="tvly-secret")])
        status = pool.status()
        assert status[0]["api_key"] == "***"
        assert status[0]["usable"] is True

    def test_entries_property_is_priority_order(self):
        entries = [entry("sp_a", "tavily"), entry("sp_b", "brave")]
        pool = SearchProviderPool(entries)
        assert [item.id for item in pool.entries] == ["sp_a", "sp_b"]
