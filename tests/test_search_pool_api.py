from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from codex_rosetta.api import search_pool_router
from codex_rosetta.main import app
from codex_rosetta.search.base import SearchProvider, SearchResponse, SearchResult


class StubProvider(SearchProvider):
    def __init__(self, response: SearchResponse) -> None:
        self.response = response

    async def search(self, query: str, max_results: int = 5) -> SearchResponse:
        return self.response


@pytest.fixture
def pool_file(tmp_path, monkeypatch):
    path = tmp_path / "search_pool.json"
    monkeypatch.setattr(search_pool_router, "_pool_file", lambda: str(path))
    from codex_rosetta.search.pool import invalidate_search_pool

    invalidate_search_pool()
    yield path
    invalidate_search_pool()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_list_is_empty_without_file(client, pool_file):
    response = client.get("/v1/search-pool")
    assert response.status_code == 200
    payload = response.json()
    assert payload["entries"] == []
    assert payload["all_cooling"] is False
    assert payload["pool_file"] == str(pool_file)


def test_create_update_and_mask_key(client, pool_file):
    created = client.post(
        "/v1/search-pool",
        json={"provider": "tavily", "api_key": "tvly-secret", "label": "primary"},
    )
    assert created.status_code == 200
    entries = created.json()["entries"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["provider"] == "tavily"
    assert entry["api_key"] == "***"
    assert entry["state"] == "ok"

    # the raw key is on disk, not in the API response
    raw = json.loads(pool_file.read_text(encoding="utf-8"))
    assert raw["entries"][0]["api_key"] == "tvly-secret"

    # "***" keeps the stored key
    updated = client.put(
        f"/v1/search-pool/{entry['id']}",
        json={"api_key": "***", "label": "renamed", "enabled": False},
    )
    assert updated.status_code == 200
    updated_entry = updated.json()["entries"][0]
    assert updated_entry["label"] == "renamed"
    assert updated_entry["enabled"] is False
    assert updated_entry["state"] == "disabled"
    raw = json.loads(pool_file.read_text(encoding="utf-8"))
    assert raw["entries"][0]["api_key"] == "tvly-secret"

    # an explicit key replaces it
    replaced = client.put(
        f"/v1/search-pool/{entry['id']}", json={"api_key": "tvly-new"}
    )
    assert replaced.status_code == 200
    raw = json.loads(pool_file.read_text(encoding="utf-8"))
    assert raw["entries"][0]["api_key"] == "tvly-new"


def test_create_requires_provider(client, pool_file):
    response = client.post("/v1/search-pool", json={"api_key": "x"})
    assert response.status_code == 400


def test_update_and_delete_unknown_entry(client, pool_file):
    assert client.put("/v1/search-pool/sp_missing", json={}).status_code == 404
    assert client.delete("/v1/search-pool/sp_missing").status_code == 404


def test_reorder_and_delete(client, pool_file):
    ids = []
    for provider in ("tavily", "brave", "searxng"):
        response = client.post(
            "/v1/search-pool", json={"provider": provider, "api_key": "k"}
        )
        ids = [entry["id"] for entry in response.json()["entries"]]

    reordered = client.put(
        "/v1/search-pool/order", json={"ids": [ids[2], ids[0], ids[1]]}
    )
    assert reordered.status_code == 200
    assert [entry["id"] for entry in reordered.json()["entries"]] == [
        ids[2],
        ids[0],
        ids[1],
    ]

    # ids that are not mentioned keep their relative order at the end
    partial = client.put("/v1/search-pool/order", json={"ids": [ids[2]]})
    assert [entry["id"] for entry in partial.json()["entries"]] == [
        ids[2],
        ids[0],
        ids[1],
    ]

    deleted = client.delete(f"/v1/search-pool/{ids[0]}")
    assert deleted.status_code == 200
    assert [entry["id"] for entry in deleted.json()["entries"]] == [ids[2], ids[1]]


def test_test_endpoint_reports_success(client, pool_file, monkeypatch):
    created = client.post(
        "/v1/search-pool", json={"provider": "tavily", "api_key": "tvly-secret"}
    )
    entry_id = created.json()["entries"][0]["id"]

    from codex_rosetta.search import pool as pool_module

    monkeypatch.setattr(
        pool_module,
        "create_provider",
        lambda provider, api_key="", base_url="": StubProvider(
            SearchResponse(
                query="q",
                results=[SearchResult(title="T", url="https://example.com")],
            )
        ),
    )

    response = client.post(f"/v1/search-pool/{entry_id}/test")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["result_count"] == 1
    assert payload["sample"][0]["url"] == "https://example.com"


def test_test_endpoint_reports_failure_and_cooldown(client, pool_file, monkeypatch):
    created = client.post(
        "/v1/search-pool", json={"provider": "tavily", "api_key": "bad"}
    )
    entry_id = created.json()["entries"][0]["id"]

    from codex_rosetta.search import pool as pool_module
    from codex_rosetta.search.base import ERROR_INVALID_KEY

    monkeypatch.setattr(
        pool_module,
        "create_provider",
        lambda provider, api_key="", base_url="": StubProvider(
            SearchResponse(error=ERROR_INVALID_KEY, error_detail="HTTP 401")
        ),
    )

    payload = client.post(f"/v1/search-pool/{entry_id}/test").json()
    assert payload["ok"] is False
    assert payload["error"] == ERROR_INVALID_KEY

    status = client.get("/v1/search-pool").json()
    assert status["entries"][0]["state"] == "cooldown"
    assert status["entries"][0]["cooldown_reason"] == ERROR_INVALID_KEY
    assert status["all_cooling"] is True


def test_legacy_settings_seed_the_pool(client, pool_file):
    from codex_rosetta import config

    monkeypatched = {
        "WEB_SEARCH_PROVIDER": "tavily",
        "WEB_SEARCH_API_KEY": "tvly-legacy",
        "WEB_SEARCH_BASE_URL": "",
    }
    original = config.get_runtime_overrides()
    config.update_settings(monkeypatched)
    try:
        payload = client.get("/v1/search-pool").json()
        assert [entry["provider"] for entry in payload["entries"]] == ["tavily"]
        assert payload["entries"][0]["api_key"] == "***"
        assert pool_file.exists()
    finally:
        config._runtime_overrides.clear()
        config._runtime_overrides.update(original)
