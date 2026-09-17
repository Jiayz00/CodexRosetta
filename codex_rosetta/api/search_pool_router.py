"""Search credential pool management API.

Entries are ordered by priority: the first usable credential is tried first,
failures put an entry into a cooldown and the next one is attempted.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from codex_rosetta.config import get_settings
from codex_rosetta.search.pool import (
    DEFAULT_POOL_FILE,
    SearchPoolEntry,
    SearchProviderPool,
    get_search_pool,
    load_pool_entries,
    save_pool_entries,
)

router = APIRouter(prefix="/v1/search-pool", tags=["search-pool"])

_MASKED_KEY = "***"


def _pool_file() -> str:
    return getattr(get_settings(), "SEARCH_POOL_FILE", DEFAULT_POOL_FILE)


def _pool() -> SearchProviderPool:
    settings = get_settings()
    return get_search_pool(
        _pool_file(),
        settings.WEB_SEARCH_PROVIDER,
        settings.WEB_SEARCH_API_KEY,
        settings.WEB_SEARCH_BASE_URL,
    )


def _snapshot() -> dict[str, Any]:
    pool = _pool()
    settings = get_settings()
    return {
        "entries": pool.status(),
        "all_cooling": pool.all_cooling(),
        "search_enabled": settings.WEB_SEARCH_ENABLED,
        "provider": settings.WEB_SEARCH_PROVIDER,
        "pool_file": _pool_file(),
    }


class SearchPoolEntryPayload(BaseModel):
    provider: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    label: str | None = None
    enabled: bool | None = None


class SearchPoolOrderPayload(BaseModel):
    ids: list[str]


@router.get("")
async def list_search_pool() -> dict[str, Any]:
    """List pool entries with their live cooldown state (keys are masked)."""
    return _snapshot()


@router.post("")
async def create_search_pool_entry(payload: SearchPoolEntryPayload) -> dict[str, Any]:
    provider = (payload.provider or "").strip().lower()
    if not provider:
        raise HTTPException(status_code=400, detail="provider 不能为空")

    entries = load_pool_entries(_pool_file())
    entries.append(
        SearchPoolEntry.create(
            provider=provider,
            api_key=payload.api_key or "",
            base_url=payload.base_url or "",
            label=payload.label or "",
            enabled=True if payload.enabled is None else payload.enabled,
        )
    )
    save_pool_entries(entries, _pool_file())
    return _snapshot()


@router.put("/order")
async def reorder_search_pool(payload: SearchPoolOrderPayload) -> dict[str, Any]:
    entries = load_pool_entries(_pool_file())
    by_id = {entry.id: entry for entry in entries}

    ordered: list[SearchPoolEntry] = []
    seen: set[str] = set()
    for entry_id in payload.ids:
        entry = by_id.get(entry_id)
        if entry is not None and entry_id not in seen:
            ordered.append(entry)
            seen.add(entry_id)
    ordered.extend(entry for entry in entries if entry.id not in seen)

    save_pool_entries(ordered, _pool_file())
    return _snapshot()


@router.put("/{entry_id}")
async def update_search_pool_entry(
    entry_id: str, payload: SearchPoolEntryPayload
) -> dict[str, Any]:
    entries = load_pool_entries(_pool_file())
    entry = next((item for item in entries if item.id == entry_id), None)
    if entry is None:
        raise HTTPException(status_code=404, detail="凭证不存在")

    if payload.provider is not None and payload.provider.strip():
        entry.provider = payload.provider.strip().lower()
    # "***" means "keep the stored key", mirroring the settings endpoint.
    if payload.api_key is not None and payload.api_key != _MASKED_KEY:
        entry.api_key = payload.api_key
    if payload.base_url is not None:
        entry.base_url = payload.base_url
    if payload.label is not None:
        entry.label = payload.label
    if payload.enabled is not None:
        entry.enabled = payload.enabled

    save_pool_entries(entries, _pool_file())
    return _snapshot()


@router.delete("/{entry_id}")
async def delete_search_pool_entry(entry_id: str) -> dict[str, Any]:
    entries = load_pool_entries(_pool_file())
    remaining = [entry for entry in entries if entry.id != entry_id]
    if len(remaining) == len(entries):
        raise HTTPException(status_code=404, detail="凭证不存在")

    save_pool_entries(remaining, _pool_file())
    _pool().reset_cooldowns(entry_id)
    return _snapshot()


@router.post("/{entry_id}/test")
async def test_search_pool_entry(entry_id: str) -> dict[str, Any]:
    """Run one live search with the entry and clear its cooldown on success."""
    return await _pool().test_entry(entry_id)
