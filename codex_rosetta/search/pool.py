from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codex_rosetta.search.base import (
    ERROR_INVALID_KEY,
    ERROR_NETWORK_ERROR,
    ERROR_QUOTA_EXCEEDED,
    ERROR_RATE_LIMITED,
    ERROR_UNKNOWN,
    ERROR_UPSTREAM_ERROR,
    QUERY_ERROR_KINDS,
    SearchProvider,
    SearchResponse,
)
from codex_rosetta.search.factory import create_provider, is_provider_configured
from codex_rosetta.utils.logging import get_logger

logger = get_logger("search")

DEFAULT_POOL_FILE = "data/search_pool.json"

# How long a credential is skipped after a failure, per error kind.
COOLDOWN_SECONDS: dict[str, float] = {
    ERROR_INVALID_KEY: 24 * 3600,
    ERROR_QUOTA_EXCEEDED: 6 * 3600,
    ERROR_RATE_LIMITED: 60.0,
    ERROR_UPSTREAM_ERROR: 30.0,
    ERROR_NETWORK_ERROR: 30.0,
    ERROR_UNKNOWN: 60.0,
}

_MASKED_KEY = "***"


@dataclass
class SearchPoolEntry:
    """One credential in the search pool."""

    id: str
    provider: str
    api_key: str = ""
    base_url: str = ""
    label: str = ""
    enabled: bool = True

    @classmethod
    def create(
        cls,
        provider: str,
        api_key: str = "",
        base_url: str = "",
        label: str = "",
        enabled: bool = True,
    ) -> "SearchPoolEntry":
        return cls(
            id=f"sp_{uuid.uuid4().hex[:10]}",
            provider=(provider or "").strip().lower(),
            api_key=api_key or "",
            base_url=base_url or "",
            label=label or "",
            enabled=bool(enabled),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchPoolEntry | None":
        if not isinstance(data, dict):
            return None
        provider = str(data.get("provider") or "").strip().lower()
        if not provider:
            return None
        entry_id = str(data.get("id") or "").strip() or f"sp_{uuid.uuid4().hex[:10]}"
        return cls(
            id=entry_id,
            provider=provider,
            api_key=str(data.get("api_key") or ""),
            base_url=str(data.get("base_url") or ""),
            label=str(data.get("label") or ""),
            enabled=bool(data.get("enabled", True)),
        )

    def to_dict(self, mask_secrets: bool = True) -> dict[str, Any]:
        return {
            "id": self.id,
            "provider": self.provider,
            "api_key": _MASKED_KEY if (mask_secrets and self.api_key) else self.api_key,
            "base_url": self.base_url,
            "label": self.label,
            "enabled": self.enabled,
        }


class SearchPoolStore:
    """Load/save the pool file; migrate the legacy single-credential config."""

    def __init__(self, path: str | Path = DEFAULT_POOL_FILE) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> list[SearchPoolEntry]:
        if not self._path.exists():
            return []
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("search_pool_file_unreadable", path=str(self._path), error=str(e))
            return []
        entries_raw = raw.get("entries") if isinstance(raw, dict) else raw
        if not isinstance(entries_raw, list):
            return []
        entries = []
        for item in entries_raw:
            entry = SearchPoolEntry.from_dict(item)
            if entry is not None:
                entries.append(entry)
        return entries

    def save(self, entries: list[SearchPoolEntry]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"entries": [entry.to_dict(mask_secrets=False) for entry in entries]}
        tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_path, self._path)

    def migrate_from_legacy(
        self,
        provider: str,
        api_key: str = "",
        base_url: str = "",
    ) -> list[SearchPoolEntry]:
        """Seed the pool from the legacy single-credential settings."""
        if not provider:
            return []
        entry = SearchPoolEntry.create(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            label="legacy-settings",
        )
        entries = [entry]
        try:
            self.save(entries)
            logger.info("search_pool_migrated", path=str(self._path), provider=provider)
        except OSError as e:
            logger.warning("search_pool_migration_failed", path=str(self._path), error=str(e))
        return entries


class SearchProviderPool(SearchProvider):
    """Priority-ordered credentials with per-entry failure cooldowns."""

    def __init__(self, entries: list[SearchPoolEntry]) -> None:
        self._entries = list(entries)
        self._cooldowns: dict[str, dict[str, Any]] = {}

    # ---- introspection -------------------------------------------------
    @property
    def entries(self) -> list[SearchPoolEntry]:
        return list(self._entries)

    def _cooldown(self, entry_id: str) -> dict[str, Any] | None:
        state = self._cooldowns.get(entry_id)
        if not state:
            return None
        if state["until"] <= time.time():
            self._cooldowns.pop(entry_id, None)
            return None
        return state

    def is_available(self, entry: SearchPoolEntry) -> bool:
        return entry.enabled and self._cooldown(entry.id) is None

    def status(self) -> list[dict[str, Any]]:
        now = time.time()
        result = []
        for entry in self._entries:
            cooldown = self._cooldown(entry.id)
            state = "ok"
            if not entry.enabled:
                state = "disabled"
            elif cooldown:
                state = "cooldown"
            result.append({
                **entry.to_dict(mask_secrets=True),
                "state": state,
                "cooldown_reason": (cooldown or {}).get("kind"),
                "cooldown_detail": (cooldown or {}).get("detail", ""),
                "cooldown_until": (cooldown or {}).get("until"),
                "cooldown_remaining_seconds": (
                    max(0, int((cooldown or {}).get("until", now) - now)) if cooldown else 0
                ),
                "usable": self.is_available(entry),
            })
        return result

    def all_cooling(self) -> bool:
        return bool(self._entries) and not any(
            self.is_available(entry) for entry in self._entries
        )

    def reset_cooldowns(self, entry_id: str | None = None) -> None:
        if entry_id is None:
            self._cooldowns.clear()
        else:
            self._cooldowns.pop(entry_id, None)

    # ---- execution -----------------------------------------------------
    def _mark_failure(self, entry: SearchPoolEntry, kind: str, detail: str) -> None:
        seconds = COOLDOWN_SECONDS.get(kind, COOLDOWN_SECONDS[ERROR_UNKNOWN])
        self._cooldowns[entry.id] = {
            "kind": kind,
            "detail": detail,
            "until": time.time() + seconds,
        }
        logger.warning(
            "search_pool_entry_cooldown",
            entry_id=entry.id,
            provider=entry.provider,
            error_kind=kind,
            cooldown_seconds=seconds,
        )

    async def _search_with_entry(
        self,
        entry: SearchPoolEntry,
        query: str,
        max_results: int,
    ) -> SearchResponse:
        provider = create_provider(entry.provider, entry.api_key, entry.base_url)
        if provider is None:
            return SearchResponse(
                query=query,
                error=ERROR_INVALID_KEY,
                error_detail=f"{entry.provider} 配置不完整（缺少 api_key 或 base_url）",
            )
        try:
            return await provider.search(query, max_results=max_results)
        finally:
            try:
                await provider.close()
            except Exception:  # pragma: no cover - defensive
                pass

    async def search(self, query: str, max_results: int = 5) -> SearchResponse:
        attempts: list[tuple[SearchPoolEntry, SearchResponse]] = []

        for entry in self._entries:
            if not entry.enabled:
                continue
            if self._cooldown(entry.id) is not None:
                continue

            response = await self._search_with_entry(entry, query, max_results)
            if response.error is None:
                logger.info(
                    "search_pool_hit",
                    entry_id=entry.id,
                    provider=entry.provider,
                    result_count=len(response.results),
                )
                return response

            if response.error in QUERY_ERROR_KINDS:
                # The query was rejected: the credential is healthy, so there
                # is nothing to cool down and no point trying the next entry.
                logger.warning(
                    "search_pool_query_rejected",
                    entry_id=entry.id,
                    provider=entry.provider,
                    detail=response.error_detail,
                )
                return response

            logger.warning(
                "search_pool_entry_failed",
                entry_id=entry.id,
                provider=entry.provider,
                error_kind=response.error,
            )
            self._mark_failure(entry, response.error, response.error_detail)
            attempts.append((entry, response))

        if attempts:
            first_error = attempts[0][1]
            logger.warning(
                "search_pool_exhausted",
                attempts=len(attempts),
                error_kind=first_error.error,
            )
            return SearchResponse(
                query=query,
                error=first_error.error,
                error_detail=first_error.error_detail,
            )

        # Everything is either disabled or cooling down.
        cooling = [
            (entry, self._cooldown(entry.id)) for entry in self._entries if entry.enabled
        ]
        cooling = [(entry, state) for entry, state in cooling if state]
        if cooling:
            entry, state = min(cooling, key=lambda pair: pair[1]["until"])
            logger.warning(
                "search_pool_all_cooling",
                entries=len(cooling),
                next_entry_id=entry.id,
            )
            return SearchResponse(
                query=query,
                error=state["kind"],
                error_detail=f"全部搜索凭证处于冷却中（下一条 {entry.id} 可用时间 "
                             f"{int(state['until'] - time.time())} 秒后）",
            )

        logger.warning("search_pool_empty")
        return SearchResponse(
            query=query,
            error=ERROR_QUOTA_EXCEEDED,
            error_detail="没有可用的搜索凭证",
        )

    async def test_entry(
        self,
        entry_id: str,
        query: str = "codex rosetta connectivity test",
        max_results: int = 3,
    ) -> dict[str, Any]:
        entry = next((e for e in self._entries if e.id == entry_id), None)
        if entry is None:
            return {"ok": False, "error": "not_found", "error_detail": "凭证不存在"}

        started = time.monotonic()
        response = await self._search_with_entry(entry, query, max_results)
        latency_ms = round((time.monotonic() - started) * 1000)

        if response.error is None:
            self.reset_cooldowns(entry.id)
            return {
                "ok": True,
                "latency_ms": latency_ms,
                "result_count": len(response.results),
                "sample": [
                    {"title": r.title, "url": r.url} for r in response.results[:3]
                ],
            }

        self._mark_failure(entry, response.error, response.error_detail)
        return {
            "ok": False,
            "latency_ms": latency_ms,
            "error": response.error,
            "error_detail": response.error_detail,
        }


# ---- process-wide cache -------------------------------------------------
_pool_cache: SearchProviderPool | None = None
_pool_signature: tuple[Any, ...] | None = None


def _file_signature(path: str | Path) -> tuple[Any, ...]:
    file_path = Path(path)
    try:
        stat = file_path.stat()
        return (str(file_path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        return (str(file_path), None, None)


def get_search_pool(
    path: str | Path = DEFAULT_POOL_FILE,
    legacy_provider: str = "",
    legacy_api_key: str = "",
    legacy_base_url: str = "",
    *,
    reload: bool = False,
) -> SearchProviderPool:
    """Return the process-wide pool, rebuilding it when the file changes."""
    global _pool_cache, _pool_signature

    signature = (
        _file_signature(path),
        legacy_provider,
        legacy_api_key,
        legacy_base_url,
    )
    if _pool_cache is not None and _pool_signature == signature and not reload:
        return _pool_cache

    store = SearchPoolStore(path)
    entries = store.load()
    if (
        not entries
        and legacy_provider
        and is_provider_configured(legacy_provider, legacy_api_key, legacy_base_url)
    ):
        entries = store.migrate_from_legacy(
            legacy_provider,
            api_key=legacy_api_key,
            base_url=legacy_base_url,
        )

    _pool_cache = SearchProviderPool(entries)
    _pool_signature = signature
    return _pool_cache


def invalidate_search_pool() -> None:
    global _pool_cache, _pool_signature
    _pool_cache = None
    _pool_signature = None


def load_pool_entries(path: str | Path = DEFAULT_POOL_FILE) -> list[SearchPoolEntry]:
    return SearchPoolStore(path).load()


def save_pool_entries(entries: list[SearchPoolEntry], path: str | Path = DEFAULT_POOL_FILE) -> None:
    SearchPoolStore(path).save(entries)
    invalidate_search_pool()
