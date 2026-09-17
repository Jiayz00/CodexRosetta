import { useCallback, useEffect, useState } from "react";

export interface SearchPoolEntry {
  id: string;
  provider: string;
  api_key: string;
  base_url: string;
  label: string;
  enabled: boolean;
  state: "ok" | "cooldown" | "disabled";
  cooldown_reason?: string | null;
  cooldown_detail?: string;
  cooldown_until?: number | null;
  cooldown_remaining_seconds?: number;
  usable: boolean;
}

export interface SearchPoolSnapshot {
  entries: SearchPoolEntry[];
  all_cooling: boolean;
  search_enabled: boolean;
  provider: string;
  pool_file: string;
}

export interface SearchPoolEntryInput {
  provider?: string;
  api_key?: string;
  base_url?: string;
  label?: string;
  enabled?: boolean;
}

export interface SearchPoolTestResult {
  ok: boolean;
  latency_ms?: number;
  result_count?: number;
  sample?: { title: string; url: string }[];
  error?: string;
  error_detail?: string;
}

async function poolRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/v1/search-pool${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error((data as { detail?: string }).detail || "搜索号池请求失败");
  }
  return data as T;
}

export function useSearchPool() {
  const [pool, setPool] = useState<SearchPoolSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setPool(await poolRequest<SearchPoolSnapshot>(""));
      setError(null);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const run = useCallback(
    async (path: string, init: RequestInit) => {
      setBusy(true);
      setError(null);
      try {
        setPool(await poolRequest<SearchPoolSnapshot>(path, init));
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : "Unknown error");
        throw e;
      } finally {
        setBusy(false);
      }
    },
    []
  );

  const addEntry = useCallback(
    (entry: SearchPoolEntryInput) =>
      run("", { method: "POST", body: JSON.stringify(entry) }),
    [run]
  );

  const updateEntry = useCallback(
    (id: string, entry: SearchPoolEntryInput) =>
      run(`/${id}`, { method: "PUT", body: JSON.stringify(entry) }),
    [run]
  );

  const deleteEntry = useCallback(
    (id: string) => run(`/${id}`, { method: "DELETE" }),
    [run]
  );

  const reorder = useCallback(
    (ids: string[]) =>
      run("/order", { method: "PUT", body: JSON.stringify({ ids }) }),
    [run]
  );

  const testEntry = useCallback(async (id: string) => {
    const result = await poolRequest<SearchPoolTestResult>(`/${id}/test`, {
      method: "POST",
    });
    return result;
  }, []);

  return {
    pool,
    loading,
    busy,
    error,
    refresh,
    addEntry,
    updateEntry,
    deleteEntry,
    reorder,
    testEntry,
  };
}
