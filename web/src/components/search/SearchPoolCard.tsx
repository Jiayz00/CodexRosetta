import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  AlertCircle,
  ArrowDown,
  ArrowUp,
  CheckCircle2,
  Database,
  Pencil,
  Plus,
  Trash2,
  X,
  Zap,
} from "lucide-react";
import { useSearchPool } from "@/hooks/useSearchPool";
import type {
  SearchPoolEntry,
  SearchPoolEntryInput,
  SearchPoolTestResult,
} from "@/hooks/useSearchPool";

const POOL_PROVIDERS = ["tavily", "brave", "searxng", "duckduckgo", "custom"];

const EMPTY_ENTRY: SearchPoolEntryInput = {
  provider: "tavily",
  api_key: "",
  base_url: "",
  label: "",
};

function formatRemaining(seconds?: number): string {
  const total = Math.max(0, Math.round(seconds ?? 0));
  if (total >= 3600) return `${Math.round(total / 3600)}h`;
  if (total >= 60) return `${Math.round(total / 60)}m`;
  return `${total}s`;
}

function StateBadge({ entry }: { entry: SearchPoolEntry }) {
  if (entry.state === "disabled") {
    return (
      <span className="px-2 py-0.5 rounded-md text-[10px] font-mono bg-rosetta-card text-rosetta-muted border border-rosetta-border">
        DISABLED
      </span>
    );
  }
  if (entry.state === "cooldown") {
    return (
      <span
        className="px-2 py-0.5 rounded-md text-[10px] font-mono bg-rosetta-warning/10 text-rosetta-warning border border-rosetta-warning/20"
        title={entry.cooldown_detail || entry.cooldown_reason || ""}
      >
        COOLDOWN {entry.cooldown_reason} · {formatRemaining(entry.cooldown_remaining_seconds)}
      </span>
    );
  }
  return (
    <span className="px-2 py-0.5 rounded-md text-[10px] font-mono bg-rosetta-success/10 text-rosetta-success border border-rosetta-success/20">
      READY
    </span>
  );
}

export function SearchPoolCard() {
  const { pool, loading, busy, error, addEntry, updateEntry, deleteEntry, reorder, testEntry } =
    useSearchPool();

  const [draft, setDraft] = useState<SearchPoolEntryInput>({ ...EMPTY_ENTRY });
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editDraft, setEditDraft] = useState<SearchPoolEntryInput>({});
  const [testingId, setTestingId] = useState<string | null>(null);
  const [results, setResults] = useState<Record<string, SearchPoolTestResult>>({});

  const entries = pool?.entries ?? [];

  const handleAdd = async () => {
    if (!draft.provider) return;
    try {
      await addEntry(draft);
      setDraft({ ...EMPTY_ENTRY, provider: draft.provider });
    } catch {
      /* error is surfaced by the hook */
    }
  };

  const handleUpdate = async (id: string) => {
    try {
      await updateEntry(id, editDraft);
      setEditingId(null);
    } catch {
      /* error is surfaced by the hook */
    }
  };

  const handleMove = async (index: number, delta: number) => {
    const target = index + delta;
    if (target < 0 || target >= entries.length) return;
    const ids = entries.map((entry) => entry.id);
    [ids[index], ids[target]] = [ids[target], ids[index]];
    try {
      await reorder(ids);
    } catch {
      /* error is surfaced by the hook */
    }
  };

  const handleTest = async (id: string) => {
    setTestingId(id);
    try {
      const result = await testEntry(id);
      setResults((prev) => ({ ...prev, [id]: result }));
    } catch (e: unknown) {
      setResults((prev) => ({
        ...prev,
        [id]: { ok: false, error_detail: e instanceof Error ? e.message : "测试失败" },
      }));
    } finally {
      setTestingId(null);
    }
  };

  return (
    <motion.div
      className="rounded-xl bg-rosetta-surface border border-rosetta-border overflow-hidden"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.25 }}
    >
      <div className="px-6 py-4 border-b border-rosetta-border flex items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="w-7 h-7 rounded-lg bg-rosetta-gold/10 flex items-center justify-center">
            <Database className="w-3.5 h-3.5 text-rosetta-gold" />
          </div>
          <div>
            <h2 className="text-sm font-semibold text-rosetta-text">
              Search Credential Pool
            </h2>
            <p className="text-[11px] text-rosetta-muted font-mono">
              Ordered by priority — a failing key cools down and the next one is used
            </p>
          </div>
        </div>
        {pool && (
          <span className="text-[10px] text-rosetta-muted font-mono">
            {entries.length} entr{entries.length === 1 ? "y" : "ies"} · {pool.pool_file}
          </span>
        )}
      </div>

      {pool?.all_cooling && (
        <div className="px-6 py-3 bg-rosetta-warning/10 border-b border-rosetta-warning/20 flex items-center gap-2">
          <AlertCircle className="w-4 h-4 text-rosetta-warning flex-shrink-0" />
          <span className="text-xs text-rosetta-warning">
            全部搜索凭证处于冷却中，搜索会立即返回「暂不可用」并让模型直接作答。可用 Test 手动恢复。
          </span>
        </div>
      )}

      {(error || pool?.search_enabled === false) && (
        <div className="px-6 py-3 border-b border-rosetta-border flex items-center gap-2">
          <AlertCircle className="w-4 h-4 text-rosetta-error flex-shrink-0" />
          <span className="text-xs text-rosetta-muted">
            {error || "Web Search 当前处于关闭状态，号池不会生效。"}
          </span>
        </div>
      )}

      <div className="divide-y divide-rosetta-border">
        {loading && (
          <div className="px-6 py-4 text-xs text-rosetta-muted font-mono">
            Loading credential pool...
          </div>
        )}

        <AnimatePresence initial={false}>
          {entries.map((entry, index) => {
            const result = results[entry.id];
            const editing = editingId === entry.id;
            return (
              <motion.div
                key={entry.id}
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="overflow-hidden"
              >
                <div className="px-6 py-4 space-y-3">
                  <div className="flex items-center justify-between gap-3">
                    <div className="flex items-center gap-3 min-w-0">
                      <span className="px-2 py-0.5 rounded-md text-[10px] font-mono bg-rosetta-card text-rosetta-gold border border-rosetta-border">
                        #{index + 1} {entry.provider}
                      </span>
                      <div className="min-w-0">
                        <p className="text-sm text-rosetta-text truncate">
                          {entry.label || "(no label)"}
                        </p>
                        <p className="text-[11px] text-rosetta-muted font-mono truncate">
                          {entry.api_key || "(no key)"}
                          {entry.base_url ? ` · ${entry.base_url}` : ""}
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center gap-2 flex-shrink-0">
                      <StateBadge entry={entry} />
                      <button
                        type="button"
                        onClick={() => handleMove(index, -1)}
                        disabled={index === 0 || busy}
                        className="p-1.5 rounded-md border border-rosetta-border text-rosetta-muted hover:text-rosetta-text disabled:opacity-30"
                        title="Move up"
                      >
                        <ArrowUp className="w-3 h-3" />
                      </button>
                      <button
                        type="button"
                        onClick={() => handleMove(index, 1)}
                        disabled={index === entries.length - 1 || busy}
                        className="p-1.5 rounded-md border border-rosetta-border text-rosetta-muted hover:text-rosetta-text disabled:opacity-30"
                        title="Move down"
                      >
                        <ArrowDown className="w-3 h-3" />
                      </button>
                      <button
                        type="button"
                        onClick={() => handleTest(entry.id)}
                        disabled={testingId === entry.id}
                        className="flex items-center gap-1 px-2 py-1 rounded-md border border-rosetta-border text-[11px] text-rosetta-muted hover:text-rosetta-text disabled:opacity-50"
                        title="Run one live search with this credential"
                      >
                        <Zap className="w-3 h-3" />
                        {testingId === entry.id ? "Testing" : "Test"}
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          setEditingId(editing ? null : entry.id);
                          setEditDraft({
                            provider: entry.provider,
                            api_key: "***",
                            base_url: entry.base_url,
                            label: entry.label,
                          });
                        }}
                        className="p-1.5 rounded-md border border-rosetta-border text-rosetta-muted hover:text-rosetta-text"
                        title="Edit"
                      >
                        {editing ? <X className="w-3 h-3" /> : <Pencil className="w-3 h-3" />}
                      </button>
                      <button
                        type="button"
                        onClick={() => updateEntry(entry.id, { enabled: !entry.enabled })}
                        className={`px-2 py-1 rounded-md text-[11px] font-mono border ${
                          entry.enabled
                            ? "border-rosetta-success/30 text-rosetta-success"
                            : "border-rosetta-border text-rosetta-muted"
                        }`}
                        title="Enable / disable"
                      >
                        {entry.enabled ? "ON" : "OFF"}
                      </button>
                      <button
                        type="button"
                        onClick={() => deleteEntry(entry.id)}
                        className="p-1.5 rounded-md border border-rosetta-border text-rosetta-muted hover:text-rosetta-error"
                        title="Delete"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    </div>
                  </div>

                  {result && (
                    <div
                      className={`text-[11px] font-mono flex items-start gap-2 ${
                        result.ok ? "text-rosetta-success" : "text-rosetta-error"
                      }`}
                    >
                      {result.ok ? (
                        <CheckCircle2 className="w-3 h-3 mt-0.5 flex-shrink-0" />
                      ) : (
                        <AlertCircle className="w-3 h-3 mt-0.5 flex-shrink-0" />
                      )}
                      <span className="break-all">
                        {result.ok
                          ? `${result.result_count ?? 0} results in ${result.latency_ms ?? 0}ms${
                              result.sample?.length
                                ? ` · ${result.sample
                                    .map((item) => item.title || item.url)
                                    .join(", ")}`
                                : ""
                            }`
                          : `${result.error || "error"}${result.error_detail ? `: ${result.error_detail}` : ""}`}
                      </span>
                    </div>
                  )}

                  {editing && (
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-2">
                      <select
                        value={editDraft.provider || "tavily"}
                        onChange={(e) => setEditDraft({ ...editDraft, provider: e.target.value })}
                        className="px-3 py-1.5 rounded-lg rosetta-terminal-input text-xs text-rosetta-text"
                      >
                        {POOL_PROVIDERS.map((provider) => (
                          <option key={provider} value={provider}>
                            {provider}
                          </option>
                        ))}
                      </select>
                      <input
                        type="text"
                        value={editDraft.api_key ?? ""}
                        onChange={(e) => setEditDraft({ ...editDraft, api_key: e.target.value })}
                        placeholder="API key (*** = keep)"
                        className="px-3 py-1.5 rounded-lg rosetta-terminal-input text-xs text-rosetta-text"
                      />
                      <input
                        type="text"
                        value={editDraft.base_url ?? ""}
                        onChange={(e) => setEditDraft({ ...editDraft, base_url: e.target.value })}
                        placeholder="Base URL (optional)"
                        className="px-3 py-1.5 rounded-lg rosetta-terminal-input text-xs text-rosetta-text"
                      />
                      <div className="flex items-center gap-2">
                        <input
                          type="text"
                          value={editDraft.label ?? ""}
                          onChange={(e) => setEditDraft({ ...editDraft, label: e.target.value })}
                          placeholder="Label"
                          className="flex-1 px-3 py-1.5 rounded-lg rosetta-terminal-input text-xs text-rosetta-text"
                        />
                        <button
                          type="button"
                          onClick={() => handleUpdate(entry.id)}
                          disabled={busy}
                          className="px-3 py-1.5 rounded-lg bg-rosetta-gold text-rosetta-black text-xs font-medium disabled:opacity-50"
                        >
                          Save
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>

      <div className="px-6 py-4 border-t border-rosetta-border space-y-3">
        <p className="text-[11px] text-rosetta-muted font-mono">
          Add credential — appended at the end of the priority list
        </p>
        <div className="grid grid-cols-1 md:grid-cols-5 gap-2">
          <select
            value={draft.provider || "tavily"}
            onChange={(e) => setDraft({ ...draft, provider: e.target.value })}
            className="px-3 py-1.5 rounded-lg rosetta-terminal-input text-xs text-rosetta-text"
          >
            {POOL_PROVIDERS.map((provider) => (
              <option key={provider} value={provider}>
                {provider}
              </option>
            ))}
          </select>
          <input
            type="text"
            value={draft.api_key ?? ""}
            onChange={(e) => setDraft({ ...draft, api_key: e.target.value })}
            placeholder="API key"
            className="px-3 py-1.5 rounded-lg rosetta-terminal-input text-xs text-rosetta-text"
          />
          <input
            type="text"
            value={draft.base_url ?? ""}
            onChange={(e) => setDraft({ ...draft, base_url: e.target.value })}
            placeholder="Base URL (searxng/custom)"
            className="px-3 py-1.5 rounded-lg rosetta-terminal-input text-xs text-rosetta-text"
          />
          <input
            type="text"
            value={draft.label ?? ""}
            onChange={(e) => setDraft({ ...draft, label: e.target.value })}
            placeholder="Label"
            className="px-3 py-1.5 rounded-lg rosetta-terminal-input text-xs text-rosetta-text"
          />
          <button
            type="button"
            onClick={handleAdd}
            disabled={busy}
            className="flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-lg bg-rosetta-gold text-rosetta-black text-xs font-medium hover:bg-rosetta-gold-light disabled:opacity-50"
          >
            <Plus className="w-3 h-3" />
            Add
          </button>
        </div>
      </div>
    </motion.div>
  );
}
