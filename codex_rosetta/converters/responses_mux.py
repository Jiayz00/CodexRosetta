from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from codex_rosetta.converters.tool_transformer import ToolTransformer
from codex_rosetta.models.common import make_simulated_function_name
from codex_rosetta.utils.id_generation import generate_item_id, generate_response_id

INTERNAL_SEARCH_NAME = make_simulated_function_name("web_search")

SEARCH_TOOL_TYPES = frozenset(
    {"web_search", "web_search_preview", "web_search_2025_08_26"}
)

FORCE_ANSWER_INSTRUCTION = (
    "联网搜索在本轮不可用。不要再调用任何搜索工具，"
    "请直接依据已有知识作答，并简要说明本次未能联网搜索。"
)

SEARCH_RESULTS_NOTE = "以下是已经获取的网络搜索结果，请勿再次调用搜索工具："


def build_search_results_message(entries: list[tuple[str, str]]) -> dict[str, Any]:
    """Carry search results upstream as a plain message.

    Replaying the internal search call as a function call would be more
    faithful, but relay-side Responses->Chat bridges cannot express it: the
    rebuilt assistant tool_calls message trips thinking-mode upstreams
    ("The `reasoning_content` in the thinking mode must be passed back to the
    API") and relays reject a replayed ``tool_call_id`` that collides with an
    earlier round. A message survives every bridge and carries the same text;
    the client still sees standard ``web_search_call`` items.
    """
    blocks: list[str] = []
    for query, body in entries:
        header = f"查询: {query}" if query else ""
        blocks.append("\n".join(part for part in (header, body or "（无结果）") if part))
    return {
        "type": "message",
        "role": "user",
        "content": [{
            "type": "input_text",
            "text": f"{SEARCH_RESULTS_NOTE}\n\n" + "\n\n".join(blocks),
        }],
    }


def build_search_function_tool() -> dict[str, Any]:
    """Responses-shaped function tool that stands in for built-in web_search."""
    definition = ToolTransformer()._get_builtin_function_definition("web_search", {})
    return {"type": "function", **definition}


def _restore_search_history(items: list[Any]) -> list[Any]:
    """Turn replayed web_search_call history into plain message history."""
    restored: list[Any] = []
    for item in items:
        if not isinstance(item, dict) or item.get("type") != "web_search_call":
            restored.append(item)
            continue
        action = item.get("action") or {}
        query = " ".join(str(q) for q in (action.get("queries") or []))
        urls = [
            str(src.get("url"))
            for src in (action.get("sources") or [])
            if isinstance(src, dict) and src.get("url")
        ]
        restored.append(build_search_results_message([
            (query, "\n".join(urls) if urls else "（本次搜索未返回来源）"),
        ]))
    return restored


def normalize_responses_request(
    body: dict[str, Any], *, search_enabled: bool
) -> tuple[dict[str, Any], bool]:
    """Prepare a client Responses request for the upstream hop.

    Built-in search tools become the internal function tool, and replayed
    built-in search items become portable function-call history.
    """
    normalized = dict(body)
    tools = [t for t in (normalized.get("tools") or []) if isinstance(t, dict)]
    has_search = any(t.get("type") in SEARCH_TOOL_TYPES for t in tools)
    kept = [t for t in tools if t.get("type") not in SEARCH_TOOL_TYPES]
    if has_search and search_enabled:
        kept.append(build_search_function_tool())
    if kept:
        normalized["tools"] = kept
    else:
        normalized.pop("tools", None)

    choice = normalized.get("tool_choice")
    if isinstance(choice, dict) and choice.get("type") in SEARCH_TOOL_TYPES:
        normalized["tool_choice"] = {"type": "function", "name": INTERNAL_SEARCH_NAME}

    if isinstance(normalized.get("input"), list):
        normalized["input"] = _restore_search_history(normalized["input"])

    return normalized, has_search


def strip_search_function_tool(body: dict[str, Any], instruction: str) -> dict[str, Any]:
    """Remove the internal search tool and force the model to answer."""
    normalized = dict(body)
    kept = [
        tool
        for tool in (normalized.get("tools") or [])
        if not (
            isinstance(tool, dict)
            and tool.get("type") == "function"
            and tool.get("name") == INTERNAL_SEARCH_NAME
        )
    ]
    if kept:
        normalized["tools"] = kept
    else:
        normalized.pop("tools", None)

    choice = normalized.get("tool_choice")
    if isinstance(choice, dict) and choice.get("name") == INTERNAL_SEARCH_NAME:
        normalized["tool_choice"] = "auto"

    instructions = str(normalized.get("instructions") or "").rstrip()
    normalized["instructions"] = f"{instructions}\n\n{instruction}".strip()
    return normalized


def is_internal_search_item(item: dict[str, Any]) -> bool:
    return item.get("type") == "function_call" and item.get("name") == INTERNAL_SEARCH_NAME


def parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def build_web_search_item(
    item_id: str, arguments: dict[str, Any], sources: list[dict[str, Any]] | None
) -> dict[str, Any]:
    query = str(arguments.get("query") or "")
    return {
        "type": "web_search_call",
        "id": item_id,
        "action": {
            "type": "search",
            "queries": [query] if query else [],
            "sources": list(sources or []),
        },
        "status": "completed",
    }


def to_search_tool_call(call: "SearchCallState") -> dict[str, Any]:
    """Adapt a Responses search call into the Chat shape search helpers expect."""
    return {
        "id": call.call_id,
        "type": "function",
        "function": {"name": INTERNAL_SEARCH_NAME, "arguments": call.arguments or "{}"},
        "_item_id": call.item_id,
    }


@dataclass
class SearchCallState:
    item_id: str
    call_id: str
    output_index: int
    web_search_id: str
    arguments: str = ""
    item_added: bool = False
    announced: bool = False
    searching: bool = False
    completed: bool = False


@dataclass
class RoundResult:
    search_calls: list[SearchCallState] = field(default_factory=list)
    client_calls: list[dict[str, Any]] = field(default_factory=list)
    terminal_event: tuple[str, dict[str, Any]] | None = None


class ResponsesStreamMux:
    """Merge several upstream Responses rounds into one logical SSE response.

    Upstream restarts ``sequence_number`` and ``output_index`` every round and
    hides its own search tool behind an internal function name; this class
    rewrites both counters, converts internal search calls into visible
    ``web_search_call`` items and emits exactly one terminal event.
    """

    def __init__(self) -> None:
        self._seq = 0
        self._next_index = 0
        self._round = 0
        self._index_by_item: dict[str, int] = {}
        self._index_by_pos: dict[tuple[int, int], int] = {}
        self._items: dict[int, dict[str, Any]] = {}
        self._search_by_item: dict[str, SearchCallState] = {}
        self._round_search: list[SearchCallState] = []
        self._round_client: list[dict[str, Any]] = []
        self._last_response: dict[str, Any] | None = None
        self._last_id = ""
        self._terminal: tuple[str, dict[str, Any]] | None = None

    def _emit(self, event_type: str, data: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        event = dict(data)
        event["type"] = event_type
        event["sequence_number"] = self._seq
        self._seq += 1
        return event_type, event

    def _allocate(self, item_id: str, upstream_index: int) -> int:
        if item_id and item_id in self._index_by_item:
            return self._index_by_item[item_id]
        key = (self._round, int(upstream_index))
        index = self._index_by_pos.get(key)
        if index is None:
            index = self._next_index
            self._next_index += 1
            self._index_by_pos[key] = index
        if item_id:
            self._index_by_item[item_id] = index
        return index

    def _rewrite(self, data: dict[str, Any]) -> dict[str, Any]:
        event = dict(data)
        if "output_index" not in event:
            return event
        item_id = str(event.get("item_id") or "")
        upstream_index = int(event.get("output_index") or 0)
        if item_id:
            event["output_index"] = self._allocate(item_id, upstream_index)
        else:
            event["output_index"] = self._allocate("", upstream_index)
        return event

    def _search_state(self, item: dict[str, Any], output_index: int) -> SearchCallState:
        item_id = str(item.get("id") or "")
        state = self._search_by_item.get(item_id)
        if state is None:
            state = SearchCallState(
                item_id=item_id,
                call_id=str(item.get("call_id") or item_id),
                output_index=output_index,
                web_search_id=generate_item_id("web_search"),
                arguments=str(item.get("arguments") or ""),
            )
            self._search_by_item[item_id] = state
            self._round_search.append(state)
        elif item.get("arguments"):
            state.arguments = str(item["arguments"])
        return state

    def _announce_search(self, state: SearchCallState) -> list[tuple[str, dict[str, Any]]]:
        """Emit the in_progress/searching lifecycle once the query is known."""
        events: list[tuple[str, dict[str, Any]]] = []
        args = parse_arguments(state.arguments)
        query = str(args.get("query") or "")
        action = {"type": "search", "queries": [query] if query else [], "sources": []}
        if not state.announced:
            state.announced = True
            events.append(self._emit(
                "response.web_search_call.in_progress",
                {
                    "id": state.web_search_id,
                    "output_index": state.output_index,
                    "status": "in_progress",
                    "action": action,
                },
            ))
        if not state.searching:
            state.searching = True
            events.append(self._emit(
                "response.web_search_call.searching",
                {
                    "id": state.web_search_id,
                    "output_index": state.output_index,
                    "action": action,
                },
            ))
        return events

    def _handle_output_item_added(
        self, data: dict[str, Any]
    ) -> list[tuple[str, dict[str, Any]]]:
        item = data.get("item") or {}
        item_id = str(item.get("id") or "")
        index = self._allocate(item_id, data.get("output_index") or 0)
        if is_internal_search_item(item):
            state = self._search_state(item, index)
            if state.item_added:
                return []
            state.item_added = True
            return [self._emit(
                "response.output_item.added",
                {
                    "output_index": index,
                    "item": {
                        "type": "web_search_call",
                        "id": state.web_search_id,
                        "status": "in_progress",
                        "action": {"type": "search", "queries": [], "sources": []},
                    },
                },
            )]
        self._items[index] = item
        if item.get("type") in {"function_call", "custom_tool_call"}:
            self._round_client.append(item)
        return [self._emit("response.output_item.added", {**data, "output_index": index})]

    def _handle_arguments_delta(
        self, etype: str, data: dict[str, Any]
    ) -> list[tuple[str, dict[str, Any]]]:
        state = self._search_by_item.get(str(data.get("item_id") or ""))
        if state is None:
            return [self._emit(etype, self._rewrite(data))]
        if etype.endswith(".delta"):
            state.arguments += str(data.get("delta") or "")
        elif data.get("arguments"):
            state.arguments = str(data["arguments"])
        return self._announce_search(state)

    def _handle_output_item_done(
        self, data: dict[str, Any]
    ) -> list[tuple[str, dict[str, Any]]]:
        item = data.get("item") or {}
        item_id = str(item.get("id") or "")
        index = self._allocate(item_id, data.get("output_index") or 0)
        if is_internal_search_item(item):
            state = self._search_state(item, index)
            return self._announce_search(state)
        self._items[index] = item
        if item.get("type") in {"function_call", "custom_tool_call"}:
            self._round_client.append(item)
        return [self._emit("response.output_item.done", {**data, "output_index": index})]

    def process_event(
        self, event_type: str | None, data: dict[str, Any], round_index: int
    ) -> list[tuple[str, dict[str, Any]]]:
        self._round = round_index
        etype = str(data.get("type") or event_type or "")

        if etype in {"response.created", "response.in_progress"}:
            if round_index:
                return []
            return [self._emit(etype, data)]

        if etype == "response.output_item.added":
            return self._handle_output_item_added(data)

        if etype in {
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
        }:
            return self._handle_arguments_delta(etype, data)

        if etype == "response.output_item.done":
            return self._handle_output_item_done(data)

        if etype == "response.completed":
            response = data.get("response") or {}
            self._last_response = response
            self._last_id = str(response.get("id") or "")
            for upstream_index, item in enumerate(response.get("output") or []):
                if isinstance(item, dict):
                    index = self._allocate(str(item.get("id") or ""), upstream_index)
                    if is_internal_search_item(item):
                        self._search_state(item, index)
                    else:
                        self._items[index] = item
            return []

        if etype in {"response.failed", "response.incomplete"}:
            if isinstance(data.get("response"), dict):
                self._last_response = data["response"]
                self._last_id = str(data["response"].get("id") or self._last_id)
            terminal = self._emit(etype, self._rewrite(data))
            self._terminal = terminal
            return [terminal]

        return [self._emit(etype, self._rewrite(data))]

    def finalize_round(self) -> RoundResult:
        return RoundResult(
            search_calls=list(self._round_search),
            client_calls=list(self._round_client),
            terminal_event=self._terminal,
        )

    def start_next_round(self) -> None:
        self._round_search.clear()
        self._round_client.clear()
        self._terminal = None

    def complete_search_calls(
        self,
        results_map: dict[str, str],
        sources_map: dict[str, list[dict[str, Any]]],
    ) -> list[tuple[str, dict[str, Any]]]:
        """Close every pending search call with its real sources."""
        events: list[tuple[str, dict[str, Any]]] = []
        for state in self._round_search:
            if state.completed:
                continue
            state.completed = True
            args = parse_arguments(state.arguments)
            query = str(args.get("query") or "")
            sources = list(sources_map.get(state.call_id) or [])
            item = build_web_search_item(state.web_search_id, args, sources)
            self._items[state.output_index] = item
            events.append(self._emit(
                "response.web_search_call.completed",
                {
                    "id": state.web_search_id,
                    "output_index": state.output_index,
                    "status": "completed",
                    "action": {
                        "type": "search",
                        "queries": [query] if query else [],
                        "sources": sources,
                    },
                },
            ))
            events.append(self._emit(
                "response.output_item.done",
                {"output_index": state.output_index, "item": item},
            ))
        return events

    def build_completed_event(self) -> tuple[str, dict[str, Any]]:
        response = dict(self._last_response or {})
        response["id"] = self._last_id or response.get("id") or ""
        response["output"] = [self._items[index] for index in sorted(self._items)]
        event_type = (
            "response.incomplete"
            if response.get("status") == "incomplete"
            else "response.completed"
        )
        return self._emit(event_type, {"response": response})

    def build_failed_event(self, message: str) -> tuple[str, dict[str, Any]]:
        response = dict(self._last_response or {})
        response["id"] = self._last_id or generate_response_id()
        response["status"] = "failed"
        response["error"] = {"code": "stream_error", "message": message}
        response["output"] = [self._items[index] for index in sorted(self._items)]
        return self._emit("response.failed", {"response": response})
