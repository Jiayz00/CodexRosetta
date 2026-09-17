from __future__ import annotations

import json
from typing import Any

from codex_rosetta.converters.content_transformer import ContentTransformer
from codex_rosetta.models.common import (
    EMPTY_ARGUMENTS,
    ConversionContext,
    UnsupportedParameterError,
    make_simulated_function_name,
    normalize_call_id,
)
from codex_rosetta.utils.logging import get_logger

logger = get_logger("input_transformer")

# Responses input item types this gateway can faithfully represent as Chat messages.
_ALLOWED_ITEM_TYPES = frozenset({
    "message",
    "reasoning",
    "function_call",
    "function_call_output",
    "custom_tool_call",
    "custom_tool_call_output",
    "web_search_call",
})

# Item types that exist upstream-side only, and that we deliberately refuse.
_REJECTED_ITEM_TYPES = frozenset({
    "file_search_call",
    "computer_call",
    "computer_call_output",
    "code_interpreter_call",
    "image_generation_call",
    "mcp_call",
    "mcp_list_tools",
    "mcp_approval_request",
    "mcp_approval_response",
})

_WEB_SEARCH_FUNCTION = make_simulated_function_name("web_search")


class InputTransformer:
    """Convert Responses API input array to Chat Completions messages array."""

    def __init__(self, content_transformer: ContentTransformer) -> None:
        self._ct = content_transformer

    def transform_input(
        self,
        input_data: str | list[Any],
        instructions: str | None = None,
        context: ConversionContext | None = None,
    ) -> list[dict[str, Any]]:
        """Convert Responses API input to Chat Completions messages.

        Handles:
        - Simple string input -> single user message
        - Array of typed items -> flat messages array
        - Grouping assistant messages with adjacent function_call items
        - function_call_output -> tool messages
        - Instructions -> system message prepended
        """
        messages: list[dict[str, Any]] = []

        # Prepend instructions as system message
        if instructions:
            messages.append({"role": "system", "content": instructions})

        # Simple string input
        if isinstance(input_data, str):
            messages.append({"role": "user", "content": input_data})
            return messages

        # Array of items
        if isinstance(input_data, list):
            assembled = self._assemble_messages_from_items(input_data, context)
            messages.extend(assembled)

        return messages

    def _assemble_messages_from_items(
        self, items: list[Any], context: ConversionContext | None = None
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        pending_assistant: dict[str, Any] | None = None
        pending_tool_calls: list[dict[str, Any]] = []
        pending_call_ids: set[str] = set()
        pending_trailing: list[dict[str, Any]] = []
        known_call_ids: set[str] = set()

        def flush_pending() -> None:
            nonlocal pending_assistant, pending_tool_calls
            nonlocal pending_call_ids, pending_trailing
            if pending_assistant is not None or pending_tool_calls:
                assistant = pending_assistant or {"role": "assistant", "content": None}
                self._flush_assistant(
                    messages, assistant, pending_tool_calls, pending_trailing
                )
            pending_assistant = None
            pending_tool_calls = []
            pending_call_ids = set()
            pending_trailing = []

        for index, item in enumerate(items):
            if not isinstance(item, dict):
                # Treat as simple string content
                flush_pending()
                messages.append({"role": "user", "content": str(item)})
                continue

            item_type = item.get("type", "")

            if item_type == "message" or (item_type == "" and "role" in item):
                role = item.get("role", "")
                content = item.get("content")

                if role in ("user", "system", "developer"):
                    flush_pending()

                    mapped_role = "system" if role == "developer" else role
                    converted_content = self._ct.responses_input_to_chat_content(content)
                    msg: dict[str, Any] = {"role": mapped_role, "content": converted_content}
                    if item.get("name"):
                        msg["name"] = item["name"]
                    messages.append(msg)

                elif role == "assistant":
                    flush_pending()

                    if item.get("phase"):
                        # Chat Completions has no equivalent; the phase is derived
                        # from the tool calls when we build the response instead.
                        logger.debug("assistant_phase_ignored", phase=item["phase"])

                    converted_content = self._ct.responses_input_to_chat_content(content)
                    pending_assistant = {"role": "assistant", "content": converted_content}

                else:
                    raise UnsupportedParameterError(
                        f"input[{index}].role",
                        f"Unsupported message role: '{role}'.",
                    )

            elif item_type == "function_call":
                call_id = normalize_call_id(item.get("call_id") or item.get("id"))
                pending_tool_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": self._upstream_name(item, context),
                        # Chat Completions requires a JSON string here; an absent
                        # value means "no arguments" for the model.
                        "arguments": item.get("arguments") or EMPTY_ARGUMENTS,
                    },
                })
                pending_call_ids.add(call_id)
                known_call_ids.add(call_id)

            elif item_type == "custom_tool_call":
                call_id = normalize_call_id(item.get("call_id") or item.get("id"))
                pending_tool_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": json.dumps({"input": item.get("input", "")}),
                    },
                })
                pending_call_ids.add(call_id)
                known_call_ids.add(call_id)

            elif item_type == "web_search_call":
                # Built-in search results are replayed as a simulated function
                # call plus its tool result so the model keeps the context.
                call_id = normalize_call_id(item.get("call_id") or item.get("id"))
                query = self._search_query(item)
                pending_tool_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": _WEB_SEARCH_FUNCTION,
                        "arguments": json.dumps({"query": query}),
                    },
                })
                pending_call_ids.add(call_id)
                known_call_ids.add(call_id)
                pending_trailing.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": self._format_search_call(item),
                })

            elif item_type in ("function_call_output", "custom_tool_call_output"):
                raw_call_id = item.get("call_id")
                call_id = normalize_call_id(raw_call_id)
                output = item.get("output", "")
                if isinstance(output, list):
                    output = self._ct.flatten_output_content(output)
                content = output if output is not None else ""

                if raw_call_id and call_id in pending_call_ids:
                    pending_trailing.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": content,
                    })
                elif raw_call_id and call_id in known_call_ids:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": content,
                    })
                else:
                    # The matching call is not in this input — e.g. an
                    # auto-created child thread replays a delegation result.
                    # Emitting a bare tool message would 400 upstream, so keep
                    # the payload as a user message instead.
                    logger.info(
                        "orphan_tool_output",
                        item_type=item_type,
                        index=index,
                        call_id=call_id,
                    )
                    messages.append({"role": "user", "content": content})

            elif item_type == "reasoning":
                # Reasoning replay is handled via reasoning_content backfill
                # (see input transformer reasoning support); nothing to emit here.
                logger.debug("reasoning_item_skipped", index=index)

            elif item_type in _REJECTED_ITEM_TYPES:
                raise UnsupportedParameterError(
                    f"input[{index}].type",
                    f"Input item type '{item_type}' is not supported by this gateway.",
                )

            elif item_type not in _ALLOWED_ITEM_TYPES:
                raise UnsupportedParameterError(
                    f"input[{index}].type",
                    f"Unsupported input item type: '{item_type or 'unknown'}'.",
                )

        # Flush final pending assistant
        flush_pending()

        return messages

    @staticmethod
    def _upstream_name(
        item: dict[str, Any], context: ConversionContext | None
    ) -> str:
        """Map a client-visible function name to the flattened upstream name."""
        name = item.get("name", "")
        namespace = item.get("namespace")
        if namespace:
            return f"{namespace}__{name}"
        if context is not None:
            return context.resolve_forward_name(name)
        return name

    @staticmethod
    def _search_query(item: dict[str, Any]) -> str:
        action = item.get("action")
        if not isinstance(action, dict):
            return ""
        query = action.get("query")
        if isinstance(query, str) and query:
            return query
        queries = action.get("queries")
        if isinstance(queries, list) and queries:
            return str(queries[0])
        return ""

    @staticmethod
    def _format_search_call(item: dict[str, Any]) -> str:
        """Render a replayed web_search_call result as tool output text."""
        action = item.get("action")
        if not isinstance(action, dict):
            return "Web search completed."
        sources = action.get("sources")
        if isinstance(sources, list) and sources:
            urls = []
            for source in sources:
                if isinstance(source, dict):
                    url = source.get("url")
                else:
                    url = None
                if url:
                    urls.append(str(url))
            if urls:
                return "Web search results:\n" + "\n".join(urls)
        return "Web search completed."

    def _flush_assistant(
        self,
        messages: list[dict[str, Any]],
        assistant_msg: dict[str, Any],
        tool_calls: list[dict[str, Any]],
        trailing: list[dict[str, Any]] | None = None,
    ) -> None:
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
            if assistant_msg.get("content") == "" or assistant_msg.get("content") is None:
                assistant_msg["content"] = None
        messages.append(assistant_msg)
        if trailing:
            messages.extend(trailing)