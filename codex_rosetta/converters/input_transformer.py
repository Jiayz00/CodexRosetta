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


def sanitize_tool_call_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop assistant tool calls that have no matching tool result.

    Codex can truncate or reorder replay input (for example after an interrupted
    turn). Chat Completions requires every assistant tool_call to be followed by
    a tool message, so unmatched calls must be removed before forwarding.
    """
    sanitized: list[dict[str, Any]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
        if message.get("role") != "assistant" or not tool_calls:
            if message.get("role") == "tool":
                # Preserve orphan replay data as user context instead of sending
                # an invalid tool message to the upstream provider.
                sanitized.append({
                    "role": "user",
                    "content": message.get("content", ""),
                })
            else:
                sanitized.append(message)
            index += 1
            continue

        next_index = index + 1
        tool_messages: list[dict[str, Any]] = []
        while next_index < len(messages) and messages[next_index].get("role") == "tool":
            tool_messages.append(messages[next_index])
            next_index += 1

        available_ids = {
            tool_message.get("tool_call_id") for tool_message in tool_messages
        }
        kept_calls = [
            tool_call
            for tool_call in tool_calls
            if tool_call.get("id") in available_ids
        ]
        if kept_calls:
            kept_message = dict(message)
            kept_message["tool_calls"] = kept_calls
            sanitized.append(kept_message)
            kept_ids = {tool_call.get("id") for tool_call in kept_calls}
            sanitized.extend(
                tool_message
                for tool_message in tool_messages
                if tool_message.get("tool_call_id") in kept_ids
            )
        elif message.get("content") not in (None, ""):
            kept_message = dict(message)
            kept_message.pop("tool_calls", None)
            sanitized.append(kept_message)

        index = next_index

    return sanitized


# Placeholder used when a tool-call turn has no upstream reasoning text to
# replay; see InputTransformer._flush_assistant. Kept constant so the
# replayed history stays byte-stable across turns (prompt-cache friendly).
MISSING_REASONING_PLACEHOLDER = "(reasoning unavailable)"

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
        - function_call_output -> tool message, or user context when orphaned
        - reasoning items -> reasoning_content on the matching assistant turn
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

        return sanitize_tool_call_messages(messages)

    def _assemble_messages_from_items(
        self, items: list[Any], context: ConversionContext | None = None
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        pending_assistant: dict[str, Any] | None = None
        pending_tool_calls: list[dict[str, Any]] = []
        pending_call_ids: set[str] = set()
        pending_trailing: list[dict[str, Any]] = []
        known_call_ids: set[str] = set()
        pending_reasoning: str = ""

        def flush_pending() -> None:
            nonlocal pending_assistant, pending_tool_calls
            nonlocal pending_call_ids, pending_trailing, pending_reasoning
            if pending_assistant is not None or pending_tool_calls:
                assistant = pending_assistant or {"role": "assistant", "content": None}
                self._flush_assistant(
                    messages,
                    assistant,
                    pending_tool_calls,
                    reasoning_content=pending_reasoning,
                    trailing=pending_trailing,
                )
            pending_assistant = None
            pending_tool_calls = []
            pending_call_ids = set()
            pending_trailing = []
            pending_reasoning = ""

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
                    converted_content = self._ct.responses_input_to_chat_content(content)
                    if not converted_content:
                        # Codex interleaves empty assistant placeholders with
                        # tool calls. Flushing here would split the tool calls
                        # from their results, which upstreams reject with "An
                        # assistant message with 'tool_calls' must be followed
                        # by tool messages responding to each 'tool_call_id'".
                        continue

                    # Reasoning that preceded this message belongs to it, so
                    # keep it across the flush of any previous turn.
                    carried_reasoning = pending_reasoning
                    flush_pending()
                    pending_reasoning = carried_reasoning

                    if item.get("phase"):
                        # Chat Completions has no equivalent; the phase is derived
                        # from the tool calls when we build the response instead.
                        logger.debug("assistant_phase_ignored", phase=item["phase"])

                    pending_assistant = {"role": "assistant", "content": converted_content}

                else:
                    raise UnsupportedParameterError(
                        f"input[{index}].role",
                        f"Unsupported message role: '{role}'.",
                    )

            elif item_type == "function_call":
                if pending_assistant is None:
                    pending_assistant = {"role": "assistant", "content": None}
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
                if pending_assistant is None:
                    pending_assistant = {"role": "assistant", "content": None}
                call_id = normalize_call_id(item.get("call_id") or item.get("id"))
                pending_tool_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": json.dumps(
                            {"input": item.get("input", "")}, ensure_ascii=False
                        ),
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
                # Thinking-mode upstreams (DeepSeek, GLM, ...) require the
                # reasoning text back as `reasoning_content` on the matching
                # assistant message. Dropping these items made every follow-up
                # turn fail with "The `reasoning_content` in the thinking mode
                # must be passed back to the API."
                reasoning_text = _extract_reasoning_text(item)
                if reasoning_text:
                    pending_reasoning = "\n".join(
                        part for part in (pending_reasoning, reasoning_text) if part
                    )

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
        namespace = item.get("namespace") or ""
        if namespace and context is not None:
            return context.flatten_namespace_tool(namespace, name)
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
        reasoning_content: str = "",
        trailing: list[dict[str, Any]] | None = None,
    ) -> None:
        """Emit a pending assistant message in Chat Completions shape.

        Thinking-mode upstreams (DeepSeek et al.) reject a history where an
        assistant message carries `tool_calls` but no `reasoning_content`:
        ``400 The `reasoning_content` in the thinking mode must be passed back
        to the API``. Not every upstream streams reasoning, so some turns have
        none to replay; a fixed placeholder keeps the history acceptable. The
        placeholder only satisfies that protocol check -- it is not part of the
        model-visible context and is not billed.
        """
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
            if assistant_msg.get("content") == "" or assistant_msg.get("content") is None:
                assistant_msg["content"] = None
        if reasoning_content:
            assistant_msg["reasoning_content"] = reasoning_content
        elif tool_calls:
            assistant_msg["reasoning_content"] = MISSING_REASONING_PLACEHOLDER
        if not tool_calls and not assistant_msg.get("content"):
            # Nothing to say and no tool call to make: emitting it would sit
            # between a tool_calls message and its tool result.
            return
        messages.append(assistant_msg)
        if trailing:
            messages.extend(trailing)


def _extract_reasoning_text(item: dict[str, Any]) -> str:
    """Collect the plain text carried by a Responses API ``reasoning`` item."""
    parts: list[str] = []
    for field in ("summary", "content"):
        entries = item.get(field)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict):
                text = entry.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
            elif isinstance(entry, str) and entry:
                parts.append(entry)
    return "\n".join(parts)