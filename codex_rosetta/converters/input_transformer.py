from __future__ import annotations

import json
from typing import Any

from codex_rosetta.converters.content_transformer import ContentTransformer
from codex_rosetta.models.common import ConversionContext


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
        - function_call_output -> tool message when paired, otherwise user context
        - Namespaced function_call replay -> flattened upstream tool name
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
        pending_reasoning: str = ""

        for item in items:
            if not isinstance(item, dict):
                # Treat as simple string content
                messages.append({"role": "user", "content": str(item)})
                continue

            item_type = item.get("type", "")

            if item_type == "message" or (item_type == "" and "role" in item):
                role = item.get("role", "")
                content = item.get("content")

                if role in ("user", "system", "developer"):
                    # Flush any pending assistant + tool calls
                    if pending_assistant is not None:
                        self._flush_assistant(
                            messages,
                            pending_assistant,
                            pending_tool_calls,
                            reasoning_content=pending_reasoning,
                        )
                        pending_assistant = None
                        pending_tool_calls = []
                        pending_reasoning = ""
                    else:
                        # A reasoning item with no assistant output to attach
                        # to must not leak into a later turn.
                        pending_reasoning = ""

                    mapped_role = "system" if role == "developer" else role
                    converted_content = self._ct.responses_input_to_chat_content(content)
                    msg: dict[str, Any] = {"role": mapped_role, "content": converted_content}
                    if item.get("name"):
                        msg["name"] = item["name"]
                    messages.append(msg)

                elif role == "assistant":
                    converted_content = self._ct.responses_input_to_chat_content(content)
                    if not converted_content:
                        # Codex interleaves empty assistant placeholders
                        # with tool calls. Opening a new turn here splits
                        # the tool calls from their results, which
                        # upstreams reject with "An assistant message with
                        # 'tool_calls' must be followed by tool messages".
                        continue

                    # Flush previous pending assistant
                    if pending_assistant is not None:
                        self._flush_assistant(
                            messages,
                            pending_assistant,
                            pending_tool_calls,
                            reasoning_content=pending_reasoning,
                        )
                        pending_tool_calls = []
                        pending_reasoning = ""

                    pending_assistant = {"role": "assistant", "content": converted_content}

            elif item_type == "function_call":
                # Codex may replay tool calls without a preceding assistant
                # message item. Create one, otherwise the tool calls are
                # dropped and the following `tool` message is orphaned,
                # which upstream APIs reject.
                if pending_assistant is None:
                    pending_assistant = {"role": "assistant", "content": None}
                # Namespaced tools are replayed as a bare tool name plus a
                # `namespace` field; upstream only ever saw the flattened name.
                name = item.get("name", "")
                namespace = item.get("namespace") or ""
                if namespace and context is not None:
                    name = context.flatten_namespace_tool(namespace, name)
                tc: dict[str, Any] = {
                    "id": item.get("call_id", item.get("id", "")),
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": item.get("arguments", "{}"),
                    },
                }
                pending_tool_calls.append(tc)

            elif item_type == "function_call_output":
                call_id = item.get("call_id", "")
                is_orphan = not call_id

                # Flush pending assistant first
                if pending_assistant is not None:
                    self._flush_assistant(
                        messages,
                        pending_assistant,
                        pending_tool_calls,
                        reasoning_content=pending_reasoning,
                    )
                    pending_assistant = None
                    pending_tool_calls = []
                    pending_reasoning = ""

                output = item.get("output", "")
                if isinstance(output, list):
                    output = self._ct.flatten_output_content(output)
                if is_orphan:
                    # Agent-created threads can start with a create_thread result
                    # but no replayed function_call or call_id. It is seed user
                    # context, not a valid chat-completions tool result.
                    messages.append({
                        "role": "user",
                        "content": output if output is not None else "",
                    })
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": output if output is not None else "",
                    })

            elif item_type == "custom_tool_call":
                if pending_assistant is None:
                    pending_assistant = {"role": "assistant", "content": None}
                # Custom tools are exposed upstream as a single-string
                # function, so replay them in that shape.
                tc: dict[str, Any] = {
                    "id": item.get("call_id", item.get("id", "")),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": json.dumps(
                            {"input": item.get("input", "")}, ensure_ascii=False
                        ),
                    },
                }
                pending_tool_calls.append(tc)

            elif item_type == "custom_tool_call_output":
                call_id = item.get("call_id", "")
                is_orphan = not call_id

                if pending_assistant is not None:
                    self._flush_assistant(
                        messages,
                        pending_assistant,
                        pending_tool_calls,
                        reasoning_content=pending_reasoning,
                    )
                    pending_assistant = None
                    pending_tool_calls = []
                    pending_reasoning = ""

                output = item.get("output", "")
                if isinstance(output, list):
                    output = self._ct.flatten_output_content(output)
                if is_orphan:
                    messages.append({
                        "role": "user",
                        "content": output if output is not None else "",
                    })
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": output if output is not None else "",
                    })

            elif item_type == "reasoning":
                # Thinking-mode upstreams (DeepSeek, GLM, ...) require the
                # reasoning text back as `reasoning_content` on the
                # assistant message. Dropping these items made every
                # follow-up turn fail with "The `reasoning_content` in the
                # thinking mode must be passed back to the API."
                reasoning_text = _extract_reasoning_text(item)
                if reasoning_text:
                    pending_reasoning = "\n".join(
                        part for part in (pending_reasoning, reasoning_text) if part
                    )

            elif item_type in (
                "web_search_call",
                "file_search_call",
                "computer_call",
                "code_interpreter_call",
                "image_generation_call",
                "mcp_call",
            ):
                # Built-in tool output items in input context — skip
                # They reference previous built-in tool calls
                pass

        # Flush final pending assistant
        if pending_assistant is not None:
            self._flush_assistant(
                messages,
                pending_assistant,
                pending_tool_calls,
                reasoning_content=pending_reasoning,
            )

        return messages

    def _flush_assistant(
        self,
        messages: list[dict[str, Any]],
        assistant_msg: dict[str, Any],
        tool_calls: list[dict[str, Any]],
        reasoning_content: str = "",
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


def _extract_reasoning_text(item: dict[str, Any]) -> str:
    """Collect the plain text carried by a Responses API `reasoning` item."""
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
