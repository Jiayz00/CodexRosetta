from __future__ import annotations

import json
from typing import Any

from codex_rosetta.converters.content_transformer import ContentTransformer


class InputTransformer:
    """Convert Responses API input array to Chat Completions messages array."""

    def __init__(self, content_transformer: ContentTransformer) -> None:
        self._ct = content_transformer

    def transform_input(
        self,
        input_data: str | list[Any],
        instructions: str | None = None,
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
            assembled = self._assemble_messages_from_items(input_data)
            messages.extend(assembled)

        return messages

    def _assemble_messages_from_items(self, items: list[Any]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        pending_assistant: dict[str, Any] | None = None
        pending_tool_calls: list[dict[str, Any]] = []
        pending_reasoning = ""

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
                    # Reasoning that never reached an assistant turn must
                    # not leak into a later one.
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
                if pending_assistant is None:
                    # A tool call can arrive without a preceding assistant
                    # message item. Without one, the following `tool`
                    # message would be orphaned and rejected upstream.
                    pending_assistant = {"role": "assistant", "content": None}
                tc: dict[str, Any] = {
                    "id": item.get("call_id", item.get("id", "")),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", "{}"),
                    },
                }
                pending_tool_calls.append(tc)

            elif item_type == "function_call_output":
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

                call_id = item.get("call_id", "")
                output = item.get("output", "")
                if isinstance(output, list):
                    output = self._ct.flatten_output_content(output)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": output if output is not None else "",
                })

            elif item_type == "custom_tool_call":
                if pending_assistant is None:
                    pending_assistant = {"role": "assistant", "content": None}
                # Custom tools are advertised to the upstream as functions
                # (see ToolTransformer._convert_custom_tool), so the
                # historical tool call has to use the same shape.
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

                call_id = item.get("call_id", "")
                output = item.get("output", "")
                if isinstance(output, list):
                    output = self._ct.flatten_output_content(output)
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
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
            if assistant_msg.get("content") == "" or assistant_msg.get("content") is None:
                assistant_msg["content"] = None
        if reasoning_content:
            assistant_msg["reasoning_content"] = reasoning_content
        if not tool_calls and not assistant_msg.get("content"):
            # Nothing to say and no tool call to make. Emitting it would sit
            # between a tool_calls message and its tool result.
            return
        messages.append(assistant_msg)


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
