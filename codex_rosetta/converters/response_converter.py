from __future__ import annotations

from typing import Any

from codex_rosetta.converters.content_transformer import ContentTransformer
from codex_rosetta.models.common import (
    EMPTY_ARGUMENTS,
    ConversionContext,
    derive_phase,
    extract_custom_tool_input,
    extract_original_type,
    is_simulated_function,
)
from codex_rosetta.models.responses_api import build_response_envelope
from codex_rosetta.utils.id_generation import generate_item_id, unix_timestamp


class ResponseConverter:
    """Convert Chat Completions responses to Responses API responses."""

    def __init__(self, content_transformer: ContentTransformer | None = None) -> None:
        self._ct = content_transformer or ContentTransformer()

    def convert(
        self,
        chat_response: dict[str, Any],
        context: ConversionContext,
    ) -> dict[str, Any]:
        """Convert a Chat Completions response to a Responses API response."""
        created_at = unix_timestamp()

        # Build output items
        output_items: list[dict[str, Any]] = []
        choices = chat_response.get("choices") or []

        for choice in choices:
            message = choice.get("message") or {}
            output_items.extend(
                self._convert_message_to_output_items(message, context)
            )

        # Convert usage
        usage = self._convert_usage(chat_response.get("usage"))

        # Inject null placeholders for requested include fields
        self._inject_include_placeholders(output_items, context)

        # Truncate tool calls if max_tool_calls is set
        self._truncate_tool_calls(output_items, context)

        error = chat_response.get("error")
        if error:
            status = "failed"
            incomplete_details = None
        else:
            status, incomplete_details = self._status_for(self._finish_reason(choices))

        return build_response_envelope(
            context,
            output=output_items,
            status=status,
            usage=usage,
            error=error,
            incomplete_details=incomplete_details,
            created_at=created_at,
            completed_at=None if status == "failed" else unix_timestamp(),
        )

    @staticmethod
    def _finish_reason(choices: list[dict[str, Any]]) -> str | None:
        for choice in choices:
            if choice.get("finish_reason"):
                return choice["finish_reason"]
        return None

    @staticmethod
    def _status_for(finish_reason: str | None) -> tuple[str, dict[str, Any] | None]:
        if finish_reason == "length":
            return "incomplete", {"reason": "max_output_tokens"}
        if finish_reason == "content_filter":
            return "incomplete", {"reason": "content_filter"}
        return "completed", None

    def _convert_message_to_output_items(
        self,
        message: dict[str, Any],
        context: ConversionContext,
    ) -> list[dict[str, Any]]:
        """Convert a Chat Completions assistant message to Responses API output items.

        A single assistant message with tool_calls becomes:
        - One "message" output item when the turn has text or a refusal
        - Separate "function_call" output items for each tool call
        """
        items: list[dict[str, Any]] = []
        tool_calls = message.get("tool_calls") or []
        content = message.get("content")
        refusal = message.get("refusal")
        reasoning_content = message.get("reasoning_content")

        # Add reasoning item if present (GLM, DeepSeek, etc.)
        if reasoning_content:
            items.append({
                "type": "reasoning",
                "id": generate_item_id("reasoning"),
                "summary": [],
                "encrypted_content": None,
                "status": "completed",
            })

        # Build content parts for the message item
        content_parts: list[dict[str, Any]] = []

        if content:
            content_parts.extend(self._ct.chat_content_to_responses_output(content))

        if refusal:
            refusal_part = self._ct.refusal_to_responses(refusal)
            if refusal_part:
                content_parts.append(refusal_part)

        # Never announce an empty assistant message: tool-call-only turns carry
        # no text, and an empty message item makes clients render a spurious
        # block (it also breaks the Codex client's grouping of consecutive
        # tool calls into one work block).
        if content_parts:
            items.append({
                "type": "message",
                "id": generate_item_id("message"),
                "role": "assistant",
                "status": "completed",
                "content": content_parts,
                "phase": derive_phase(bool(tool_calls)),
            })

        # Create function_call output items
        for tc in tool_calls:
            fc_item = self._convert_tool_call(tc, context)
            if fc_item is not None:
                items.append(fc_item)

        return items

    def _convert_tool_call(
        self,
        tool_call: dict[str, Any],
        context: ConversionContext,
    ) -> dict[str, Any] | None:
        """Convert a Chat Completions tool_call to a Responses API output item."""
        tc_type = tool_call.get("type", "function")
        tc_id = tool_call.get("id", "")
        call_id = tc_id  # Chat Completions uses id; Responses uses call_id

        if tc_type == "function":
            func = tool_call.get("function", {})
            name = func.get("name", "")
            # Normalise an empty/missing value so the client always sees valid JSON.
            arguments = func.get("arguments") or EMPTY_ARGUMENTS

            # Custom (freeform) tools are exposed upstream as single-string
            # functions; restore the `custom_tool_call` shape.
            if name in context.custom_tool_names:
                return {
                    "type": "custom_tool_call",
                    "id": generate_item_id("custom_tool_call"),
                    "call_id": call_id,
                    "name": name,
                    "input": extract_custom_tool_input(arguments),
                    "status": "completed",
                }

            # Check if this maps to a simulated built-in tool
            if is_simulated_function(name):
                return self._convert_simulated_builtin(name, arguments, call_id, context)

            resolved = context.resolve_namespace_tool(name)
            if resolved:
                namespace, client_name = resolved
            else:
                namespace, client_name = None, name
            item: dict[str, Any] = {
                "type": "function_call",
                "id": generate_item_id("function_call"),
                "call_id": call_id,
                "name": client_name,
                "arguments": arguments,
                "status": "completed",
            }
            if namespace:
                item["namespace"] = namespace
            return item

            # Restore the namespace the client declared this tool in
            namespace_pair = context.resolve_namespace_tool(name)
            if namespace_pair:
                item["name"] = namespace_pair[1]
                item["namespace"] = namespace_pair[0]

            return item

        elif tc_type == "custom":
            custom = tool_call.get("custom", {})
            return {
                "type": "custom_tool_call",
                "id": generate_item_id("custom_tool_call"),
                "call_id": call_id,
                "name": custom.get("name", ""),
                "input": custom.get("input", ""),
                "status": "completed",
            }

        return None

    def _convert_simulated_builtin(
        self,
        function_name: str,
        arguments: str,
        call_id: str,
        context: ConversionContext,
    ) -> dict[str, Any]:
        """Convert a simulated built-in tool function call back to its native type."""
        import json

        original_type = extract_original_type(function_name)

        try:
            args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            args = {}

        if original_type in ("web_search", "web_search_2025_08_26"):
            return {
                "type": "web_search_call",
                "id": generate_item_id("web_search"),
                "action": {
                    "type": "search",
                    "queries": [],
                    "sources": [],
                },
                "status": "completed",
            }

        elif original_type == "file_search":
            return {
                "type": "file_search_call",
                "id": generate_item_id("file_search"),
                "queries": [args.get("query", "")],
                "status": "completed",
            }

        elif original_type in ("computer_use_preview", "computer"):
            action = args.get("action", {})
            return {
                "type": "computer_call",
                "id": generate_item_id("computer_call"),
                "call_id": call_id,
                "action": action,
                "pending_safety_checks": [],
                "status": "completed",
            }

        elif original_type == "code_interpreter":
            return {
                "type": "code_interpreter_call",
                "id": generate_item_id("code_interpreter"),
                "code": args.get("code", ""),
                "container_id": None,
                "outputs": [],
                "status": "completed",
            }

        elif original_type == "image_generation":
            return {
                "type": "image_generation_call",
                "id": generate_item_id("image_generation"),
                "result": "",
                "status": "completed",
            }

        # Fallback: treat as regular function call
        return {
            "type": "function_call",
            "id": generate_item_id("function_call"),
            "call_id": call_id,
            "name": function_name,
            "arguments": arguments,
            "status": "completed",
        }

    def _convert_usage(
        self, usage: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Convert Chat Completions usage to Responses API usage."""
        if usage is None:
            return {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            }

        prompt_details = usage.get("prompt_tokens_details") or {}
        completion_details = usage.get("completion_tokens_details") or {}

        return {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "input_tokens_details": {
                "cached_tokens": prompt_details.get("cached_tokens", 0),
            },
            "output_tokens_details": {
                "reasoning_tokens": completion_details.get("reasoning_tokens", 0),
            },
        }

    def _inject_include_placeholders(
        self, output_items: list[dict[str, Any]], context: ConversionContext
    ) -> None:
        """Inject null placeholders for requested include fields."""
        if not context.include_fields:
            return

        for field in context.include_fields:
            if field == "reasoning.encrypted_content":
                # Never fabricate an empty reasoning item — Codex would render a
                # phantom reasoning block. Real items already carry the field.
                for item in output_items:
                    if item.get("type") == "reasoning":
                        item.setdefault("encrypted_content", None)

            elif field == "message.output_text.logprobs":
                for item in output_items:
                    if item.get("type") == "message":
                        for part in item.get("content", []):
                            if part.get("type") == "output_text":
                                part.setdefault("logprobs", None)

    def _truncate_tool_calls(
        self, output_items: list[dict[str, Any]], context: ConversionContext
    ) -> None:
        """Truncate tool call output items when max_tool_calls limit is exceeded."""
        if context.max_tool_calls is None:
            return

        # Count tool-call-like items (function_call, web_search_call, etc.)
        tool_call_types = {
            "function_call", "web_search_call", "file_search_call",
            "computer_call", "code_interpreter_call", "image_generation_call",
            "custom_tool_call", "mcp_call",
        }
        tool_indices = [
            i for i, item in enumerate(output_items)
            if item.get("type") in tool_call_types
        ]

        if len(tool_indices) <= context.max_tool_calls:
            return

        # Remove excess tool call items (keep first max_tool_calls)
        excess_indices = set(tool_indices[context.max_tool_calls:])
        output_items[:] = [
            item for i, item in enumerate(output_items)
            if i not in excess_indices
        ]
