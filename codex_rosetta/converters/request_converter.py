from __future__ import annotations

import json
from typing import Any

from codex_rosetta.converters.content_transformer import ContentTransformer
from codex_rosetta.converters.input_transformer import (
    InputTransformer,
    sanitize_tool_call_messages,
)
from codex_rosetta.converters.tool_transformer import ToolTransformer
from codex_rosetta.models.common import ConversionContext, UnsupportedParameterError
from codex_rosetta.utils.id_generation import generate_response_id
from codex_rosetta.utils.logging import get_logger

logger = get_logger("request_converter")

# Responses API fields that map onto a Chat Completions field (handled explicitly below).
_MAPPED_FIELDS = frozenset({
    "model",
    "input",
    "instructions",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "metadata",
    "service_tier",
    "store",
    "stream",
    "temperature",
    "top_p",
    "top_logprobs",
    "logprobs",
    "user",
    "safety_identifier",
    "prompt_cache_key",
    "reasoning",
    "text",
    # Chat Completions native sampling knobs some clients send next to Responses params.
    "seed",
    "stop",
    "frequency_penalty",
    "presence_penalty",
})

# Handled inside the gateway; must never leak into the Chat Completions body.
_LOCAL_FIELDS = frozenset({
    "max_output_tokens",
    "max_tool_calls",
    "truncation",
    "include",
    "stream_options",
    "previous_response_id",
    "conversation",
})

# Accepted and understood, but with no Chat Completions equivalent — dropped with a log.
_IGNORED_FIELDS = frozenset({
    "client_metadata",
    "background",
    "moderation",
    "prompt_cache_retention",
    "prompt_cache_options",
})

# Responses-only capabilities that need a server-side state machine this gateway
# does not have. Failing fast beats pretending to honour them.
_REJECTED_FIELDS = frozenset({
    "context_management",
    "prompt",
    "modalities",
    "audio",
})

_KNOWN_FIELDS = _MAPPED_FIELDS | _LOCAL_FIELDS | _IGNORED_FIELDS | _REJECTED_FIELDS

_ALLOWED_INCLUDE = frozenset({
    "file_search_call.results",
    "web_search_call.results",
    "web_search_call.action.sources",
    "message.input_image.image_url",
    "computer_call_output.output.image_url",
    "code_interpreter_call.outputs",
    "reasoning.encrypted_content",
    "message.output_text.logprobs",
})

_ALLOWED_REASONING_KEYS = frozenset({"effort", "summary", "generate_summary"})
_ALLOWED_TEXT_KEYS = frozenset({"format", "verbosity"})


class RequestConverter:
    """Convert Responses API requests to Chat Completions requests."""

    def __init__(self, tool_transformer: ToolTransformer | None = None) -> None:
        self._ct = ContentTransformer()
        self._input_tf = InputTransformer(self._ct)
        self._tool_tf = tool_transformer or ToolTransformer()

    async def convert(
        self,
        responses_request: dict[str, Any],
        conversation_messages: list[dict[str, Any]] | None = None,
        auditor: Any = None,
    ) -> tuple[dict[str, Any], ConversionContext]:
        """Convert a Responses API request to a Chat Completions request.

        Returns:
            Tuple of (chat_completions_request_body, conversion_context)

        Raises:
            UnsupportedParameterError: when a Responses field cannot be mapped,
                is handled locally, or is explicitly not supported.
        """
        self._validate_request(responses_request)

        context = ConversionContext(
            response_id=generate_response_id(),
            model=responses_request.get("model", ""),
            original_instructions=responses_request.get("instructions"),
            original_request=dict(responses_request),
        )

        chat_request: dict[str, Any] = {}

        # Model
        chat_request["model"] = responses_request.get("model", "")

        # Tools — converted before the input so that replayed namespaced tool
        # calls can be mapped back to their flattened upstream names.
        tools = responses_request.get("tools", [])
        if tools:
            chat_tools = self._tool_tf.convert_tools(tools, context)
            if chat_tools:
                chat_request["tools"] = chat_tools

        # Messages from input
        input_data = responses_request.get("input", "")
        instructions = responses_request.get("instructions")
        messages = self._input_tf.transform_input(input_data, instructions, context)

        # If previous_response_id provided, prepend stored conversation history
        if conversation_messages:
            # Insert stored messages before the new ones, but after system message
            system_msgs = [m for m in messages if m.get("role") == "system"]
            non_system_msgs = [m for m in messages if m.get("role") != "system"]
            messages = system_msgs + conversation_messages + non_system_msgs

        messages = sanitize_tool_call_messages(messages)
        chat_request["messages"] = messages

        # Tool choice
        tool_choice = responses_request.get("tool_choice")
        if tool_choice is not None:
            chat_request["tool_choice"] = self._tool_tf.convert_tool_choice(
                tool_choice, context
            )

        # max_output_tokens -> max_completion_tokens
        max_output = responses_request.get("max_output_tokens")
        if max_output is not None:
            chat_request["max_completion_tokens"] = max_output

        # Stream
        is_streaming = bool(responses_request.get("stream", False))
        chat_request["stream"] = is_streaming
        context.had_streaming = is_streaming

        if is_streaming:
            # Chat Completions only understands include_usage; include_obfuscation
            # is a Responses-only key that makes strict upstreams 400.
            stream_options = responses_request.get("stream_options") or {}
            if not isinstance(stream_options, dict):
                raise UnsupportedParameterError(
                    "stream_options", "stream_options must be an object."
                )
            chat_request["stream_options"] = {
                "include_usage": bool(stream_options.get("include_usage", True)),
            }

        # text.format -> response_format, text.verbosity -> system hint
        text_config = responses_request.get("text")
        if isinstance(text_config, dict):
            context.original_text_format = text_config
            fmt = text_config.get("format")
            if fmt:
                if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
                    # Many Chat Completions upstreams reject
                    # `response_format: {"type": "json_schema"}` outright
                    # ("This response_format type is unavailable now").
                    # Enforce the schema through the system prompt instead
                    # of forwarding a parameter the upstream cannot honour.
                    schema_hint = self._build_json_schema_hint(fmt)
                    if schema_hint:
                        self._apply_system_hint(messages, schema_hint)
                else:
                    chat_request["response_format"] = fmt

            verbosity = text_config.get("verbosity")
            if verbosity:
                self._apply_verbosity(messages, verbosity)

        # reasoning.effort -> reasoning_effort
        reasoning = responses_request.get("reasoning")
        if isinstance(reasoning, dict):
            effort = reasoning.get("effort")
            if effort:
                chat_request["reasoning_effort"] = effort

        # Pass-through fields that exist in Chat Completions
        for field in (
            "temperature",
            "top_p",
            "parallel_tool_calls",
            "seed",
            "stop",
            "service_tier",
            "store",
            "metadata",
            "user",
            "safety_identifier",
            "prompt_cache_key",
            "top_logprobs",
            "logprobs",
            "frequency_penalty",
            "presence_penalty",
        ):
            value = responses_request.get(field)
            if value is not None:
                chat_request[field] = value

        # Chat Completions requires logprobs=true for top_logprobs to be honoured.
        if chat_request.get("top_logprobs") and not chat_request.get("logprobs"):
            chat_request["logprobs"] = True

        # include — recorded; used to decide which optional output parts to emit
        include = responses_request.get("include")
        if isinstance(include, list):
            context.include_fields = include

        # max_tool_calls — record for truncation in response
        max_tool_calls = responses_request.get("max_tool_calls")
        if max_tool_calls is not None:
            context.max_tool_calls = max_tool_calls

        # truncation — record for retry-on-context-overflow
        truncation = responses_request.get("truncation")
        if truncation:
            context.truncation = truncation

        # Sanitize messages — fix broken tool_call arguments
        self._sanitize_messages(messages)

        # Audit: record converted request
        if auditor is not None:
            auditor.record_converted_request(chat_request)

        return chat_request, context

    @staticmethod
    def _validate_request(body: dict[str, Any]) -> None:
        """Classify every top-level field as mapped / local / ignored / rejected.

        Anything that does not fall into one of those buckets is refused so that
        Responses-only fields never silently reach the Chat Completions upstream.
        """
        for key, value in body.items():
            if value is None:
                continue
            if key not in _KNOWN_FIELDS:
                raise UnsupportedParameterError(
                    key, f"Unsupported parameter: '{key}'."
                )
            if key in _REJECTED_FIELDS:
                raise UnsupportedParameterError(
                    key, f"Parameter '{key}' is not supported by this gateway."
                )
            if key in _IGNORED_FIELDS:
                logger.info("request_field_ignored", field=key)

        if body.get("background") is True:
            raise UnsupportedParameterError(
                "background", "Parameter 'background=true' is not supported."
            )

        include = body.get("include")
        if include is not None:
            if not isinstance(include, list):
                raise UnsupportedParameterError("include", "include must be an array.")
            for value in include:
                if value not in _ALLOWED_INCLUDE:
                    raise UnsupportedParameterError(
                        "include", f"Unsupported include value: '{value}'."
                    )

        reasoning = body.get("reasoning")
        if isinstance(reasoning, dict):
            for key in reasoning:
                if key not in _ALLOWED_REASONING_KEYS:
                    raise UnsupportedParameterError(
                        f"reasoning.{key}",
                        f"Unsupported reasoning parameter: '{key}'.",
                    )

        text_config = body.get("text")
        if isinstance(text_config, dict):
            for key in text_config:
                if key not in _ALLOWED_TEXT_KEYS:
                    raise UnsupportedParameterError(
                        f"text.{key}", f"Unsupported text parameter: '{key}'."
                    )

    def _apply_verbosity(
        self, messages: list[dict[str, Any]], verbosity: str
    ) -> None:
        """Append a verbosity hint to the system message."""
        hints = {
            "low": "Be concise and brief in your responses.",
            "high": "Provide detailed and thorough responses.",
        }
        hint = hints.get(verbosity)
        if not hint:
            return

        # Append to existing system message, or create one
        for msg in messages:
            if msg.get("role") == "system" and isinstance(msg.get("content"), str):
                msg["content"] += f"\n{hint}"
                return

        messages.insert(0, {"role": "system", "content": hint})

    @staticmethod
    def _build_json_schema_hint(fmt: dict[str, Any]) -> str:
        """Render a system-prompt instruction that enforces a JSON Schema."""
        schema = fmt.get("schema")
        if not isinstance(schema, dict) or not schema:
            return ""
        try:
            schema_text = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            return ""
        header = (
            "You must respond with a single JSON value that strictly conforms "
            "to the following JSON Schema"
        )
        name = fmt.get("name")
        if isinstance(name, str) and name:
            header += f" (schema name: {name})"
        header += (
            ". Return the raw JSON only: no prose, no markdown code fences "
            "and no extra keys.\n\nJSON Schema:\n"
        )
        return header + schema_text

    @staticmethod
    def _apply_system_hint(messages: list[dict[str, Any]], hint: str) -> None:
        """Append an instruction hint to the leading system message."""
        for msg in messages:
            if msg.get("role") == "system" and isinstance(msg.get("content"), str):
                msg["content"] += f"\n\n{hint}"
                return
        messages.insert(0, {"role": "system", "content": hint})

    @staticmethod
    def _sanitize_messages(messages: list[dict[str, Any]]) -> None:
        """Fix broken tool_call arguments in messages to prevent upstream 400 errors.

        Some models (e.g. GLM) may generate malformed JSON in tool_call arguments.
        This breaks the upstream API contract. We attempt to repair truncated JSON
        or replace it with an empty object if unrepairable.
        """
        import json as _json

        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                continue

            for tc in tool_calls:
                func = tc.get("function")
                if not func:
                    continue
                args = func.get("arguments")
                if not args or not isinstance(args, str):
                    continue

                # Try parsing — if valid, skip
                try:
                    _json.loads(args)
                    continue
                except _json.JSONDecodeError:
                    pass

                # Attempt to repair truncated JSON
                repaired = RequestConverter._try_repair_json(args)
                func["arguments"] = repaired

    @staticmethod
    def _try_repair_json(raw: str) -> str:
        """Attempt to repair truncated/malformed JSON arguments.

        Common cases:
        - Truncated string: {"key": "value that got cut...
        - Truncated object: {"key": "value", "key2": "val...
        """
        import json as _json

        stripped = raw.strip()

        # Try closing open strings and objects
        attempts = []

        # Simple: just close all open structures
        repaired = stripped
        # Count unclosed quotes (outside of escaped ones)
        in_string = False
        escape = False
        for ch in stripped:
            if escape:
                escape = False
                continue
            if ch == '\\' and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string

        if in_string:
            repaired += '"'
        # Close open braces/brackets
        open_braces = repaired.count('{') - repaired.count('}')
        open_brackets = repaired.count('[') - repaired.count(']')
        repaired += ']' * max(0, open_brackets)
        repaired += '}' * max(0, open_braces)

        attempts.append(repaired)

        # Also try empty object as fallback
        attempts.append('{}')

        for attempt in attempts:
            try:
                _json.loads(attempt)
                return attempt
            except _json.JSONDecodeError:
                continue

        return '{}'