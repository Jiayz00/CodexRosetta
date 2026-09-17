"""Regression tests for Responses -> Chat Completions compatibility.

Each class here pins a bug that broke Codex-style clients against
Chat Completions upstreams.
"""
from __future__ import annotations

import json

import pytest

from codex_rosetta.converters.stream_converter import StreamConverter
from codex_rosetta.models.common import ConversionContext

JSON_SCHEMA_FORMAT = {
    "type": "json_schema",
    "name": "plan",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"steps": {"type": "array", "items": {"type": "string"}}},
        "required": ["steps"],
        "additionalProperties": False,
    },
}


class TestJsonSchemaFormat:
    """`response_format: {"type": "json_schema"}` is widely unsupported.

    Upstreams answer with "This response_format type is unavailable now", so the
    schema is enforced through the system prompt instead of being forwarded.
    """

    @pytest.mark.asyncio
    async def test_json_schema_not_forwarded_and_hinted_in_prompt(self, request_converter):
        chat_req, ctx = await request_converter.convert({
            "model": "gpt-4o",
            "instructions": "You are a coding agent.",
            "input": "Plan the work.",
            "text": {"format": JSON_SCHEMA_FORMAT},
        })

        assert "response_format" not in chat_req
        assert ctx.original_text_format == {"format": JSON_SCHEMA_FORMAT}

        system = chat_req["messages"][0]
        assert system["role"] == "system"
        assert "You are a coding agent." in system["content"]
        assert "JSON Schema" in system["content"]
        assert json.dumps(
            JSON_SCHEMA_FORMAT["schema"], separators=(",", ":")
        ) in system["content"]

    @pytest.mark.asyncio
    async def test_json_schema_creates_system_message_when_absent(self, request_converter):
        chat_req, _ = await request_converter.convert({
            "model": "gpt-4o",
            "input": "Plan the work.",
            "text": {"format": JSON_SCHEMA_FORMAT},
        })

        assert "response_format" not in chat_req
        assert chat_req["messages"][0]["role"] == "system"
        assert "JSON Schema" in chat_req["messages"][0]["content"]
        assert "plan" in chat_req["messages"][0]["content"]

    @pytest.mark.asyncio
    async def test_json_object_is_still_forwarded(self, request_converter):
        chat_req, _ = await request_converter.convert({
            "model": "gpt-4o",
            "input": "Return JSON.",
            "text": {"format": {"type": "json_object"}},
        })

        assert chat_req["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_json_schema_without_schema_is_dropped(self, request_converter):
        chat_req, _ = await request_converter.convert({
            "model": "gpt-4o",
            "input": "Hi",
            "text": {"format": {"type": "json_schema", "name": "empty"}},
        })

        assert "response_format" not in chat_req
        assert all(
            "JSON Schema" not in str(msg.get("content", ""))
            for msg in chat_req["messages"]
        )

    @pytest.mark.asyncio
    async def test_plain_text_format_is_untouched(self, request_converter):
        chat_req, _ = await request_converter.convert({
            "model": "gpt-4o",
            "input": "Hi",
            "text": {"format": {"type": "text"}},
        })

        assert chat_req["response_format"] == {"type": "text"}


class TestReasoningRoundTrip:
    """Thinking-mode upstreams require the reasoning text back on the assistant
    message; it used to be dropped, so every follow-up turn failed with
    "The `reasoning_content` in the thinking mode must be passed back to the API."
    """

    def test_reasoning_attaches_to_tool_call_turn(self, input_transformer):
        messages = input_transformer.transform_input(
            [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "ping"}]},
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": "I should call ping."}],
                    "encrypted_content": None,
                },
                {"type": "function_call", "name": "ping", "call_id": "call_1", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "pong"},
            ],
            instructions="You are Codex.",
        )

        assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool"]
        assert messages[2]["reasoning_content"] == "I should call ping."
        assert messages[2]["tool_calls"][0]["id"] == "call_1"
        assert messages[3]["content"] == "pong"

    def test_reasoning_attaches_to_plain_assistant_turn(self, input_transformer):
        messages = input_transformer.transform_input([
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "think"}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hello"}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "again"}]},
        ])

        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "hello"
        assert messages[1]["reasoning_content"] == "think"

    def test_multiple_reasoning_items_are_joined(self, input_transformer):
        messages = input_transformer.transform_input([
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "first"}]},
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "second"}]},
            {"type": "function_call", "name": "ping", "call_id": "c", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c", "output": "pong"},
        ])

        assert messages[0]["reasoning_content"] == "first\nsecond"

    def test_reasoning_content_parts_are_collected(self, input_transformer):
        """Reasoning payloads may carry `content` as well as `summary`."""
        messages = input_transformer.transform_input([
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "summarised"}],
                "content": [{"type": "reasoning_text", "text": "raw"}],
            },
            {"type": "function_call", "name": "ping", "call_id": "c", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c", "output": "pong"},
        ])

        assert messages[0]["reasoning_content"] == "summarised\nraw"

    def test_reasoning_without_text_gets_placeholder(self, input_transformer):
        """Thinking-mode upstreams still require a value on tool-call turns."""
        from codex_rosetta.converters.input_transformer import (
            MISSING_REASONING_PLACEHOLDER,
        )

        messages = input_transformer.transform_input([
            {"type": "reasoning", "summary": [], "encrypted_content": "abc"},
            {"type": "function_call", "name": "ping", "call_id": "c", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c", "output": "pong"},
        ])

        assert messages[0]["reasoning_content"] == MISSING_REASONING_PLACEHOLDER

    def test_stale_reasoning_does_not_leak_into_later_turn(self, input_transformer):
        messages = input_transformer.transform_input([
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "stale"}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]},
        ])

        assert messages[-1]["role"] == "assistant"
        assert "reasoning_content" not in messages[-1]


class TestToolCallOrdering:
    """Chat Completions requires a `tool` message to directly follow the
    assistant message that declared its `tool_calls`.

    Codex interleaves empty assistant placeholders, and can replay a tool call
    with no assistant message item at all; both shapes used to produce
    "An assistant message with 'tool_calls' must be followed by tool messages
    responding to each 'tool_call_id'."
    """

    def test_empty_assistant_placeholder_is_dropped(self, input_transformer):
        messages = input_transformer.transform_input([
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "run it"}]},
            {"type": "function_call", "name": "shell", "call_id": "call_1", "arguments": "{}"},
            {"type": "message", "role": "assistant", "content": []},
            {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
        ])

        assert [m["role"] for m in messages] == ["user", "assistant", "tool"]
        assert messages[1]["tool_calls"][0]["id"] == "call_1"
        assert messages[2]["tool_call_id"] == "call_1"

    def test_empty_string_content_assistant_is_dropped(self, input_transformer):
        messages = input_transformer.transform_input([
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            {"type": "message", "role": "assistant", "content": ""},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "again"}]},
        ])

        assert [m["role"] for m in messages] == ["user", "user"]

    def test_parallel_calls_stay_on_one_assistant_message(self, input_transformer):
        messages = input_transformer.transform_input([
            {"type": "function_call", "name": "a", "call_id": "c1", "arguments": "{}"},
            {"type": "message", "role": "assistant", "content": []},
            {"type": "function_call", "name": "b", "call_id": "c2", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "1"},
            {"type": "function_call_output", "call_id": "c2", "output": "2"},
        ])

        assert [m["role"] for m in messages] == ["assistant", "tool", "tool"]
        assert [tc["id"] for tc in messages[0]["tool_calls"]] == ["c1", "c2"]
        assert [m["tool_call_id"] for m in messages[1:]] == ["c1", "c2"]

    def test_tool_call_without_assistant_message_gets_one(self, input_transformer):
        messages = input_transformer.transform_input([
            {"type": "function_call", "name": "shell", "call_id": "call_9", "arguments": '{"cmd":"ls"}'},
            {"type": "function_call_output", "call_id": "call_9", "output": "ok"},
        ])

        assert len(messages) == 2
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] is None
        assert messages[0]["tool_calls"][0]["function"]["name"] == "shell"
        assert messages[1] == {"role": "tool", "tool_call_id": "call_9", "content": "ok"}

    def test_assistant_text_kept_alongside_tool_calls(self, input_transformer):
        messages = input_transformer.transform_input([
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "run it"}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Let me check."}]},
            {"type": "function_call", "name": "shell", "call_id": "call_t", "arguments": "{}"},
            {"type": "message", "role": "assistant", "content": []},
            {"type": "function_call_output", "call_id": "call_t", "output": "ok"},
        ])

        assert [m["role"] for m in messages] == ["user", "assistant", "tool"]
        assert messages[1]["content"] == "Let me check."
        assert messages[1]["tool_calls"][0]["id"] == "call_t"
        assert messages[2]["tool_call_id"] == "call_t"


class TestCustomToolCalls:
    """Custom tools are advertised upstream as functions, so the replayed tool
    call has to use the same shape or the request is rejected."""

    def test_custom_tool_call_uses_function_shape(self, input_transformer):
        messages = input_transformer.transform_input([
            {"type": "custom_tool_call", "name": "apply_patch", "call_id": "call_c", "input": "patch"},
            {"type": "custom_tool_call_output", "call_id": "call_c", "output": "applied"},
        ])

        assert messages[0]["role"] == "assistant"
        tool_call = messages[0]["tool_calls"][0]
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "apply_patch"
        assert json.loads(tool_call["function"]["arguments"]) == {"input": "patch"}
        assert messages[1]["role"] == "tool"
        assert messages[1]["tool_call_id"] == "call_c"


class TestReasoningStreamEventName:
    """`response.reasoning.delta` is not part of the Responses API event set;
    clients such as Codex ignore it, so reasoning was invisible while streaming."""

    @pytest.mark.asyncio
    async def test_reasoning_delta_uses_standard_event_name(self):
        ctx = ConversionContext(response_id="resp_x", model="gpt-4o")
        converter = StreamConverter(ctx)
        chunk = {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1746000000,
            "model": "gpt-4o",
            "choices": [
                {"index": 0, "delta": {"reasoning_content": "thinking"}, "finish_reason": None}
            ],
        }

        names = []
        async for event_type, event_data in converter.process_chunk(
            f"data: {json.dumps(chunk)}\n\n"
        ):
            if event_data is not None:
                names.append(event_type)

        assert "response.reasoning_summary_text.delta" in names
        assert "response.reasoning.delta" not in names

    @pytest.mark.asyncio
    async def test_reasoning_item_still_finalised_with_summary(self):
        ctx = ConversionContext(response_id="resp_y", model="gpt-4o")
        converter = StreamConverter(ctx)
        chunk = {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1746000000,
            "model": "gpt-4o",
            "choices": [
                {"index": 0, "delta": {"reasoning_content": "why"}, "finish_reason": None}
            ],
        }
        async for _ in converter.process_chunk(f"data: {json.dumps(chunk)}\n\n"):
            pass

        events = []
        async for event_type, event_data in converter.finalize():
            events.append((event_type, event_data))

        done = [d for t, d in events if t == "response.output_item.done"]
        reasoning = [d for d in done if d.get("item", {}).get("type") == "reasoning"]
        assert len(reasoning) == 1
        assert reasoning[0]["item"]["summary"] == [{"type": "summary_text", "text": "why"}]