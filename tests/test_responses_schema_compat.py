"""Responses API field-consistency tests for PR #12.

Covers the four layers the gateway has to get right: request fields, input
items, tools/search, and the response/SSE/error envelope.
"""

from __future__ import annotations

import json

import pytest

from codex_rosetta.converters.content_transformer import ContentTransformer
from codex_rosetta.converters.request_converter import RequestConverter
from codex_rosetta.converters.response_converter import ResponseConverter
from codex_rosetta.converters.stream_converter import StreamConverter
from codex_rosetta.converters.tool_transformer import ToolTransformer
from codex_rosetta.models.common import (
    ConversionContext,
    UnsupportedParameterError,
)

LONG_CALL_ID = "call_" + "0" * 30


def _chunk(delta: dict, finish_reason: str | None = None) -> str:
    return json.dumps({
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1746000000,
        "model": "gpt-4o",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    })


async def _run_stream(context: ConversionContext, chunks: list[str]):
    converter = StreamConverter(context)
    events = []
    for chunk in chunks:
        async for evt in converter.process_chunk(f"data: {chunk}\n\n"):
            events.append(evt)
    async for evt in converter.finalize():
        events.append(evt)
    return events


# --------------------------------------------------------------------------
# 1. request field matrix
# --------------------------------------------------------------------------

MAPPED_CASES = [
    ("temperature", 0.3, "temperature"),
    ("top_p", 0.9, "top_p"),
    ("parallel_tool_calls", False, "parallel_tool_calls"),
    ("metadata", {"k": "v"}, "metadata"),
    ("service_tier", "auto", "service_tier"),
    ("store", False, "store"),
    ("user", "u-1", "user"),
    ("safety_identifier", "s-1", "safety_identifier"),
    ("prompt_cache_key", "cache-1", "prompt_cache_key"),
    ("logprobs", True, "logprobs"),
    ("seed", 7, "seed"),
    ("stop", ["\n\n"], "stop"),
    ("frequency_penalty", 0.1, "frequency_penalty"),
    ("presence_penalty", 0.2, "presence_penalty"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value,chat_key", MAPPED_CASES)
async def test_mapped_fields_reach_chat_body(field, value, chat_key):
    chat, _ = await RequestConverter().convert(
        {"model": "m", "input": "hi", field: value}
    )
    assert chat[chat_key] == value


@pytest.mark.asyncio
async def test_max_output_tokens_maps_to_max_completion_tokens():
    chat, _ = await RequestConverter().convert(
        {"model": "m", "input": "hi", "max_output_tokens": 256}
    )
    assert chat["max_completion_tokens"] == 256
    assert "max_output_tokens" not in chat


@pytest.mark.asyncio
async def test_reasoning_effort_maps_and_summary_is_ignored():
    chat, _ = await RequestConverter().convert({
        "model": "m",
        "input": "hi",
        "reasoning": {"effort": "high", "summary": "detailed"},
    })
    assert chat["reasoning_effort"] == "high"
    assert "reasoning" not in chat


IGNORED_CASES = [
    ("client_metadata", {"thread_id": "t1"}),
    ("background", False),
    ("moderation", {"model": "omni-moderation-latest"}),
    ("prompt_cache_retention", "24h"),
    ("prompt_cache_options", {"mode": "explicit", "ttl": "1h"}),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", IGNORED_CASES)
async def test_ignored_fields_never_reach_chat_body(field, value):
    chat, _ = await RequestConverter().convert(
        {"model": "m", "input": "hi", field: value}
    )
    assert field not in chat


REJECTED_CASES = [
    ({"context_management": [{"type": "truncation"}]}, "context_management"),
    ({"prompt": {"id": "pmpt_1"}}, "prompt"),
    ({"modalities": ["audio"]}, "modalities"),
    ({"audio": {"format": "wav"}}, "audio"),
    ({"background": True}, "background"),
    ({"reasoning": {"mode": "pro"}}, "reasoning.mode"),
    ({"reasoning": {"context": "all_turns"}}, "reasoning.context"),
    ({"include": ["not_a_real_include"]}, "include"),
    ({"text": {"not_a_real_text_key": 1}}, "text.not_a_real_text_key"),
    ({"totally_unknown_field": 1}, "totally_unknown_field"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("extra,param", REJECTED_CASES)
async def test_unsupported_fields_fail_fast(extra, param):
    with pytest.raises(UnsupportedParameterError) as exc:
        await RequestConverter().convert({"model": "m", "input": "hi", **extra})
    assert exc.value.param == param
    body = exc.value.to_error_body()
    assert body["error"]["code"] == "unsupported_parameter"
    assert body["error"]["param"] == param


CHAT_BODY_ALLOWLIST = {
    "model", "messages", "tools", "tool_choice", "temperature", "top_p",
    "max_completion_tokens", "parallel_tool_calls", "stream", "stream_options",
    "response_format", "reasoning_effort", "seed", "stop", "service_tier",
    "store", "metadata", "user", "safety_identifier", "prompt_cache_key",
    "top_logprobs", "logprobs", "frequency_penalty", "presence_penalty",
}


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value,chat_key", MAPPED_CASES)
async def test_no_field_leaks_beyond_chat_allowlist(field, value, chat_key):
    """Responses-only fields must never be forwarded verbatim."""
    chat, _ = await RequestConverter().convert(
        {"model": "m", "input": "hi", field: value}
    )
    assert set(chat) <= CHAT_BODY_ALLOWLIST


@pytest.mark.asyncio
async def test_stream_options_drops_include_obfuscation():
    chat, _ = await RequestConverter().convert({
        "model": "m",
        "input": "hi",
        "stream": True,
        "stream_options": {"include_usage": True, "include_obfuscation": True},
    })
    assert chat["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_top_logprobs_turns_on_logprobs():
    chat, _ = await RequestConverter().convert(
        {"model": "m", "input": "hi", "top_logprobs": 3}
    )
    assert chat["top_logprobs"] == 3
    assert chat["logprobs"] is True


# --------------------------------------------------------------------------
# 2. input items, content parts, tool call ids
# --------------------------------------------------------------------------


def test_input_file_becomes_chat_file_part():
    ct = ContentTransformer()
    assert ct.responses_input_to_chat_content([
        {"type": "input_file", "file_id": "file-1", "filename": "a.txt"},
    ]) == [{"type": "file", "file": {"file_id": "file-1", "filename": "a.txt"}}]


def test_input_file_never_becomes_image_url():
    ct = ContentTransformer()
    out = ct.responses_input_to_chat_content([
        {"type": "input_file", "file_data": "QQ==", "filename": "a.bin"},
    ])
    assert out[0]["type"] == "file"
    assert "image_url" not in out[0]


def test_unknown_content_part_fails_fast():
    ct = ContentTransformer()
    with pytest.raises(UnsupportedParameterError) as exc:
        ct.responses_input_to_chat_content([{"type": "input_audio", "input_audio": {}}])
    assert exc.value.param == "input.content"


def test_computer_screenshot_content_part_fails_fast():
    ct = ContentTransformer()
    with pytest.raises(UnsupportedParameterError):
        ct.responses_input_to_chat_content([{"type": "computer_screenshot"}])


@pytest.mark.asyncio
@pytest.mark.parametrize("item_type", ["file_search_call", "computer_call", "mcp_call"])
async def test_rejected_input_item_types_fail_fast(item_type):
    with pytest.raises(UnsupportedParameterError) as exc:
        await RequestConverter().convert({
            "model": "m",
            "input": [{"type": item_type, "id": "x"}],
        })
    assert exc.value.param == "input[0].type"


@pytest.mark.asyncio
async def test_unknown_input_item_type_fails_fast():
    with pytest.raises(UnsupportedParameterError) as exc:
        await RequestConverter().convert({
            "model": "m",
            "input": [{"type": "not_a_real_item"}],
        })
    assert exc.value.param == "input[0].type"


@pytest.mark.asyncio
async def test_short_call_id_is_normalized_and_stays_paired():
    chat, _ = await RequestConverter().convert({
        "model": "m",
        "input": [
            {"type": "function_call", "name": "f", "call_id": "call_m1", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_m1", "output": "ok"},
        ],
    })
    messages = chat["messages"]
    call_id = messages[0]["tool_calls"][0]["id"]
    assert len(call_id) >= 29
    assert messages[1]["tool_call_id"] == call_id


@pytest.mark.asyncio
async def test_orphan_tool_output_becomes_user_message():
    chat, _ = await RequestConverter().convert({
        "model": "m",
        "input": [
            {"type": "function_call_output", "call_id": "", "output": "delegation result"},
        ],
    })
    assert chat["messages"][0]["role"] == "user"
    assert chat["messages"][0]["content"] == "delegation result"
    assert all(m.get("tool_call_id") != "" for m in chat["messages"])


@pytest.mark.asyncio
async def test_web_search_call_replayed_as_tool_pair():
    chat, _ = await RequestConverter().convert({
        "model": "m",
        "input": [
            {"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "Searching"}
            ]},
            {"type": "web_search_call", "id": "ws_1", "status": "completed", "action": {
                "type": "search",
                "query": "weather",
                "sources": [{"type": "url", "url": "https://a.example"}],
            }},
            {"type": "message", "role": "user", "content": "thanks"},
        ],
    })
    messages = chat["messages"]
    assert [m["role"] for m in messages] == ["assistant", "tool", "user"]
    tool_call = messages[0]["tool_calls"][0]
    assert tool_call["function"]["name"] == "__rosetta_web_search"
    assert messages[1]["tool_call_id"] == tool_call["id"]
    assert "https://a.example" in messages[1]["content"]


@pytest.mark.asyncio
async def test_assistant_phase_from_client_is_not_leaked_upstream():
    chat, _ = await RequestConverter().convert({
        "model": "m",
        "input": [
            {"type": "message", "role": "assistant", "content": "thinking", "phase": "commentary"},
            {"type": "message", "role": "user", "content": "go"},
        ],
    })
    assert "phase" not in chat["messages"][0]


# --------------------------------------------------------------------------
# 3. tools, tool_choice, namespace
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_tool_type_fails_fast():
    with pytest.raises(UnsupportedParameterError) as exc:
        await RequestConverter().convert({
            "model": "m",
            "input": "hi",
            "tools": [{"type": "function", "name": "ok", "parameters": {}},
                      {"type": "not_a_tool"}],
        })
    assert exc.value.param == "tools[1].type"


@pytest.mark.asyncio
async def test_namespace_tools_flattened_and_restored():
    converter = RequestConverter()
    chat, context = await converter.convert({
        "model": "m",
        "input": "hi",
        "tools": [{
            "type": "namespace",
            "name": "multi_agent_v1",
            "tools": [
                {"type": "function", "name": "spawn_agent", "description": "spawn", "parameters": {}},
                {"type": "function", "name": "close_agent", "description": "close", "parameters": {}},
            ],
        }],
    })
    names = [t["function"]["name"] for t in chat["tools"]]
    assert names == ["multi_agent_v1__spawn_agent", "multi_agent_v1__close_agent"]
    assert context.namespace_tools["multi_agent_v1__close_agent"] == ("multi_agent_v1", "close_agent")
    assert chat["tools"][0]["function"]["description"].startswith("[multi_agent_v1]")

    response = ResponseConverter().convert({
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": LONG_CALL_ID,
                    "type": "function",
                    "function": {"name": "multi_agent_v1__close_agent", "arguments": "{}"},
                }],
            },
        }],
        "usage": {},
    }, context)

    assert [o["type"] for o in response["output"]] == ["function_call"]
    fc = response["output"][0]
    assert fc["name"] == "close_agent"
    assert fc["namespace"] == "multi_agent_v1"


@pytest.mark.asyncio
async def test_namespace_collision_gets_suffix():
    chat, context = await RequestConverter().convert({
        "model": "m",
        "input": "hi",
        "tools": [
            {"type": "function", "name": "ns__dup", "parameters": {}},
            {"type": "namespace", "name": "ns", "tools": [
                {"type": "function", "name": "dup", "parameters": {}},
            ]},
        ],
    })
    names = [t["function"]["name"] for t in chat["tools"]]
    assert names == ["ns__dup", "ns__dup_2"]
    assert context.namespace_tools["ns__dup_2"] == ("ns", "dup")


@pytest.mark.asyncio
async def test_namespace_function_call_replay_uses_flattened_name():
    chat, _ = await RequestConverter().convert({
        "model": "m",
        "tools": [{
            "type": "namespace",
            "name": "multi_agent_v1",
            "tools": [{"type": "function", "name": "close_agent", "parameters": {}}],
        }],
        "input": [
            {"type": "function_call", "name": "close_agent", "namespace": "multi_agent_v1",
             "call_id": LONG_CALL_ID, "arguments": "{}"},
            {"type": "function_call_output", "call_id": LONG_CALL_ID, "output": "ok"},
        ],
    })
    assert chat["messages"][0]["tool_calls"][0]["function"]["name"] == "multi_agent_v1__close_agent"


@pytest.mark.asyncio
async def test_custom_tool_becomes_function_with_input_argument():
    chat, context = await RequestConverter().convert({
        "model": "m",
        "input": "hi",
        "tools": [{"type": "custom", "custom": {"name": "apply_patch", "description": "patch"}}],
    })
    tool = chat["tools"][0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "apply_patch"
    assert "input" in tool["function"]["parameters"]["properties"]
    assert "apply_patch" in context.custom_tool_names


@pytest.mark.asyncio
@pytest.mark.parametrize("choice,expected", [
    ("auto", "auto"),
    ("none", "none"),
    ("required", "required"),
    ({"type": "function", "name": "get_weather"},
     {"type": "function", "function": {"name": "get_weather"}}),
    ({"type": "custom", "custom": {"name": "apply_patch"}},
     {"type": "function", "function": {"name": "apply_patch"}}),
    ({"type": "web_search"},
     {"type": "function", "function": {"name": "__rosetta_web_search"}}),
    ({"type": "namespace", "name": "multi_agent_v1"}, "auto"),
    ({"type": "bogus_tool_choice"}, "auto"),
])
async def test_tool_choice_shapes(choice, expected):
    context = ConversionContext(response_id="r", model="m")
    assert ToolTransformer().convert_tool_choice(choice, context) == expected


@pytest.mark.asyncio
async def test_tool_choice_function_uses_flattened_namespace_name():
    converter = RequestConverter()
    _, context = await converter.convert({
        "model": "m",
        "input": "hi",
        "tools": [{
            "type": "namespace",
            "name": "multi_agent_v1",
            "tools": [{"type": "function", "name": "spawn_agent", "parameters": {}}],
        }],
    })
    converted = ToolTransformer().convert_tool_choice(
        {"type": "function", "name": "multi_agent_v1.spawn_agent"}, context
    )
    assert converted == {
        "type": "function",
        "function": {"name": "multi_agent_v1__spawn_agent"},
    }


# --------------------------------------------------------------------------
# 4. response / SSE / error envelope
# --------------------------------------------------------------------------


def _chat_response(message: dict, finish_reason: str = "stop") -> dict:
    return {
        "id": "chatcmpl-abc",
        "object": "chat.completion",
        "created": 1746000000,
        "model": "gpt-4o",
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def test_envelope_always_has_core_fields():
    context = ConversionContext(response_id="resp_1", model="m")
    result = ResponseConverter().convert(_chat_response({"role": "assistant", "content": "Hi"}), context)
    for key in ("id", "object", "created_at", "completed_at", "model", "status",
                "output", "error", "incomplete_details", "usage"):
        assert key in result
    assert result["status"] == "completed"
    assert result["error"] is None


def test_envelope_echoes_accepted_request_fields():
    context = ConversionContext(
        response_id="resp_1",
        model="m",
        original_request={"model": "m", "input": "hi", "temperature": 0.2, "store": False},
    )
    result = ResponseConverter().convert(_chat_response({"role": "assistant", "content": "Hi"}), context)
    assert result["temperature"] == 0.2
    assert result["store"] is False
    assert "input" not in result


def test_finish_reason_length_is_incomplete():
    context = ConversionContext(response_id="resp_1", model="m")
    result = ResponseConverter().convert(
        _chat_response({"role": "assistant", "content": "cut off"}, "length"), context
    )
    assert result["status"] == "incomplete"
    assert result["incomplete_details"] == {"reason": "max_output_tokens"}


def test_finish_reason_content_filter_is_incomplete():
    context = ConversionContext(response_id="resp_1", model="m")
    result = ResponseConverter().convert(
        _chat_response({"role": "assistant", "content": "x"}, "content_filter"), context
    )
    assert result["status"] == "incomplete"
    assert result["incomplete_details"] == {"reason": "content_filter"}


def test_no_empty_assistant_message_item():
    context = ConversionContext(response_id="resp_1", model="m")
    result = ResponseConverter().convert(_chat_response({
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": LONG_CALL_ID,
            "type": "function",
            "function": {"name": "f", "arguments": "{}"},
        }],
    }, "tool_calls"), context)
    assert [o["type"] for o in result["output"]] == ["function_call"]


def test_phase_is_commentary_when_tool_calls_present():
    context = ConversionContext(response_id="resp_1", model="m")
    result = ResponseConverter().convert(_chat_response({
        "role": "assistant",
        "content": "let me check",
        "tool_calls": [{
            "id": LONG_CALL_ID,
            "type": "function",
            "function": {"name": "f", "arguments": "{}"},
        }],
    }, "tool_calls"), context)
    message = [o for o in result["output"] if o["type"] == "message"][0]
    assert message["phase"] == "commentary"


def test_phase_is_final_answer_without_tool_calls():
    context = ConversionContext(response_id="resp_1", model="m")
    result = ResponseConverter().convert(
        _chat_response({"role": "assistant", "content": "done"}), context
    )
    message = [o for o in result["output"] if o["type"] == "message"][0]
    assert message["phase"] == "final_answer"


def test_empty_output_arguments_are_normalized():
    context = ConversionContext(response_id="resp_1", model="m")
    result = ResponseConverter().convert(_chat_response({
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": LONG_CALL_ID,
            "type": "function",
            "function": {"name": "f", "arguments": ""},
        }],
    }, "tool_calls"), context)
    assert result["output"][0]["arguments"] == "{}"


@pytest.mark.asyncio
async def test_stream_created_event_is_in_progress():
    context = ConversionContext(response_id="resp_1", model="m")
    events = await _run_stream(context, [_chunk({"content": "hi"})])
    created = [e[1] for e in events if e[0] == "response.created"][0]
    assert created["response"]["status"] == "in_progress"


@pytest.mark.asyncio
async def test_stream_empty_content_frame_produces_no_message_events():
    context = ConversionContext(response_id="resp_1", model="m")
    events = await _run_stream(context, [
        _chunk({"role": "assistant", "content": ""}),
        _chunk({"content": "Hello"}),
    ])
    added_messages = [
        e for e in events
        if e[0] == "response.output_item.added"
        and e[1].get("item", {}).get("type") == "message"
    ]
    assert len(added_messages) == 1
    assert added_messages[0][1]["item"]["id"]
    deltas = [e[1]["delta"] for e in events if e[0] == "response.output_text.delta"]
    assert deltas == ["Hello"]


@pytest.mark.asyncio
async def test_stream_phase_commentary_for_tool_round():
    context = ConversionContext(response_id="resp_1", model="m")
    events = await _run_stream(context, [
        _chunk({"content": "let me check"}),
        _chunk({"tool_calls": [
            {"index": 0, "id": LONG_CALL_ID, "type": "function",
             "function": {"name": "f", "arguments": "{}"}},
        ]}),
        _chunk({}, finish_reason="tool_calls"),
    ])
    done = [e[1] for e in events if e[0] == "response.output_item.done"]
    message = [d for d in done if d["item"]["type"] == "message"][0]
    assert message["item"]["phase"] == "commentary"


@pytest.mark.asyncio
async def test_stream_finish_reason_length_emits_response_incomplete():
    context = ConversionContext(response_id="resp_1", model="m")
    events = await _run_stream(context, [
        _chunk({"content": "cut"}),
        _chunk({}, finish_reason="length"),
    ])
    types = [e[0] for e in events]
    assert "response.incomplete" in types
    assert "response.completed" not in types
    terminal = [e[1] for e in events if e[0] == "response.incomplete"][0]
    assert terminal["response"]["status"] == "incomplete"
    assert terminal["response"]["incomplete_details"] == {"reason": "max_output_tokens"}


@pytest.mark.asyncio
async def test_stream_sequence_numbers_strictly_increase():
    context = ConversionContext(response_id="resp_1", model="m")
    events = await _run_stream(context, [
        _chunk({"role": "assistant", "content": ""}),
        _chunk({"content": "hi"}),
        _chunk({"tool_calls": [
            {"index": 0, "id": LONG_CALL_ID, "type": "function",
             "function": {"name": "f", "arguments": "{}"}},
        ]}),
        _chunk({}, finish_reason="tool_calls"),
    ])
    seq = [e[1]["sequence_number"] for e in events if e[1]]
    assert seq == sorted(seq)
    assert len(seq) == len(set(seq))


def test_router_returns_400_for_unsupported_parameter():
    from fastapi.testclient import TestClient

    from codex_rosetta.main import app

    with TestClient(app) as client:
        resp = client.post("/v1/responses", json={
            "model": "m",
            "input": "hi",
            "context_management": [{"type": "truncation"}],
        })
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["code"] == "unsupported_parameter"
    assert body["error"]["param"] == "context_management"
    assert body["error"]["type"] == "invalid_request_error"