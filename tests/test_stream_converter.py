import json
import pytest

from codex_rosetta.converters.stream_converter import StreamConverter
from codex_rosetta.models.common import ConversionContext, make_simulated_function_name


@pytest.fixture
def ctx():
    return ConversionContext(response_id="resp_stream_test", model="gpt-4o")


def make_chunk(delta: dict, finish_reason=None, usage=None):
    """Helper to create a Chat Completions SSE chunk."""
    chunk = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1746000000,
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    }
    if usage:
        chunk["usage"] = usage
    return f"data: {json.dumps(chunk)}\n\n"


DONE_SENTINEL = "data: [DONE]\n\n"


class TestTextOnlyStreaming:
    @pytest.mark.asyncio
    async def test_text_stream_lifecycle(self, ctx):
        sc = StreamConverter(ctx)
        events = []

        # First chunk with role
        async for evt in sc.process_chunk(
            f'data: {json.dumps({"id":"chatcmpl-test","object":"chat.completion.chunk","created":1746000000,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":None}]})}\n\n'
        ):
            events.append(evt)

        # The empty-content frame must not materialize a message item: doing so
        # used to make the Codex client render a spurious empty assistant
        # message after every tool round, which stopped it from grouping
        # consecutive commands into one folded work block.
        assert [e[0] for e in events] == ["response.created", "response.in_progress"]

        # Text delta chunks
        async for evt in sc.process_chunk(make_chunk({"content": "Hello"})):
            events.append(evt)
        async for evt in sc.process_chunk(make_chunk({"content": " world"})):
            events.append(evt)

        # Finalize
        async for evt in sc.finalize():
            events.append(evt)

        event_types = [e[0] for e in events]
        assert "response.created" in event_types
        assert "response.in_progress" in event_types
        assert "response.output_item.added" in event_types
        assert "response.content_part.added" in event_types
        assert "response.output_text.delta" in event_types
        assert "response.output_text.done" in event_types
        assert "response.content_part.done" in event_types
        assert "response.output_item.done" in event_types
        assert "response.completed" in event_types

        # The role-only frame carries content="" and must NOT produce a message
        # item or an empty text delta (that is what broke Codex work-block folding).
        deltas = [e[1] for e in events if e[0] == "response.output_text.delta"]
        assert len(deltas) == 2
        assert [d["delta"] for d in deltas] == ["Hello", " world"]
        assert all(d["delta"] != "" for d in deltas)
        added_items = [e[1]["item"] for e in events if e[0] == "response.output_item.added"]
        assert len(added_items) == 1

        # Check done text is accumulated
        done_events = [e[1] for e in events if e[0] == "response.output_text.done"]
        assert done_events[0]["text"] == "Hello world"

    @pytest.mark.asyncio
    async def test_sequence_numbers_monotonic(self, ctx):
        sc = StreamConverter(ctx)
        events = []

        async for evt in sc.process_chunk(make_chunk({"content": "Hi"})):
            events.append(evt)
        async for evt in sc.finalize():
            events.append(evt)

        seq_nums = [e[1].get("sequence_number", 0) for e in events if e[1]]
        assert seq_nums == sorted(seq_nums)  # monotonically increasing
        assert len(seq_nums) == len(set(seq_nums))  # unique


class TestToolCallStreaming:
    @pytest.mark.asyncio
    async def test_tool_call_stream(self, ctx):
        sc = StreamConverter(ctx)
        events = []

        # Tool call start
        async for evt in sc.process_chunk(make_chunk({
            "tool_calls": [{
                "index": 0,
                "id": "call_abc",
                "type": "function",
                "function": {"name": "get_weather", "arguments": ""},
            }]
        })):
            events.append(evt)

        # Arguments delta
        async for evt in sc.process_chunk(make_chunk({
            "tool_calls": [{
                "index": 0,
                "function": {"arguments": '{"location":"SF"}'},
            }]
        })):
            events.append(evt)

        # Finalize
        async for evt in sc.finalize():
            events.append(evt)

        event_types = [e[0] for e in events]
        assert "response.output_item.added" in event_types
        assert "response.function_call_arguments.delta" in event_types
        assert "response.function_call_arguments.done" in event_types
        assert "response.output_item.done" in event_types

        # Check function call item
        added_events = [e[1] for e in events if e[0] == "response.output_item.added"]
        fc_added = [e for e in added_events if e.get("item", {}).get("type") == "function_call"]
        assert len(fc_added) == 1
        assert fc_added[0]["item"]["call_id"] == "call_abc"
        assert fc_added[0]["item"]["name"] == "get_weather"

        # Check arguments done
        args_done = [e[1] for e in events if e[0] == "response.function_call_arguments.done"]
        assert args_done[0]["arguments"] == '{"location":"SF"}'

    @pytest.mark.asyncio
    async def test_parallel_tool_calls_stream(self, ctx):
        sc = StreamConverter(ctx)
        events = []

        # First tool call
        async for evt in sc.process_chunk(make_chunk({
            "tool_calls": [{
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "func_a", "arguments": ""},
            }]
        })):
            events.append(evt)

        # Second tool call
        async for evt in sc.process_chunk(make_chunk({
            "tool_calls": [{
                "index": 1,
                "id": "call_2",
                "type": "function",
                "function": {"name": "func_b", "arguments": ""},
            }]
        })):
            events.append(evt)

        # Finalize
        async for evt in sc.finalize():
            events.append(evt)

        fc_items = [e[1] for e in events if e[0] == "response.output_item.added"
                     and isinstance(e[1], dict) and e[1].get("item", {}).get("type") == "function_call"]
        assert len(fc_items) == 2


class TestDoneSentinel:
    @pytest.mark.asyncio
    async def test_done_sentinel_ignored(self, ctx):
        sc = StreamConverter(ctx)
        events = []

        async for evt in sc.process_chunk(DONE_SENTINEL):
            events.append(evt)

        async for evt in sc.finalize():
            events.append(evt)

        # Should still get created/in_progress/completed events
        event_types = [e[0] for e in events]
        assert "response.created" in event_types
        assert "response.completed" in event_types


class TestUsageInStream:
    @pytest.mark.asyncio
    async def test_usage_in_final_chunk(self, ctx):
        sc = StreamConverter(ctx)
        events = []

        async for evt in sc.process_chunk(make_chunk({"content": "Hi"})):
            events.append(evt)

        # Usage chunk (Chat Completions sends this before [DONE] when include_usage=True)
        async for evt in sc.process_chunk(
            f'data: {json.dumps({"id":"chatcmpl-test","object":"chat.completion.chunk","created":1746000000,"model":"gpt-4o","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}})}\n\n'
        ):
            events.append(evt)

        async for evt in sc.finalize():
            events.append(evt)

        completed_events = [e[1] for e in events if e[0] == "response.completed"]
        assert len(completed_events) == 1
        usage = completed_events[0]["response"]["usage"]
        assert usage["input_tokens"] == 10
        assert usage["output_tokens"] == 5
        assert usage["total_tokens"] == 15


class TestBuiltinToolReverseMapping:
    @pytest.mark.asyncio
    async def test_web_search_reverse_in_stream(self, ctx):
        sim_name = make_simulated_function_name("web_search")
        ctx.register_builtin_tool(sim_name, "web_search")

        sc = StreamConverter(ctx)
        events = []

        async for evt in sc.process_chunk(make_chunk({
            "tool_calls": [{
                "index": 0,
                "id": "call_ws",
                "type": "function",
                "function": {"name": sim_name, "arguments": ""},
            }]
        })):
            events.append(evt)

        async for evt in sc.process_chunk(make_chunk({
            "tool_calls": [{
                "index": 0,
                "function": {"arguments": '{"query":"AI news"}'},
            }]
        })):
            events.append(evt)

        async for evt in sc.finalize():
            events.append(evt)

        # Check that the final output item is a web_search_call, not function_call
        done_events = [e[1] for e in events if e[0] == "response.output_item.done"]
        ws_items = [e for e in done_events if e.get("item", {}).get("type") == "web_search_call"]
        assert len(ws_items) == 1

        # Check web_search lifecycle events
        ws_lifecycle = [e[0] for e in events if "web_search_call" in e[0]]
        assert "response.web_search_call.completed" in ws_lifecycle

    @pytest.mark.asyncio
    async def test_builtin_tool_current_output_items(self, ctx):
        sim_name = make_simulated_function_name("file_search")
        ctx.register_builtin_tool(sim_name, "file_search")

        sc = StreamConverter(ctx)

        async for evt in sc.process_chunk(make_chunk({
            "tool_calls": [{
                "index": 0,
                "id": "call_fs",
                "type": "function",
                "function": {"name": sim_name, "arguments": '{"query":"docs"}'},
            }]
        })):
            pass

        async for evt in sc.finalize():
            pass

        items = sc.current_output_items
        fs_items = [i for i in items if i.get("type") == "file_search_call"]
        assert len(fs_items) == 1


class TestMixedTextAndToolCalls:
    @pytest.mark.asyncio
    async def test_text_then_tool_call(self, ctx):
        sc = StreamConverter(ctx)
        events = []

        # Text content
        async for evt in sc.process_chunk(make_chunk({"content": "Let me search."})):
            events.append(evt)

        # Tool call
        async for evt in sc.process_chunk(make_chunk({
            "tool_calls": [{
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": "{}"},
            }]
        })):
            events.append(evt)

        async for evt in sc.finalize():
            events.append(evt)

        # Should have both message and function_call output items
        output = sc.current_output_items
        types = [i.get("type") for i in output]
        assert "message" in types
        assert "function_call" in types


class TestRoundFiltering:
    @pytest.mark.asyncio
    async def test_current_output_items_keep_search_items_and_latest_round(self, ctx):
        sim_name = make_simulated_function_name("web_search")
        ctx.register_builtin_tool(sim_name, "web_search")

        sc = StreamConverter(ctx)

        async for _ in sc.process_chunk(make_chunk({
            "tool_calls": [{
                "index": 0,
                "id": "call_ws",
                "type": "function",
                "function": {"name": sim_name, "arguments": '{"query":"AI news"}'},
            }]
        })):
            pass

        sc.prepare_for_next_round()

        async for _ in sc.process_chunk(make_chunk({"content": "Final answer"})):
            pass

        items = sc.current_output_items
        # Built-in tool items survive the round transition so the client keeps
        # the visible search history; chat items only reflect the last round.
        assert [item["type"] for item in items] == ["web_search_call", "message"]
        assert items[0]["action"]["queries"] == ["AI news"]
        assert items[1]["content"][0]["text"] == "Final answer"

    @pytest.mark.asyncio
    async def test_completed_response_includes_search_and_latest_round(self, ctx):
        sim_name = make_simulated_function_name("web_search")
        ctx.register_builtin_tool(sim_name, "web_search")

        sc = StreamConverter(ctx)
        events = []

        async for evt in sc.process_chunk(make_chunk({
            "tool_calls": [{
                "index": 0,
                "id": "call_ws",
                "type": "function",
                "function": {"name": sim_name, "arguments": '{"query":"AI news"}'},
            }]
        })):
            events.append(evt)

        sc.prepare_for_next_round()

        async for evt in sc.process_chunk(make_chunk({"content": "Final answer"})):
            events.append(evt)

        async for evt in sc.finalize():
            events.append(evt)

        completed = [event for event in events if event[0] == "response.completed"][0][1]
        output = completed["response"]["output"]
        assert [item["type"] for item in output] == ["web_search_call", "message"]
        assert output[0]["action"]["queries"] == ["AI news"]
        assert output[1]["content"][0]["text"] == "Final answer"
def _sequence_numbers(events):
    return [
        e[1]["sequence_number"]
        for e in events
        if isinstance(e[1], dict) and "sequence_number" in e[1]
    ]


class TestEmptyContentFrames:
    @pytest.mark.asyncio
    async def test_role_only_frame_creates_no_message_item(self, ctx):
        sc = StreamConverter(ctx)
        events = []

        # Every upstream round opens with a role + empty content frame
        async for evt in sc.process_chunk(make_chunk({"role": "assistant", "content": ""})):
            events.append(evt)

        async for evt in sc.process_chunk(make_chunk({
            "tool_calls": [{
                "index": 0,
                "id": "call_abc",
                "type": "function",
                "function": {"name": "exec_command", "arguments": '{"cmd":"ls"}'},
            }]
        }, finish_reason="tool_calls")):
            events.append(evt)

        async for evt in sc.finalize():
            events.append(evt)

        event_types = [e[0] for e in events]
        assert "response.content_part.added" not in event_types
        assert "response.output_text.delta" not in event_types
        assert "response.output_text.done" not in event_types
        assert "response.content_part.done" not in event_types

        added_items = [
            e[1]["item"]["type"] for e in events if e[0] == "response.output_item.added"
        ]
        assert added_items == ["function_call"]

        completed = [e[1] for e in events if e[0] == "response.completed"][0]
        assert [item["type"] for item in completed["response"]["output"]] == ["function_call"]
        assert [item["type"] for item in sc.current_output_items] == ["function_call"]

    @pytest.mark.asyncio
    async def test_empty_text_delta_between_tool_rounds_is_dropped(self, ctx):
        sc = StreamConverter(ctx)
        events = []

        async for evt in sc.process_chunk(make_chunk({"content": "First round output"})):
            events.append(evt)

        sc.prepare_for_next_round()

        async for evt in sc.process_chunk(make_chunk({"role": "assistant", "content": ""})):
            events.append(evt)

        async for evt in sc.process_chunk(make_chunk({
            "tool_calls": [{
                "index": 0,
                "id": "call_2",
                "type": "function",
                "function": {"name": "exec_command", "arguments": '{"cmd":"ls"}'},
            }]
        }, finish_reason="tool_calls")):
            events.append(evt)

        async for evt in sc.finalize():
            events.append(evt)

        completed = [e[1] for e in events if e[0] == "response.completed"][0]
        output_types = [item["type"] for item in completed["response"]["output"]]
        # the empty frame from the second round must not add a message item
        assert output_types == ["function_call"]

    @pytest.mark.asyncio
    async def test_empty_reasoning_delta_creates_no_item(self, ctx):
        sc = StreamConverter(ctx)
        events = []

        async for evt in sc.process_chunk(make_chunk({"reasoning_content": ""})):
            events.append(evt)

        async for evt in sc.finalize():
            events.append(evt)

        event_types = [e[0] for e in events]
        assert not [t for t in event_types if t.startswith("response.reasoning")]
        assert "response.output_item.added" not in event_types

        completed = [e[1] for e in events if e[0] == "response.completed"][0]
        assert completed["response"]["output"] == []


class TestReasoningSummaryStreaming:
    @pytest.mark.asyncio
    async def test_reasoning_streams_official_summary_events(self, ctx):
        sc = StreamConverter(ctx)
        events = []

        async for evt in sc.process_chunk(make_chunk({"reasoning_content": "Let me "})):
            events.append(evt)
        async for evt in sc.process_chunk(make_chunk({"reasoning_content": "think."})):
            events.append(evt)
        async for evt in sc.process_chunk(make_chunk({"content": "Answer"})):
            events.append(evt)
        async for evt in sc.finalize():
            events.append(evt)

        event_types = [e[0] for e in events]
        # the non-standard event name is gone
        assert "response.reasoning.delta" not in event_types
        assert [t for t in event_types if t.startswith("response.reasoning")] == [
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_text.delta",
            "response.reasoning_summary_text.delta",
            "response.reasoning_summary_text.done",
            "response.reasoning_summary_part.done",
        ]

        added = [e[1] for e in events if e[0] == "response.output_item.added"]
        assert added[0]["item"]["type"] == "reasoning"
        assert added[0]["item"]["summary"] == []

        part_added = [
            e[1] for e in events if e[0] == "response.reasoning_summary_part.added"
        ][0]
        assert part_added["summary_index"] == 0
        assert part_added["part"] == {"type": "summary_text", "text": ""}

        deltas = [
            e[1] for e in events if e[0] == "response.reasoning_summary_text.delta"
        ]
        assert [d["delta"] for d in deltas] == ["Let me ", "think."]
        assert all(d["summary_index"] == 0 for d in deltas)

        text_done = [
            e[1] for e in events if e[0] == "response.reasoning_summary_text.done"
        ][0]
        assert text_done["text"] == "Let me think."
        part_done = [
            e[1] for e in events if e[0] == "response.reasoning_summary_part.done"
        ][0]
        assert part_done["part"] == {"type": "summary_text", "text": "Let me think."}

        # done events are emitted before output_item.done for the same item
        assert event_types.index("response.reasoning_summary_text.done") < event_types.index(
            "response.output_item.done"
        )

        completed = [e[1] for e in events if e[0] == "response.completed"][0]
        output = completed["response"]["output"]
        assert [item["type"] for item in output] == ["reasoning", "message"]
        assert output[0]["summary"] == [{"type": "summary_text", "text": "Let me think."}]
        assert output[0]["status"] == "completed"
        assert output[1]["content"][0]["text"] == "Answer"

        seq_nums = _sequence_numbers(events)
        assert seq_nums == sorted(seq_nums)
        assert len(seq_nums) == len(set(seq_nums))
