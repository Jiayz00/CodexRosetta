import pytest

from codex_rosetta.converters.request_converter import RequestConverter
from codex_rosetta.models.common import is_simulated_function


@pytest.fixture
def rc():
    return RequestConverter()


class TestSimpleRequest:
    @pytest.mark.asyncio
    async def test_string_input(self, rc):
        chat_req, ctx = await rc.convert({"model": "gpt-4o", "input": "Hello"})
        assert chat_req["model"] == "gpt-4o"
        assert chat_req["messages"][-1] == {"role": "user", "content": "Hello"}
        assert ctx.model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_instructions_becomes_system(self, rc):
        chat_req, ctx = await rc.convert({
            "model": "gpt-4o",
            "instructions": "Be helpful",
            "input": "Hi",
        })
        assert chat_req["messages"][0] == {"role": "system", "content": "Be helpful"}
        assert ctx.original_instructions == "Be helpful"


class TestParameterMapping:
    @pytest.mark.asyncio
    async def test_max_output_tokens(self, rc):
        chat_req, _ = await rc.convert({
            "model": "gpt-4o",
            "input": "Hi",
            "max_output_tokens": 100,
        })
        assert chat_req["max_completion_tokens"] == 100
        assert "max_output_tokens" not in chat_req

    @pytest.mark.asyncio
    async def test_temperature_passthrough(self, rc):
        chat_req, _ = await rc.convert({
            "model": "gpt-4o",
            "input": "Hi",
            "temperature": 0.7,
        })
        assert chat_req["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_top_p_passthrough(self, rc):
        chat_req, _ = await rc.convert({
            "model": "gpt-4o",
            "input": "Hi",
            "top_p": 0.9,
        })
        assert chat_req["top_p"] == 0.9

    @pytest.mark.asyncio
    async def test_stream_flag(self, rc):
        chat_req, ctx = await rc.convert({
            "model": "gpt-4o",
            "input": "Hi",
            "stream": True,
        })
        assert chat_req["stream"] is True
        assert ctx.had_streaming is True
        assert chat_req.get("stream_options", {}).get("include_usage") is True


class TestTextFormatMapping:
    @pytest.mark.asyncio
    async def test_json_schema_is_not_forwarded_as_response_format(self, rc):
        """`response_format: json_schema` is rejected by many Chat Completions
        upstreams, so the schema is enforced via the system prompt instead.
        See tests/test_chat_completions_compat.py for the full coverage."""
        text_format = {"type": "json_schema", "schema": {"type": "object"}}
        chat_req, ctx = await rc.convert({
            "model": "gpt-4o",
            "input": "Hi",
            "text": {"format": text_format},
        })
        assert "response_format" not in chat_req
        assert ctx.original_text_format == {"format": text_format}
        assert "JSON Schema" in chat_req["messages"][0]["content"]

    @pytest.mark.asyncio
    async def test_json_object_is_forwarded_as_response_format(self, rc):
        text_format = {"type": "json_object"}
        chat_req, ctx = await rc.convert({
            "model": "gpt-4o",
            "input": "Hi",
            "text": {"format": text_format},
        })
        assert chat_req["response_format"] == text_format
        assert ctx.original_text_format == {"format": text_format}


class TestReasoningMapping:
    @pytest.mark.asyncio
    async def test_reasoning_effort(self, rc):
        chat_req, _ = await rc.convert({
            "model": "o3",
            "input": "Think carefully",
            "reasoning": {"effort": "high"},
        })
        assert chat_req["reasoning_effort"] == "high"


class TestToolsInRequest:
    @pytest.mark.asyncio
    async def test_function_tools_converted(self, rc):
        chat_req, ctx = await rc.convert({
            "model": "gpt-4o",
            "input": "Hi",
            "tools": [
                {"type": "function", "name": "get_weather", "parameters": {"type": "object", "properties": {}}},
            ],
        })
        assert len(chat_req["tools"]) == 1
        assert chat_req["tools"][0]["type"] == "function"
        assert chat_req["tools"][0]["function"]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_builtin_tools_simulated(self, rc):
        chat_req, ctx = await rc.convert({
            "model": "gpt-4o",
            "input": "Search",
            "tools": [{"type": "web_search"}],
        })
        sim_name = chat_req["tools"][0]["function"]["name"]
        assert is_simulated_function(sim_name)
        assert ctx.get_original_tool_type(sim_name) == "web_search"


class TestPreviousResponseId:
    @pytest.mark.asyncio
    async def test_previous_response_id_prepends_messages(self, rc):
        conversation_messages = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]
        chat_req, ctx = await rc.convert(
            {"model": "gpt-4o", "input": "Follow up", "instructions": "Be helpful"},
            conversation_messages=conversation_messages,
        )
        # System + previous messages + new user message
        assert chat_req["messages"][0]["role"] == "system"
        assert chat_req["messages"][1]["content"] == "Previous question"
        assert chat_req["messages"][2]["content"] == "Previous answer"
        # New user message at end
        assert chat_req["messages"][-1]["content"] == "Follow up"


class TestMultiTurnInput:
    @pytest.mark.asyncio
    async def test_full_multi_turn(self, rc):
        chat_req, ctx = await rc.convert({
            "model": "gpt-4o",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Weather?"}]},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Checking..."}]},
                {"type": "function_call", "name": "get_weather", "call_id": "call_1", "arguments": '{"location":"SF"}'},
                {"type": "function_call_output", "call_id": "call_1", "output": '{"temp":72}'},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Thanks"}]},
            ],
        })
        # user, assistant+tool_calls, tool, user
        assert len(chat_req["messages"]) == 4
        assert chat_req["messages"][0]["role"] == "user"
        assert chat_req["messages"][1]["role"] == "assistant"
        assert "tool_calls" in chat_req["messages"][1]
        assert chat_req["messages"][2]["role"] == "tool"
        assert chat_req["messages"][3]["role"] == "user"
