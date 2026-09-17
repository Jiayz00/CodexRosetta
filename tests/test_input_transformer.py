import pytest

from codex_rosetta.converters.content_transformer import ContentTransformer
from codex_rosetta.converters.input_transformer import InputTransformer
from codex_rosetta.converters.request_converter import RequestConverter


@pytest.fixture
def tf():
    return InputTransformer(ContentTransformer())


class TestSimpleInput:
    @pytest.mark.asyncio
    async def test_string_input(self, tf):
        result = tf.transform_input("Hello")
        assert result == [{"role": "user", "content": "Hello"}]

    @pytest.mark.asyncio
    async def test_string_with_instructions(self, tf):
        result = tf.transform_input("Hello", instructions="Be helpful")
        assert result[0] == {"role": "system", "content": "Be helpful"}
        assert result[1] == {"role": "user", "content": "Hello"}

    @pytest.mark.asyncio
    async def test_instructions_only(self, tf):
        result = tf.transform_input("Hello", instructions="Be helpful")
        assert len(result) == 2
        assert result[0]["role"] == "system"


class TestMessageItems:
    @pytest.mark.asyncio
    async def test_user_message_with_text(self, tf):
        result = tf.transform_input([
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hi"}]}
        ])
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hi"

    @pytest.mark.asyncio
    async def test_developer_role_mapped_to_system(self, tf):
        result = tf.transform_input([
            {"type": "message", "role": "developer", "content": "Be precise"}
        ])
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "Be precise"

    @pytest.mark.asyncio
    async def test_system_message(self, tf):
        result = tf.transform_input([
            {"type": "message", "role": "system", "content": "Be helpful"}
        ])
        assert result[0]["role"] == "system"

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, tf):
        result = tf.transform_input([
            {"type": "message", "role": "user", "content": "Hi"},
            {"type": "message", "role": "assistant", "content": "Hello!"},
            {"type": "message", "role": "user", "content": "How are you?"},
        ])
        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[2]["role"] == "user"


class TestFunctionCallGrouping:
    @pytest.mark.asyncio
    async def test_assistant_with_function_calls(self, tf):
        result = tf.transform_input([
            {"type": "message", "role": "user", "content": "Weather in SF?"},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Let me check."}]},
            {"type": "function_call", "name": "get_weather", "call_id": "call_abc", "arguments": '{"location":"SF"}'},
        ])
        # The unanswered call cannot be sent to Chat Completions, but the
        # assistant text is preserved.
        assert len(result) == 2
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "Let me check."
        assert "tool_calls" not in result[1]

    @pytest.mark.asyncio
    async def test_orphan_function_call_output_becomes_user_message(self, tf):
        delegation = "<codex_delegation>\n  <source_thread_id>parent</source_thread_id>\n  <input>do work</input>\n</codex_delegation>"
        result = tf.transform_input([
            {"type": "message", "role": "developer", "content": "context"},
            {"type": "message", "role": "user", "content": "instructions"},
            {
                "type": "function_call_output",
                "id": "fco_01a0b026-4af2-7950-bb2f-d276b77900c1",
                "name": "create_thread",
                "namespace": "codex_app",
                "output": delegation,
            },
        ])
        assert result[-1] == {"role": "user", "content": delegation}
        assert all(message.get("role") != "tool" for message in result)

    @pytest.mark.asyncio
    async def test_orphan_custom_tool_output_becomes_user_message(self, tf):
        result = tf.transform_input([
            {
                "type": "custom_tool_call_output",
                "id": "ctco_1",
                "name": "apply_patch",
                "output": "orphan output",
            },
        ])
        assert result == [{"role": "user", "content": "orphan output"}]

    @pytest.mark.asyncio
    async def test_orphan_tool_call_is_removed_before_next_message(self, tf):
        result = tf.transform_input([
            {"type": "message", "role": "assistant", "content": None},
            {"type": "function_call", "name": "get_weather", "call_id": "call_abc", "arguments": "{}"},
            {"type": "message", "role": "user", "content": "continue"},
        ])
        assert result == [{"role": "user", "content": "continue"}]

    @pytest.mark.asyncio
    async def test_partial_tool_results_remove_only_orphan_tool_call(self, tf):
        result = tf.transform_input([
            {"type": "function_call", "name": "first", "call_id": "call_1", "arguments": "{}"},
            {"type": "function_call", "name": "second", "call_id": "call_2", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
        ])
        assert len(result) == 2
        assert [tc["id"] for tc in result[0]["tool_calls"]] == ["call_1"]
        assert result[1]["tool_call_id"] == "call_1"

    @pytest.mark.asyncio
    async def test_function_call_output(self, tf):
        result = tf.transform_input([
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Let me check."}]},
            {"type": "function_call", "name": "get_weather", "call_id": "call_abc", "arguments": '{"location":"SF"}'},
            {"type": "function_call_output", "call_id": "call_abc", "output": '{"temp":72}'},
        ])
        # assistant msg + tool msg
        assert len(result) == 2
        assert result[0]["role"] == "assistant"
        assert result[0]["tool_calls"][0]["id"] == "call_abc"
        assert result[1]["role"] == "tool"
        assert result[1]["tool_call_id"] == "call_abc"
        assert result[1]["content"] == '{"temp":72}'

    @pytest.mark.asyncio
    async def test_assistant_with_null_content_and_tool_calls(self, tf):
        result = tf.transform_input([
            {"type": "message", "role": "assistant", "content": None},
            {"type": "function_call", "name": "get_weather", "call_id": "call_1", "arguments": '{}'},
        ])
        assert result == []

    @pytest.mark.asyncio
    async def test_parallel_tool_calls(self, tf):
        result = tf.transform_input([
            {"type": "message", "role": "assistant", "content": None},
            {"type": "function_call", "name": "get_weather", "call_id": "call_1", "arguments": '{"location":"SF"}'},
            {"type": "function_call", "name": "get_weather", "call_id": "call_2", "arguments": '{"location":"NYC"}'},
        ])
        assert result == []

    @pytest.mark.asyncio
    async def test_full_tool_call_lifecycle(self, tf):
        """Complete multi-turn: user -> assistant+tool_call -> tool_result -> user followup"""
        result = tf.transform_input([
            {"type": "message", "role": "user", "content": "Weather in SF?"},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Checking..."}]},
            {"type": "function_call", "name": "get_weather", "call_id": "call_abc", "arguments": '{"location":"SF"}'},
            {"type": "function_call_output", "call_id": "call_abc", "output": '{"temp":72}'},
            {"type": "message", "role": "user", "content": "Thanks!"},
        ])
        # user, assistant+tool_calls, tool, user
        assert len(result) == 4
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert "tool_calls" in result[1]
        assert result[2]["role"] == "tool"
        assert result[3]["role"] == "user"


class TestFunctionCallOutputWithListContent:
    @pytest.mark.asyncio
    async def test_function_call_output_with_list_content(self, tf):
        result = tf.transform_input([
            {"type": "function_call", "name": "get_data", "call_id": "call_1", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": [
                {"type": "output_text", "text": "line1"},
                {"type": "output_text", "text": "line2"},
            ]},
        ])
        assert result[0]["role"] == "assistant"
        assert result[1]["role"] == "tool"
        assert "line1" in result[1]["content"]
        assert "line2" in result[1]["content"]


class TestHistoricalToolCallSanitization:
    @pytest.mark.asyncio
    async def test_request_converter_removes_historical_orphan_tool_call(self):
        chat_request, _ = await RequestConverter().convert(
            {"model": "gpt-4o", "input": "continue"},
            conversation_messages=[
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_orphan",
                            "type": "function",
                            "function": {"name": "missing", "arguments": "{}"},
                        }
                    ],
                }
            ],
        )
        assert all(not message.get("tool_calls") for message in chat_request["messages"])


class TestMixedContent:
    @pytest.mark.asyncio
    async def test_user_with_image(self, tf):
        result = tf.transform_input([
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "What is this?"},
                {"type": "input_image", "image_url": "https://img.png"},
            ]}
        ])
        assert result[0]["role"] == "user"
        content = result[0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"
