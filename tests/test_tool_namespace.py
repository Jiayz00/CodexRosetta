"""Namespace tool groups, custom (freeform) tools, and GET /v1/models.

Codex sends `{"type": "namespace"}` tool groups for sub-agents and MCP servers
(`multi_agent_v1`, `mcp__node_repl`, `mcp__cua_repl`). Chat Completions cannot
express them, so they are flattened to `<namespace>__<tool>` upstream and the
`namespace` field is restored on the call item the client receives.
"""

import json

from codex_rosetta.converters.content_transformer import ContentTransformer
from codex_rosetta.converters.input_transformer import InputTransformer
from codex_rosetta.converters.request_converter import RequestConverter
from codex_rosetta.converters.response_converter import ResponseConverter
from codex_rosetta.converters.stream_converter import StreamConverter
from codex_rosetta.converters.tool_transformer import ToolTransformer
from codex_rosetta.models.common import ConversionContext


def make_namespace_tools() -> list[dict]:
    return [
        {
            "type": "namespace",
            "name": "multi_agent_v1",
            "description": "Tools for spawning and managing sub-agents.",
            "tools": [
                {
                    "type": "function",
                    "name": "spawn_agent",
                    "description": "Spawn a sub-agent.",
                    "parameters": {
                        "type": "object",
                        "properties": {"prompt": {"type": "string"}},
                        "required": ["prompt"],
                    },
                },
                {
                    "type": "function",
                    "name": "close_agent",
                    "description": "Close an agent.",
                    "parameters": {
                        "type": "object",
                        "properties": {"target": {"type": "string"}},
                        "required": ["target"],
                    },
                },
            ],
        },
        {
            "type": "namespace",
            "name": "mcp__node_repl",
            "tools": [
                {"type": "function", "name": "js", "parameters": {"type": "object", "properties": {}}},
                {"type": "function", "name": "js_reset", "parameters": {"type": "object", "properties": {}}},
            ],
        },
        {
            "type": "namespace",
            "name": "mcp__cua_repl",
            "tools": [
                {"type": "function", "name": "js", "parameters": {"type": "object", "properties": {}}},
            ],
        },
    ]


def make_context() -> ConversionContext:
    return ConversionContext(response_id="resp_ns", model="deepseek-flash")


def make_chunk(delta: dict, finish_reason=None) -> str:
    chunk = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1746000000,
        "model": "deepseek-flash",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def chat_response_with_tool_call(name: str, arguments: str) -> dict:
    return {
        "id": "chatcmpl-ns",
        "object": "chat.completion",
        "created": 1746000000,
        "model": "deepseek-flash",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


class TestNamespaceFlattening:
    def test_tools_are_flattened_with_namespace_prefix(self):
        transformer = ToolTransformer()
        ctx = make_context()

        tools = transformer.convert_tools(make_namespace_tools(), ctx)

        names = [t["function"]["name"] for t in tools]
        assert names == [
            "multi_agent_v1__spawn_agent",
            "multi_agent_v1__close_agent",
            "mcp__node_repl__js",
            "mcp__node_repl__js_reset",
            "mcp__cua_repl__js",
        ]
        assert all(t["type"] == "function" for t in tools)

        spawn = tools[0]["function"]
        assert spawn["description"].startswith("[multi_agent_v1] ")
        assert spawn["parameters"]["required"] == ["prompt"]

        assert ctx.namespace_tools["multi_agent_v1__spawn_agent"] == (
            "multi_agent_v1",
            "spawn_agent",
        )

    def test_same_tool_name_in_two_namespaces_stays_distinct(self):
        transformer = ToolTransformer()
        ctx = make_context()

        tools = transformer.convert_tools(make_namespace_tools(), ctx)

        names = [t["function"]["name"] for t in tools]
        assert "mcp__node_repl__js" in names
        assert "mcp__cua_repl__js" in names
        assert ctx.resolve_namespace_tool("mcp__node_repl__js") == ("mcp__node_repl", "js")
        assert ctx.resolve_namespace_tool("mcp__cua_repl__js") == ("mcp__cua_repl", "js")
        # Ambiguous bare name must not be mapped
        assert ctx.resolve_namespace_tool("js") is None

    def test_flattened_name_does_not_shadow_top_level_function(self):
        transformer = ToolTransformer()
        ctx = make_context()

        tools = transformer.convert_tools(
            [
                {
                    "type": "namespace",
                    "name": "multi_agent_v1",
                    "tools": [
                        {
                            "type": "function",
                            "name": "close_agent",
                            "parameters": {"type": "object", "properties": {}},
                        }
                    ],
                },
                {
                    "type": "function",
                    "name": "multi_agent_v1__close_agent",
                    "parameters": {"type": "object", "properties": {}},
                },
            ],
            ctx,
        )

        names = [t["function"]["name"] for t in tools]
        assert names == ["multi_agent_v1__close_agent_2", "multi_agent_v1__close_agent"]

    def test_resolve_accepts_dotted_and_unique_bare_alias(self):
        ctx = make_context()
        ctx.register_namespace_tool("multi_agent_v1__close_agent", "multi_agent_v1", "close_agent")
        ctx.register_namespace_tool("mcp__node_repl__js", "mcp__node_repl", "js")
        ctx.register_namespace_tool("mcp__cua_repl__js", "mcp__cua_repl", "js")
        ctx.top_level_tool_names.add("exec_command")

        assert ctx.resolve_namespace_tool("multi_agent_v1__close_agent") == (
            "multi_agent_v1",
            "close_agent",
        )
        assert ctx.resolve_namespace_tool("multi_agent_v1.close_agent") == (
            "multi_agent_v1",
            "close_agent",
        )
        assert ctx.resolve_namespace_tool("close_agent") == ("multi_agent_v1", "close_agent")
        assert ctx.resolve_namespace_tool("exec_command") is None
        assert ctx.resolve_namespace_tool("js") is None

    def test_custom_tool_exposed_as_single_string_function(self):
        transformer = ToolTransformer()
        ctx = make_context()

        tools = transformer.convert_tools(
            [
                {
                    "type": "custom",
                    "name": "apply_patch",
                    "description": "Apply a patch.",
                    "format": {"type": "grammar", "syntax": "lark", "definition": "start: ..."},
                }
            ],
            ctx,
        )

        func = tools[0]["function"]
        assert func["name"] == "apply_patch"
        assert func["parameters"]["required"] == ["input"]
        assert func["parameters"]["properties"]["input"]["type"] == "string"
        assert "apply_patch" in ctx.custom_tool_names


class TestInputReplay:
    def test_namespaced_function_call_replays_flattened_name(self):
        ctx = make_context()
        ctx.register_namespace_tool("multi_agent_v1__close_agent", "multi_agent_v1", "close_agent")
        transformer = InputTransformer(ContentTransformer())

        messages = transformer.transform_input(
            [
                {"type": "message", "role": "user", "content": "go"},
                {"type": "message", "role": "assistant", "content": "closing"},
                {
                    "type": "function_call",
                    "name": "close_agent",
                    "namespace": "multi_agent_v1",
                    "call_id": "call_1",
                    "arguments": "{}",
                },
                {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
            ],
            None,
            ctx,
        )

        assistant = [m for m in messages if m.get("tool_calls")][0]
        assert assistant["tool_calls"][0]["function"]["name"] == "multi_agent_v1__close_agent"

    def test_custom_tool_call_replays_as_single_string_function(self):
        transformer = InputTransformer(ContentTransformer())

        messages = transformer.transform_input(
            [
                {"type": "message", "role": "user", "content": "go"},
                {"type": "message", "role": "assistant", "content": "patching"},
                {
                    "type": "custom_tool_call",
                    "name": "apply_patch",
                    "call_id": "call_1",
                    "input": "*** Begin Patch",
                },
                {"type": "custom_tool_call_output", "call_id": "call_1", "output": "Done!"},
            ]
        )

        assistant = [m for m in messages if m.get("tool_calls")][0]
        tool_call = assistant["tool_calls"][0]
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "apply_patch"
        assert json.loads(tool_call["function"]["arguments"]) == {"input": "*** Begin Patch"}

    async def test_request_converter_maps_replay_to_flattened_name(self):
        converter = RequestConverter()

        chat_request, ctx = await converter.convert(
            {
                "model": "deepseek-flash",
                "tools": make_namespace_tools(),
                "input": [
                    {"type": "message", "role": "user", "content": "go"},
                    {"type": "message", "role": "assistant", "content": "spawning"},
                    {
                        "type": "function_call",
                        "name": "spawn_agent",
                        "namespace": "multi_agent_v1",
                        "call_id": "call_1",
                        "arguments": "{}",
                    },
                    {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
                ],
            }
        )

        tool_names = [t["function"]["name"] for t in chat_request["tools"]]
        assert "multi_agent_v1__spawn_agent" in tool_names

        assistant = [m for m in chat_request["messages"] if m.get("tool_calls")][0]
        assert assistant["tool_calls"][0]["function"]["name"] == "multi_agent_v1__spawn_agent"


class TestResponseConversion:
    def test_namespaced_call_item_carries_namespace(self):
        ctx = make_context()
        ctx.register_namespace_tool("multi_agent_v1__close_agent", "multi_agent_v1", "close_agent")
        converter = ResponseConverter()

        response = converter.convert(
            chat_response_with_tool_call("multi_agent_v1__close_agent", "{\"target\": \"a1\"}"),
            ctx,
        )

        item = [i for i in response["output"] if i["type"] == "function_call"][0]
        assert item["name"] == "close_agent"
        assert item["namespace"] == "multi_agent_v1"
        assert item["arguments"] == "{\"target\": \"a1\"}"

    def test_custom_tool_call_item_is_restored(self):
        ctx = make_context()
        ctx.custom_tool_names.add("apply_patch")
        converter = ResponseConverter()

        response = converter.convert(
            chat_response_with_tool_call("apply_patch", json.dumps({"input": "*** Begin Patch"})),
            ctx,
        )

        item = [i for i in response["output"] if i["type"] == "custom_tool_call"][0]
        assert item["name"] == "apply_patch"
        assert item["input"] == "*** Begin Patch"

    def test_unmapped_tool_call_stays_plain_function_call(self):
        ctx = make_context()
        converter = ResponseConverter()

        response = converter.convert(
            chat_response_with_tool_call("exec_command", "{}"), ctx
        )

        item = [i for i in response["output"] if i["type"] == "function_call"][0]
        assert item["name"] == "exec_command"
        assert "namespace" not in item


class TestStreamConversion:
    async def test_namespaced_stream_item_carries_namespace(self):
        ctx = make_context()
        ctx.register_namespace_tool("multi_agent_v1__close_agent", "multi_agent_v1", "close_agent")
        sc = StreamConverter(ctx)
        events = []

        async for evt in sc.process_chunk(
            make_chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "function": {"name": "multi_agent_v1__close_agent", "arguments": ""},
                        }
                    ]
                }
            )
        ):
            events.append(evt)

        async for evt in sc.process_chunk(
            make_chunk(
                {
                    "tool_calls": [
                        {"index": 0, "function": {"arguments": "{\"target\": \"a1\"}"}}
                    ]
                },
                finish_reason="tool_calls",
            )
        ):
            events.append(evt)

        async for evt in sc.finalize():
            events.append(evt)

        added = [e[1]["item"] for e in events if e[0] == "response.output_item.added"][0]
        assert added["name"] == "close_agent"
        assert added["namespace"] == "multi_agent_v1"

        done = [e[1]["item"] for e in events if e[0] == "response.output_item.done"][0]
        assert done["name"] == "close_agent"
        assert done["namespace"] == "multi_agent_v1"

        completed = [e[1]["response"] for e in events if e[0] == "response.completed"][0]
        output = [i for i in completed["output"] if i["type"] == "function_call"][0]
        assert output["name"] == "close_agent"
        assert output["namespace"] == "multi_agent_v1"

    async def test_custom_tool_stream_emits_custom_input_events(self):
        ctx = make_context()
        ctx.custom_tool_names.add("apply_patch")
        sc = StreamConverter(ctx)
        events = []

        async for evt in sc.process_chunk(
            make_chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "function": {"name": "apply_patch", "arguments": '{"input": "he'},
                        }
                    ]
                }
            )
        ):
            events.append(evt)

        async for evt in sc.process_chunk(
            make_chunk(
                {"tool_calls": [{"index": 0, "function": {"arguments": 'llo"}'}}]},
                finish_reason="tool_calls",
            )
        ):
            events.append(evt)

        async for evt in sc.finalize():
            events.append(evt)

        types = [e[0] for e in events]
        assert "response.function_call_arguments.delta" not in types
        assert "response.custom_tool_call_input.delta" in types

        delta = [e[1]["delta"] for e in events if e[0] == "response.custom_tool_call_input.delta"][0]
        assert delta == "hello"

        done = [e[1] for e in events if e[0] == "response.custom_tool_call_input.done"][0]
        assert done["input"] == "hello"

        item = [e[1]["item"] for e in events if e[0] == "response.output_item.done"][0]
        assert item["type"] == "custom_tool_call"
        assert item["input"] == "hello"

        completed = [e[1]["response"] for e in events if e[0] == "response.completed"][0]
        output = [i for i in completed["output"] if i["type"] == "custom_tool_call"][0]
        assert output["input"] == "hello"


class TestModelsEndpoint:
    async def test_models_endpoint_lists_configured_model(self, monkeypatch):
        from codex_rosetta.api import router as router_module

        class FakeSettings:
            MODELS_MODEL_ID = "deepseek-flash"

        monkeypatch.setattr(router_module, "get_settings", lambda: FakeSettings())

        payload = await router_module.list_models()

        assert payload == {
            "object": "list",
            "data": [
                {"id": "deepseek-flash", "object": "model", "owned_by": "codex-rosetta"}
            ],
        }

    async def test_models_endpoint_without_configured_model(self, monkeypatch):
        from codex_rosetta.api import router as router_module

        class FakeSettings:
            MODELS_MODEL_ID = ""

        monkeypatch.setattr(router_module, "get_settings", lambda: FakeSettings())

        payload = await router_module.list_models()

        assert payload == {"object": "list", "data": []}
