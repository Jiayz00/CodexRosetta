from __future__ import annotations

import copy
from typing import Any

from codex_rosetta.models.common import (
    BUILTIN_TOOL_TYPES,
    ROSETTA_TOOL_PREFIX,
    ConversionContext,
    UnsupportedParameterError,
    is_simulated_function,
    make_simulated_function_name,
)
from codex_rosetta.utils.logging import get_logger

logger = get_logger("tool_transformer")


class ToolTransformer:
    """Convert tool definitions between Responses API and Chat Completions formats."""

    def __init__(self, builtin_registry: Any = None) -> None:
        self._registry = builtin_registry

    def convert_tools(
        self, responses_tools: list[dict[str, Any]], context: ConversionContext
    ) -> list[dict[str, Any]]:
        """Convert Responses API tool definitions to Chat Completions format.

        - Function tools: flatten structure (name/parameters/description from nested `function` key)
        - Built-in tools: simulate as function tools with __rosetta_ prefix
        - Namespace groups: flatten to `<namespace>__<tool>` and record the mapping
        - Custom tools: flatten to a single freeform `input` string parameter
        """
        if not responses_tools:
            return []

        used_names = self._collect_declared_names(responses_tools)
        chat_tools: list[dict[str, Any]] = []

        for index, tool in enumerate(responses_tools):
            tool_type = tool.get("type", "function")
            converted = self._convert_tool(tool, tool_type, context, index, used_names)
            if isinstance(converted, list):
                chat_tools.extend(converted)
            else:
                chat_tools.append(converted)

        return chat_tools

    @staticmethod
    def _collect_declared_names(responses_tools: list[dict[str, Any]]) -> set[str]:
        names: set[str] = set()
        for tool in responses_tools:
            if not isinstance(tool, dict):
                continue
            tool_type = tool.get("type", "function")
            if tool_type == "function":
                names.add(tool.get("name", ""))
            elif tool_type == "custom":
                custom = tool.get("custom", tool)
                names.add(custom.get("name", ""))
            elif tool_type in BUILTIN_TOOL_TYPES:
                names.add(make_simulated_function_name(tool_type))
        return names

    def _convert_tool(
        self,
        tool: dict[str, Any],
        tool_type: str,
        context: ConversionContext,
        index: int = 0,
        used_names: set[str] | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        if tool_type == "function":
            return self._convert_function_tool(tool)

        if tool_type in BUILTIN_TOOL_TYPES:
            return self._convert_builtin_tool(tool, tool_type, context)

        if tool_type == "custom":
            return self._convert_custom_tool(tool, context)

        if tool_type == "namespace":
            return self._convert_namespace_tools(
                tool, context, index, used_names if used_names is not None else set()
            )

        raise UnsupportedParameterError(
            f"tools[{index}].type",
            f"Unsupported tool type: '{tool_type}'.",
        )

    def _convert_namespace_tools(
        self,
        tool: dict[str, Any],
        context: ConversionContext,
        index: int,
        used_names: set[str],
    ) -> list[dict[str, Any]]:
        """Flatten a Responses `namespace` tool group into Chat function tools.

        Chat Completions rejects dotted function names, so each tool becomes
        ``<namespace>__<tool>`` and the mapping is recorded on the context so
        responses can hand the client back the namespaced shape.
        """
        namespace = tool.get("name", "")
        inner_tools = tool.get("tools") or []
        converted: list[dict[str, Any]] = []

        for sub_index, sub in enumerate(inner_tools):
            if not isinstance(sub, dict) or sub.get("type", "function") != "function":
                raise UnsupportedParameterError(
                    f"tools[{index}].tools[{sub_index}].type",
                    "Only function tools are supported inside a namespace group.",
                )

            tool_name = sub.get("name", "")
            flat_name = self._uniquify(f"{namespace}__{tool_name}", used_names)
            context.register_namespace_tool(flat_name, namespace, tool_name)

            description = sub.get("description", "")
            if namespace:
                description = f"[{namespace}] {description}".strip()

            func_def: dict[str, Any] = {"name": flat_name, "description": description}
            if "parameters" in sub:
                func_def["parameters"] = sub["parameters"]
            if "strict" in sub:
                func_def["strict"] = sub["strict"]

            converted.append({"type": "function", "function": func_def})

        return converted

    @staticmethod
    def _uniquify(name: str, used_names: set[str]) -> str:
        if name not in used_names:
            used_names.add(name)
            return name
        suffix = 2
        while f"{name}_{suffix}" in used_names:
            suffix += 1
        unique = f"{name}_{suffix}"
        used_names.add(unique)
        return unique

    def _convert_function_tool(self, tool: dict[str, Any]) -> dict[str, Any]:
        """Responses function tool -> Chat Completions function tool.

        Responses: {type: "function", name: "...", parameters: {...}, description: "...", strict: true}
        ChatCC:    {type: "function", function: {name: "...", parameters: {...}, description: "...", strict: true}}
        """
        func_def: dict[str, Any] = {}

        if "name" in tool:
            func_def["name"] = tool["name"]
        if "parameters" in tool:
            func_def["parameters"] = tool["parameters"]
        if "description" in tool:
            func_def["description"] = tool["description"]
        if "strict" in tool:
            func_def["strict"] = tool["strict"]

        return {"type": "function", "function": func_def}

    def _convert_builtin_tool(
        self, tool: dict[str, Any], tool_type: str, context: ConversionContext
    ) -> dict[str, Any]:
        """Convert a built-in tool to a simulated function tool."""
        sim_name = make_simulated_function_name(tool_type)
        context.register_builtin_tool(sim_name, tool_type)

        func_def = self._get_builtin_function_definition(tool_type, tool)
        return {"type": "function", "function": func_def}

    def _get_builtin_function_definition(
        self, tool_type: str, tool: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate function definition for a simulated built-in tool."""
        definitions = {
            "web_search": {
                "name": make_simulated_function_name("web_search"),
                "description": "Search the web for information. Use this to find up-to-date information on any topic. Call this when you need to look something up online.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query string",
                        },
                        "search_context_size": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                            "description": "Amount of context window for search results",
                        },
                        "filters": {
                            "type": "object",
                            "properties": {
                                "allowed_domains": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Allowed domains for the search",
                                },
                            },
                        },
                        "user_location": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                                "country": {"type": "string"},
                                "region": {"type": "string"},
                                "timezone": {"type": "string"},
                            },
                        },
                    },
                    "required": ["query"],
                },
            },
            "web_search_2025_08_26": {
                "name": make_simulated_function_name("web_search_2025_08_26"),
                "description": "Search the web for information. Use this to find up-to-date information on any topic. Call this when you need to look something up online.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query string",
                        },
                        "search_context_size": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                            "description": "Amount of context window for search results",
                        },
                        "filters": {
                            "type": "object",
                            "properties": {
                                "allowed_domains": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Allowed domains for the search",
                                },
                            },
                        },
                        "user_location": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                                "country": {"type": "string"},
                                "region": {"type": "string"},
                                "timezone": {"type": "string"},
                            },
                        },
                    },
                    "required": ["query"],
                },
            },
            "file_search": {
                "name": make_simulated_function_name("file_search"),
                "description": "Search through files and documents. Simulates the built-in file_search tool.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"},
                        "max_num_results": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                },
            },
            "computer_use_preview": {
                "name": make_simulated_function_name("computer_use_preview"),
                "description": "Use a computer interface (click, type, scroll, screenshot). Simulates the built-in computer_use tool.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "object",
                            "description": "The computer action to perform",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["click", "double_click", "drag", "type", "scroll", "screenshot", "wait", "move"],
                                },
                                "x": {"type": "integer"},
                                "y": {"type": "integer"},
                                "text": {"type": "string"},
                                "button": {"type": "string", "enum": ["left", "right", "middle"]},
                                "direction": {"type": "string", "enum": ["up", "down"]},
                            },
                            "required": ["type"],
                        },
                    },
                    "required": ["action"],
                },
            },
            "computer": {
                "name": make_simulated_function_name("computer"),
                "description": "Use a computer interface. Simulates the built-in computer tool.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "object",
                            "description": "The computer action to perform",
                            "properties": {
                                "type": {"type": "string"},
                            },
                            "required": ["type"],
                        },
                    },
                    "required": ["action"],
                },
            },
            "code_interpreter": {
                "name": make_simulated_function_name("code_interpreter"),
                "description": "Execute code in an interpreter. Simulates the built-in code_interpreter tool.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "The code to execute"},
                        "language": {"type": "string", "description": "Programming language"},
                    },
                    "required": ["code"],
                },
            },
            "image_generation": {
                "name": make_simulated_function_name("image_generation"),
                "description": "Generate or edit images. Simulates the built-in image_generation tool.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "Description of the image to generate"},
                        "size": {
                            "type": "string",
                            "enum": ["1024x1024", "1024x1536", "1536x1024", "auto"],
                            "default": "auto",
                        },
                        "quality": {
                            "type": "string",
                            "enum": ["low", "medium", "high", "auto"],
                            "default": "auto",
                        },
                    },
                    "required": ["prompt"],
                },
            },
        }

        if tool_type in definitions:
            return copy.deepcopy(definitions[tool_type])

        # Generic fallback
        return {
            "name": make_simulated_function_name(tool_type),
            "description": f"Simulates the built-in {tool_type} tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": f"Input for {tool_type}"},
                },
            },
        }

    def _convert_custom_tool(
        self, tool: dict[str, Any], context: ConversionContext
    ) -> dict[str, Any]:
        """Convert a custom (freeform) tool to a Chat Completions function tool.

        Chat Completions only supports ``type: "function"``; the previous
        implementation returned ``type: "custom"`` which is not understood by
        most upstream providers (e.g. litellm, vLLM) and causes 400 errors.
        The freeform payload travels in a single ``input`` string argument.
        """
        custom = tool.get("custom", tool)
        name = custom.get("name", "custom_tool")
        context.custom_tool_names.add(name)
        func_def = {
            "name": name,
            "description": custom.get("description", ""),
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Freeform text payload for this tool.",
                    },
                },
                "required": ["input"],
            },
        }
        return {"type": "function", "function": func_def}

    def convert_tool_choice(
        self, tool_choice: Any, context: ConversionContext
    ) -> Any:
        """Convert tool_choice between Responses and Chat Completions shapes.

        Responses: {"type": "function", "name": "..."} / {"type": "custom", ...}
                   {"type": "web_search", ...} / "auto" | "none" | "required"
        ChatCC:    {"type": "function", "function": {"name": "..."}}

        Anything that cannot be pinned down is downgraded to "auto" (with a
        warning) rather than forwarded in a shape the upstream would reject.
        """
        if tool_choice is None:
            return None

        if isinstance(tool_choice, str):
            if tool_choice in ("auto", "none", "required"):
                return tool_choice
            logger.warning("tool_choice_unmappable", tool_choice=tool_choice)
            return "auto"

        if not isinstance(tool_choice, dict):
            logger.warning("tool_choice_unmappable", tool_choice=repr(tool_choice))
            return "auto"

        tc_type = tool_choice.get("type", "")

        if tc_type == "function":
            name = tool_choice.get("name", "")
            return {
                "type": "function",
                "function": {"name": context.resolve_forward_name(name)},
            }

        if tc_type == "custom":
            custom = tool_choice.get("custom", tool_choice)
            return {
                "type": "function",
                "function": {"name": custom.get("name", "")},
            }

        if tc_type in BUILTIN_TOOL_TYPES:
            return {
                "type": "function",
                "function": {"name": make_simulated_function_name(tc_type)},
            }

        logger.warning("tool_choice_unmappable", tool_choice=tc_type)
        return "auto"
