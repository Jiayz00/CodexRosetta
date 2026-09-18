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
        - Anything else: fail fast instead of silently dropping the tool
        """
        if not responses_tools:
            return []

        # Collect top-level function names first: flattened namespace tools must
        # not collide with them no matter which order the client declared them.
        for tool in responses_tools:
            if not isinstance(tool, dict):
                continue
            if tool.get("type", "function") == "function" and tool.get("name"):
                context.top_level_tool_names.add(tool["name"])

        chat_tools: list[dict[str, Any]] = []

        for index, tool in enumerate(responses_tools):
            tool_type = tool.get("type", "function")
            if tool_type == "namespace":
                chat_tools.extend(self._convert_namespace_tools(tool, context))
                continue
            converted = self._convert_tool(tool, tool_type, context, index)
            if converted is not None:
                chat_tools.append(converted)

        return chat_tools

    def _convert_tool(
        self,
        tool: dict[str, Any],
        tool_type: str,
        context: ConversionContext,
        index: int = 0,
    ) -> dict[str, Any] | None:
        if tool_type == "function":
            return self._convert_function_tool(tool)

        elif tool_type in BUILTIN_TOOL_TYPES:
            return self._convert_builtin_tool(tool, tool_type, context)

        elif tool_type == "custom":
            return self._convert_custom_tool(tool, context)

        raise UnsupportedParameterError(
            f"tools[{index}].type",
            f"Unsupported tool type: '{tool_type}'.",
        )

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

    def _convert_namespace_tools(
        self, tool: dict[str, Any], context: ConversionContext
    ) -> list[dict[str, Any]]:
        """Flatten a Responses ``namespace`` tool group into flat functions.

        Chat Completions has no namespaced tools, so every nested function is
        exposed as ``<namespace>__<tool>`` and recorded on the context; the
        response converters put the ``namespace`` field back on the call item.
        """
        namespace = tool.get("name") or ""
        namespace_description = tool.get("description") or ""

        converted: list[dict[str, Any]] = []
        for nested in tool.get("tools") or []:
            if not isinstance(nested, dict):
                continue
            if nested.get("type", "function") != "function":
                continue
            tool_name = nested.get("name") or ""
            if not tool_name:
                continue

            flat_name = self._unique_flat_name(namespace, tool_name, context)
            context.register_namespace_tool(flat_name, namespace, tool_name)

            description = nested.get("description") or namespace_description or ""
            func_def: dict[str, Any] = {"name": flat_name}
            if description:
                func_def["description"] = (
                    f"[{namespace}] {description}" if namespace else description
                )
            if "parameters" in nested:
                func_def["parameters"] = nested["parameters"]
            if "strict" in nested:
                func_def["strict"] = nested["strict"]

            converted.append({"type": "function", "function": func_def})

        return converted

    def _unique_flat_name(
        self, namespace: str, tool_name: str, context: ConversionContext
    ) -> str:
        """Return a flat upstream name that no other tool has claimed."""
        base = f"{namespace}__{tool_name}" if namespace else tool_name
        candidate = base
        suffix = 2
        while (
            candidate in context.namespace_tools
            or candidate in context.top_level_tool_names
        ):
            candidate = f"{base}_{suffix}"
            suffix += 1
        return candidate

    def _convert_custom_tool(
        self, tool: dict[str, Any], context: ConversionContext
    ) -> dict[str, Any]:
        """Convert a custom (freeform) tool to a single-string function.

        Chat Completions only supports ``type: "function"``; the previous
        implementation returned ``type: "custom"`` which is not understood by
        most upstream providers (e.g. litellm, vLLM) and causes 400 errors.

        Freeform grammars cannot be expressed, so the tool is exposed as one
        ``input`` string parameter; the response converters turn the resulting
        call back into a ``custom_tool_call`` item.
        """
        custom = tool.get("custom", tool)
        name = custom.get("name", "custom_tool")
        context.custom_tool_names.add(name)

        description = custom.get("description", "") or ""
        if custom.get("format"):
            description = (
                f"{description}\nPass the tool input verbatim as the `input` string."
            ).strip()

        func_def = {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Raw tool input, passed to the tool verbatim.",
                    }
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
            namespace = tool_choice.get("namespace") or ""
            if namespace and context is not None:
                # Client sends the bare tool name plus its namespace; the tool
                # list upstream was flattened, so the forced choice must use the
                # same flattened name or the upstream rejects the request.
                name = context.flatten_namespace_tool(namespace, name)
            return {
                "type": "function",
                "function": {"name": self._forward_name(name, context)},
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

    @staticmethod
    def _forward_name(name: str, context: ConversionContext) -> str:
        """Map a client-visible function tool_choice name to the upstream name."""
        if name in context.namespace_tools:
            return name
        if "." in name:
            namespace, _, tool_name = name.rpartition(".")
            return context.flatten_namespace_tool(namespace, tool_name)
        return name
