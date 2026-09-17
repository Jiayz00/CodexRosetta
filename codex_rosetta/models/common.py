from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# Prefix for simulated built-in tool function names
ROSETTA_TOOL_PREFIX = "__rosetta_"

# Mapping from Responses API built-in tool types to their simulated function names
BUILTIN_TOOL_TYPES = {
    "web_search",
    "web_search_2025_08_26",
    "file_search",
    "computer_use_preview",
    "computer",
    "code_interpreter",
    "image_generation",
}


def make_simulated_function_name(original_type: str) -> str:
    return f"{ROSETTA_TOOL_PREFIX}{original_type}"


def is_simulated_function(name: str) -> bool:
    return name.startswith(ROSETTA_TOOL_PREFIX)


def extract_original_type(simulated_name: str) -> str:
    if not is_simulated_function(simulated_name):
        return simulated_name
    return simulated_name[len(ROSETTA_TOOL_PREFIX):]


def extract_custom_tool_input(arguments: str) -> str:
    """Return the freeform input string of a custom tool call.

    Custom (freeform) tools are exposed to Chat Completions as a single
    ``input`` string parameter, so the upstream model answers with JSON such as
    ``{"input": "..."}``. Anything that is not that shape is passed through
    unchanged to avoid losing the model's output.
    """
    raw = arguments or ""
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
    if isinstance(parsed, dict) and isinstance(parsed.get("input"), str):
        return parsed["input"]
    return raw


@dataclass
class ConversionContext:
    """Metadata tracked during request conversion for use in response conversion."""

    response_id: str = ""
    model: str = ""
    original_tool_types: dict[str, str] = field(default_factory=dict)
    # simulated_function_name -> original_builtin_type
    original_instructions: str | None = None
    original_text_format: dict | None = None
    had_streaming: bool = False
    include_fields: list[str] = field(default_factory=list)
    # Requested include fields for null placeholder injection
    max_tool_calls: int | None = None
    truncation: str | None = None
    # "auto" or "disabled"
    # Flattened namespace tool name -> (namespace, tool name). Chat Completions
    # has no namespaced tools, so `{"type": "namespace"}` groups are exposed as
    # `<namespace>__<tool>` and restored on the way back.
    namespace_tools: dict[str, tuple[str, str]] = field(default_factory=dict)
    top_level_tool_names: set[str] = field(default_factory=set)
    custom_tool_names: set[str] = field(default_factory=set)

    def register_builtin_tool(self, simulated_name: str, original_type: str) -> None:
        self.original_tool_types[simulated_name] = original_type

    def get_original_tool_type(self, function_name: str) -> str | None:
        return self.original_tool_types.get(function_name)

    def register_namespace_tool(
        self, flat_name: str, namespace: str, tool_name: str
    ) -> None:
        """Remember a flattened namespace tool so responses can restore it."""
        self.namespace_tools[flat_name] = (namespace, tool_name)

    def flatten_namespace_tool(self, namespace: str, tool_name: str) -> str:
        """Map (namespace, tool name) back to the flattened upstream name."""
        for flat_name, pair in self.namespace_tools.items():
            if pair == (namespace, tool_name):
                return flat_name
        return tool_name

    def resolve_namespace_tool(self, function_name: str) -> tuple[str, str] | None:
        """Map an upstream function name back to (namespace, tool name).

        Accepts the flattened name we sent, plus two defensive forms some
        models produce: the dotted `<namespace>.<tool>` and a bare tool name
        when it is unambiguous. Top-level function names are never captured.
        """
        if function_name in self.namespace_tools:
            return self.namespace_tools[function_name]
        if function_name in self.top_level_tool_names:
            return None
        if "." in function_name:
            namespace, _, tool_name = function_name.rpartition(".")
            if (namespace, tool_name) in self.namespace_tools.values():
                return (namespace, tool_name)
        matches = [
            pair for pair in self.namespace_tools.values() if pair[1] == function_name
        ]
        if len(matches) == 1:
            return matches[0]
        return None
