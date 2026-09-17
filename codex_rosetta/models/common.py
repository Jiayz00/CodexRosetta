from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

# Prefix for simulated built-in tool function names
ROSETTA_TOOL_PREFIX = "__rosetta_"

# Mapping from Responses API built-in tool types to their simulated function names
BUILTIN_TOOL_TYPES = {
    "web_search",
    "web_search_2025_08_26",
    "web_search_preview",
    "file_search",
    "computer_use_preview",
    "computer",
    "code_interpreter",
    "image_generation",
}

# Chat Completions requires every tool_call `arguments` value to be a JSON string.
EMPTY_ARGUMENTS = "{}"

# Upstream providers reject tool call ids shorter than this (seen: `call_m1`).
MIN_CALL_ID_LENGTH = 29


class UnsupportedParameterError(Exception):
    """A Responses API parameter cannot be faithfully mapped to Chat Completions.

    Raised instead of silently dropping or leaking the field upstream, so the
    client gets an actionable 400 rather than a confusing upstream error.
    """

    def __init__(self, param: str, message: str) -> None:
        super().__init__(message)
        self.param = param
        self.message = message

    def to_error_body(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": "invalid_request_error",
                "param": self.param,
                "code": "unsupported_parameter",
            }
        }


def normalize_call_id(raw: Any) -> str:
    """Return a Chat-Completions-safe tool call id.

    Short ids (``call_m1``) are rejected by strict upstreams, and empty ids break
    assistant/tool pairing, so both get expanded into a deterministic 29+ char id.
    """
    value = raw if isinstance(raw, str) else ""
    if len(value) >= MIN_CALL_ID_LENGTH:
        return value
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:24]
    return f"call_{digest}"


def derive_phase(has_tool_calls: bool) -> str:
    """Responses API assistant message phase for a turn.

    Official contract: ``commentary`` when the turn also emitted tool calls,
    ``final_answer`` when it is the user-visible answer.
    """
    return "commentary" if has_tool_calls else "final_answer"


def make_simulated_function_name(original_type: str) -> str:
    return f"{ROSETTA_TOOL_PREFIX}{original_type}"


def is_simulated_function(name: str) -> bool:
    return name.startswith(ROSETTA_TOOL_PREFIX)


def extract_original_type(simulated_name: str) -> str:
    if not is_simulated_function(simulated_name):
        return simulated_name
    return simulated_name[len(ROSETTA_TOOL_PREFIX):]


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
    original_request: dict[str, Any] = field(default_factory=dict)
    # Accepted original request fields, echoed back in the response envelope
    namespace_tools: dict[str, tuple[str, str]] = field(default_factory=dict)
    # flattened function name -> (namespace, bare tool name)
    custom_tool_names: set[str] = field(default_factory=set)
    # names of `type: "custom"` tools declared by the client

    def register_builtin_tool(self, simulated_name: str, original_type: str) -> None:
        self.original_tool_types[simulated_name] = original_type

    def get_original_tool_type(self, function_name: str) -> str | None:
        return self.original_tool_types.get(function_name)

    def register_namespace_tool(
        self, flat_name: str, namespace: str, tool_name: str
    ) -> None:
        self.namespace_tools[flat_name] = (namespace, tool_name)

    def resolve_forward_name(self, name: str) -> str:
        """Map a client-visible tool name back to the flattened upstream name."""
        if name in self.namespace_tools:
            return name
        if "." in name:
            namespace, _, tool = name.partition(".")
            flat = f"{namespace}__{tool}"
            if flat in self.namespace_tools:
                return flat
        for flat, (_, tool) in self.namespace_tools.items():
            if tool == name:
                return flat
        return name

    def resolve_output_name(self, name: str) -> tuple[str, str | None]:
        """Map an upstream function name back to (client-visible name, namespace)."""
        if name in self.namespace_tools:
            namespace, tool = self.namespace_tools[name]
            return tool, namespace
        if "." in name:
            namespace, _, tool = name.partition(".")
            if f"{namespace}__{tool}" in self.namespace_tools:
                return tool, namespace
        return name, None
