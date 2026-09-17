from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ResponsesApiRequest(BaseModel):
    model: str
    input: str | list[Any]
    instructions: str | None = None
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: Any = None
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    max_tool_calls: int | None = None
    parallel_tool_calls: bool | None = None
    previous_response_id: str | None = None
    stream: bool = False
    store: bool | None = None
    metadata: dict[str, str] | None = None
    reasoning: dict[str, Any] | None = None
    text: dict[str, Any] | None = None
    truncation: Literal["auto", "disabled"] | None = None
    service_tier: str | None = None
    user: str | None = None
    safety_identifier: str | None = None
    prompt_cache_key: str | None = None
    top_logprobs: int | None = None

    model_config = {"extra": "allow"}


class ResponseOutputText(BaseModel):
    type: Literal["output_text"] = "output_text"
    text: str = ""
    annotations: list[Any] = Field(default_factory=list)


class ResponseRefusal(BaseModel):
    type: Literal["refusal"] = "refusal"
    refusal: str


class ResponseOutputMessage(BaseModel):
    type: Literal["message"] = "message"
    id: str
    role: Literal["assistant"] = "assistant"
    status: str = "completed"
    content: list[dict[str, Any]] = Field(default_factory=list)
    phase: str | None = None


class ResponseFunctionCall(BaseModel):
    type: Literal["function_call"] = "function_call"
    id: str
    call_id: str
    name: str
    arguments: str
    status: str = "completed"


class ResponseUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_tokens_details: dict[str, int] = Field(default_factory=lambda: {"cached_tokens": 0})
    output_tokens_details: dict[str, int] = Field(default_factory=lambda: {"reasoning_tokens": 0})


class ResponsesApiResponse(BaseModel):
    id: str
    object: Literal["response"] = "response"
    created_at: float
    completed_at: float | None = None
    model: str
    status: str = "completed"
    output: list[dict[str, Any]] = Field(default_factory=list)
    usage: ResponseUsage | None = None
    error: dict[str, Any] | None = None
    instructions: str | None = None
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: Any = None
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    metadata: dict[str, str] | None = None
    previous_response_id: str | None = None
    parallel_tool_calls: bool | None = None

    model_config = {"extra": "allow"}

# Request fields echoed back on the Response object, per the Responses API contract.
_ECHOED_REQUEST_FIELDS = (
    "instructions",
    "tools",
    "tool_choice",
    "temperature",
    "top_p",
    "max_output_tokens",
    "metadata",
    "previous_response_id",
    "parallel_tool_calls",
    "prompt_cache_key",
    "reasoning",
    "text",
    "truncation",
    "store",
    "service_tier",
    "safety_identifier",
    "user",
)


def build_response_envelope(
    context: Any,
    *,
    output: list[dict[str, Any]],
    status: str = "completed",
    usage: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
    incomplete_details: dict[str, Any] | None = None,
    created_at: float | None = None,
    completed_at: float | None = None,
) -> dict[str, Any]:
    """Build a Responses API envelope shared by the streaming and non-streaming paths.

    Always carries ``error``/``incomplete_details``/``completed_at``/``usage``/
    ``status``/``output`` so both paths produce the same shape, and echoes back
    every request field the gateway accepted.
    """
    from codex_rosetta.utils.id_generation import unix_timestamp

    response: dict[str, Any] = {
        "id": context.response_id,
        "object": "response",
        "created_at": created_at if created_at is not None else unix_timestamp(),
        "completed_at": completed_at,
        "model": context.model,
        "status": status,
        "output": output,
        "error": error,
        "incomplete_details": incomplete_details,
        "usage": usage,
    }

    original = getattr(context, "original_request", None) or {}
    for field_name in _ECHOED_REQUEST_FIELDS:
        value = original.get(field_name)
        if value is not None:
            response[field_name] = value

    if "instructions" not in response and context.original_instructions:
        response["instructions"] = context.original_instructions

    return response
