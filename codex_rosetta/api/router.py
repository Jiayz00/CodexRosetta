from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from codex_rosetta.api.dependencies import get_conversation_store, get_upstream_client
from codex_rosetta.audit.logger import NoOpAuditLogger
from codex_rosetta.config import get_settings
from codex_rosetta.converters.request_converter import RequestConverter
from codex_rosetta.converters.response_converter import ResponseConverter
from codex_rosetta.converters.stream_converter import StreamConverter
from codex_rosetta.converters.responses_mux import (
    FORCE_ANSWER_INSTRUCTION,
    INTERNAL_SEARCH_NAME,
    ResponsesStreamMux,
    SearchCallState,
    build_web_search_item,
    is_internal_search_item,
    normalize_responses_request,
    parse_arguments,
    strip_search_function_tool,
    to_search_tool_call,
)
from codex_rosetta.models.common import (
    ROSETTA_TOOL_PREFIX,
    UnsupportedParameterError,
    extract_original_type,
    is_simulated_function,
)
from codex_rosetta.search.base import QUERY_ERROR_KINDS, SearchProvider
from codex_rosetta.search.formatter import (
    format_search_results,
    format_search_unavailable,
)
from codex_rosetta.search.pool import (
    DEFAULT_POOL_FILE,
    SearchProviderPool,
    get_search_pool,
)
from codex_rosetta.utils.id_generation import generate_item_id, generate_response_id
from codex_rosetta.utils.logging import get_logger
from codex_rosetta.utils.sse import format_sse_event, parse_upstream_sse_stream

logger = get_logger("router")

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/v1/models")
async def list_models() -> dict[str, Any]:
    """Minimal model list.

    Codex probes `GET {base_url}/models` for reachability and treats a 404 as a
    misconfigured API prefix, so this answers 200 even when no model is
    configured. Model selection itself is driven by the client's own catalog.
    """
    model_id = get_settings().MODELS_MODEL_ID.strip()
    data = (
        [{"id": model_id, "object": "model", "owned_by": "codex-rosetta"}]
        if model_id
        else []
    )
    return {"object": "list", "data": data}


def _get_search_provider() -> SearchProvider | None:
    """Return the active search credential pool, or None when disabled."""
    if not get_settings().WEB_SEARCH_ENABLED:
        return None
    return _load_search_pool()


def _load_search_pool() -> SearchProviderPool:
    """Build (or reuse) the search credential pool."""
    s = get_settings()
    return get_search_pool(
        getattr(s, "SEARCH_POOL_FILE", DEFAULT_POOL_FILE),
        s.WEB_SEARCH_PROVIDER,
        s.WEB_SEARCH_API_KEY,
        s.WEB_SEARCH_BASE_URL,
    )


@router.post("/v1/responses", response_model=None)
async def create_response(request: Request):
    request_id = getattr(request.state, "request_id", "unknown")
    log = logger.bind(request_id=request_id)

    body = await request.json()

    blocked_models = {
        model.strip()
        for model in get_settings().BLOCKED_MODEL_IDS.split(",")
        if model.strip()
    }
    requested_model = body.get("model", "")
    if requested_model in blocked_models:
        log.warning("blocked_model", model=requested_model)
        return JSONResponse(
            status_code=400,
            content=_make_error_response(
                generate_response_id(),
                "model_not_supported",
                f"Model '{requested_model}' is not supported by this gateway.",
            ),
        )

    auditor = getattr(request.state, "auditor", NoOpAuditLogger())

    log.info(
        "request_received",
        model=body.get("model"),
        stream=body.get("stream", False),
        has_tools=bool(body.get("tools")),
        has_previous_response_id=bool(body.get("previous_response_id")),
        has_conversation=bool(body.get("conversation")),
        has_instructions=bool(body.get("instructions")),
    )

    if get_settings().UPSTREAM_API_MODE == "responses":
        return await _responses_mode_response(
            body, log, auditor, _get_search_provider()
        )

    upstream = get_upstream_client()
    store = get_conversation_store()
    request_converter = RequestConverter()

    conversation_messages: list[dict[str, Any]] | None = None
    prev_id = body.get("previous_response_id")
    conv_param = body.get("conversation")
    if prev_id:
        conversation_messages = await store.retrieve_messages(prev_id)
        log.debug("conversation_resolved", via="previous_response_id", response_id=prev_id, found=conversation_messages is not None)
    elif conv_param:
        conv_id = conv_param if isinstance(conv_param, str) else conv_param.get("id")
        if conv_id:
            conversation_messages = await store.retrieve_by_conversation_id(conv_id)
            log.debug("conversation_resolved", via="conversation", conversation_id=conv_id, found=conversation_messages is not None)

    try:
        chat_request, context = await request_converter.convert(
            body, conversation_messages, auditor=auditor
        )
    except UnsupportedParameterError as exc:
        log.warning("unsupported_parameter", param=exc.param, message=exc.message)
        return JSONResponse(status_code=400, content=exc.to_error_body())

    search_provider = _get_search_provider()
    if search_provider is None:
        chat_request, stripped_count = _strip_search_tools(chat_request)
        if stripped_count:
            log.info(
                "search_tools_stripped",
                reason="search_disabled",
                count=stripped_count,
            )

    log.info(
        "request_converted",
        response_id=context.response_id,
        message_count=len(chat_request.get("messages", [])),
        has_tools=bool(chat_request.get("tools")),
        tool_count=len(chat_request.get("tools", [])),
    )

    if get_settings().LOG_UPSTREAM_REQUESTS:
        log.debug("upstream_request_body", body=chat_request)

    is_streaming = body.get("stream", False)

    if is_streaming:
        return StreamingResponse(
            _stream_response(upstream, chat_request, context, store, body, log, auditor, search_provider),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        try:
            if search_provider:
                chat_response = await _search_loop_non_streaming(
                    upstream, chat_request, context, log, search_provider, auditor,
                )
            else:
                chat_response = await _call_upstream_with_retry(
                    upstream, chat_request, context, log
                )
        except httpx.HTTPStatusError as e:
            log.warning("upstream_error", status=e.response.status_code, error=str(e))
            return JSONResponse(
                status_code=e.response.status_code,
                content=_make_error_response(
                    context.response_id,
                    "upstream_error",
                    str(e),
                ),
            )

        if get_settings().LOG_UPSTREAM_RESPONSES:
            log.debug("upstream_response_body", body=chat_response)

        response_converter = ResponseConverter()
        responses_response = response_converter.convert(chat_response, context)

        output_items = responses_response.get("output", [])
        item_types = [item.get("type", "unknown") for item in output_items]
        usage = responses_response.get("usage", {})

        log.info(
            "response_sent",
            response_id=context.response_id,
            output_count=len(output_items),
            output_types=item_types,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )

        conv_id = _extract_conversation_id(body)
        await store.store(
            context.response_id,
            chat_request.get("messages", []),
            responses_response.get("output", []),
            conversation_id=conv_id,
        )
        log.debug("conversation_stored", response_id=context.response_id, conversation_id=conv_id)

        auditor.record_output_event("response.completed", {"response": responses_response})

        return JSONResponse(content=responses_response)


def _extract_web_search_tool_calls(
    chat_response: dict[str, Any],
) -> list[dict[str, Any]]:
    choices = chat_response.get("choices") or []
    web_search_calls = []
    for choice in choices:
        message = choice.get("message") or {}
        tool_calls = message.get("tool_calls") or []
        for tc in tool_calls:
            func = tc.get("function") or {}
            name = func.get("name", "")
            if _is_web_search_function_name(name):
                web_search_calls.append(tc)
    return web_search_calls


def _client_tool_calls(chat_response: dict[str, Any]) -> list[dict[str, Any]]:
    """Tool calls in a Chat Completions response that are not search calls."""
    choices = chat_response.get("choices") or []
    calls = []
    for choice in choices:
        message = choice.get("message") or {}
        for tc in message.get("tool_calls") or []:
            name = (tc.get("function") or {}).get("name", "")
            if not _is_web_search_function_name(name):
                calls.append(tc)
    return calls


def _is_web_search_function_name(name: str) -> bool:
    return name in {
        f"{ROSETTA_TOOL_PREFIX}web_search",
        f"{ROSETTA_TOOL_PREFIX}web_search_2025_08_26",
    }


def _chat_request_has_web_search_tool(chat_request: dict[str, Any]) -> bool:
    tools = chat_request.get("tools") or []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        func = tool.get("function") or {}
        if _is_web_search_function_name(func.get("name", "")):
            return True
    return False


def _strip_search_tools(chat_request: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Remove simulated web search tools so the model has to answer.

    Called when search is disabled/unavailable or when a turn must be forced
    to finish, so the client never receives a search call it cannot execute.
    """
    tools = chat_request.get("tools") or []
    kept = []
    removed = 0
    for tool in tools:
        if tool.get("type") == "function":
            name = (tool.get("function") or {}).get("name", "")
            if _is_web_search_function_name(name):
                removed += 1
                continue
        kept.append(tool)

    if not removed:
        return chat_request, 0

    new_request = dict(chat_request)
    if kept:
        new_request["tools"] = kept
    else:
        new_request.pop("tools", None)

    tool_choice = new_request.get("tool_choice")
    if isinstance(tool_choice, dict):
        chosen = (tool_choice.get("function") or {}).get("name", "")
        if _is_web_search_function_name(chosen):
            new_request["tool_choice"] = "auto"

    return new_request, removed


def _parse_search_query(tool_call: dict[str, Any]) -> str:
    func = tool_call.get("function") or {}
    arguments_str = func.get("arguments", "{}")
    try:
        arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
    except json.JSONDecodeError:
        arguments = {}
    return arguments.get("query", "")


def _search_sources(results: list[Any]) -> list[dict[str, Any]]:
    """URLs reported back to the client as web_search_call sources."""
    sources = []
    for result in results[:5]:
        url = getattr(result, "url", "")
        if url:
            sources.append({"type": "url", "url": url})
    return sources


def _inject_search_results(
    chat_request: dict[str, Any],
    chat_response: dict[str, Any],
    search_results_map: dict[str, str],
) -> dict[str, Any]:
    messages = list(chat_request.get("messages", []))

    choices = chat_response.get("choices") or []
    for choice in choices:
        message = choice.get("message") or {}
        assistant_msg = {"role": "assistant"}
        content = message.get("content")
        if content:
            assistant_msg["content"] = content
        tool_calls = message.get("tool_calls")
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        for tc in tool_calls or []:
            tc_id = tc.get("id", "")
            tc_result = search_results_map.get(tc_id, "搜索未返回结果。")
            messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": tc_result,
            })

    new_request = dict(chat_request)
    new_request["messages"] = messages
    return new_request


def _append_search_exchange(
    chat_request: dict[str, Any],
    tool_calls: list[dict[str, Any]],
    search_results_map: dict[str, str],
    assistant_text: str = "",
) -> dict[str, Any]:
    """Append the assistant tool call plus its results to the next request."""
    messages = list(chat_request.get("messages", []))

    assistant_msg: dict[str, Any] = {
        "role": "assistant",
        "content": assistant_text or None,
        "tool_calls": [
            {
                "id": tc.get("id", ""),
                "type": "function",
                "function": tc.get("function") or {},
            }
            for tc in tool_calls
        ],
    }
    messages.append(assistant_msg)

    for tc in tool_calls:
        tc_id = tc.get("id", "")
        messages.append({
            "role": "tool",
            "tool_call_id": tc_id,
            "content": search_results_map.get(tc_id, "搜索未返回结果。"),
        })

    new_request = dict(chat_request)
    new_request["messages"] = messages
    return new_request


async def _execute_searches(
    tool_calls: list[dict[str, Any]],
    search_provider: SearchProvider | None,
    max_results: int,
    log: Any,
    auditor: Any,
) -> tuple[dict[str, str], dict[str, list[dict[str, Any]]], int]:
    """Run each search call and render its result for the model.

    Returns the per call result text, the per call source list and the number
    of calls that failed.
    """
    results_map: dict[str, str] = {}
    sources_map: dict[str, list[dict[str, Any]]] = {}
    failures = 0

    for tc in tool_calls:
        tc_id = tc.get("id", "")
        query = _parse_search_query(tc)
        if not query:
            results_map[tc_id] = format_search_unavailable("搜索查询为空")
            sources_map[tc_id] = []
            failures += 1
            continue

        if search_provider is None:
            results_map[tc_id] = format_search_unavailable("未配置可用的搜索凭证")
            sources_map[tc_id] = []
            failures += 1
            log.warning("web_search_unavailable", query=query)
            continue

        log.info("web_search_executing", query=query)
        search_response = await search_provider.search(query, max_results=max_results)

        results_map[tc_id] = format_search_results(search_response)
        sources_map[tc_id] = _search_sources(search_response.results)

        if search_response.error:
            if search_response.error in QUERY_ERROR_KINDS:
                # The query was refused (e.g. "site:" without search terms).
                # The service is healthy, so this must not count towards the
                # "search is broken"止损 budget — the model just rephrases.
                log.warning(
                    "web_search_query_rejected",
                    query=query,
                    detail=search_response.error_detail,
                )
            else:
                failures += 1
                log.warning(
                    "web_search_failed",
                    query=query,
                    error_kind=search_response.error,
                    detail=search_response.error_detail,
                )
        else:
            log.info(
                "web_search_completed",
                query=query,
                result_count=len(search_response.results),
            )

        auditor.record_output_event("web_search", {
            "query": query,
            "result_count": len(search_response.results),
            "error": search_response.error,
        })

    return results_map, sources_map, failures


async def _search_loop_non_streaming(
    upstream: Any,
    chat_request: dict[str, Any],
    context: Any,
    log: Any,
    search_provider: SearchProvider | None,
    auditor: Any,
) -> dict[str, Any]:
    settings = get_settings()
    max_rounds = max(1, int(settings.WEB_SEARCH_MAX_ROUNDS or 1))
    max_results = int(settings.WEB_SEARCH_MAX_RESULTS or 5)

    current_request = chat_request
    consecutive_failures = 0
    rounds = 0
    forced = False
    chat_response: dict[str, Any] = {}

    # max_rounds search rounds plus (at most) two rounds that are already
    # forced to answer because the search tool was taken away.
    for round_num in range(max_rounds + 2):
        chat_response = await _call_upstream_with_retry(
            upstream, current_request, context, log
        )

        web_search_calls = _extract_web_search_tool_calls(chat_response)
        if not web_search_calls:
            return chat_response

        client_calls = _client_tool_calls(chat_response)
        if client_calls:
            log.info(
                "web_search_mixed_round",
                mode="non_streaming",
                search_calls=len(web_search_calls),
                client_tool_calls=len(client_calls),
            )
            return chat_response

        log.info(
            "search_loop_round",
            round=round_num + 1,
            search_calls=len(web_search_calls),
            forced=forced,
        )

        if forced:
            # The search tool was already removed: do not burn a query on a
            # call the model should not have made, just refuse politely.
            results_map = {
                call.get("id", ""): format_search_unavailable(
                    "本轮已停止联网搜索，请直接作答"
                )
                for call in web_search_calls
            }
            failures = len(web_search_calls)
        else:
            rounds += 1
            results_map, _, failures = await _execute_searches(
                web_search_calls, search_provider, max_results, log, auditor,
            )

        if failures == len(web_search_calls):
            consecutive_failures += 1
        else:
            consecutive_failures = 0

        current_request = _inject_search_results(current_request, chat_response, results_map)

        if not forced and (consecutive_failures >= 2 or rounds >= max_rounds):
            current_request, removed = _strip_search_tools(current_request)
            forced = True
            if removed:
                log.warning(
                    "search_loop_forcing_answer",
                    mode="non_streaming",
                    round=round_num + 1,
                    consecutive_failures=consecutive_failures,
                )

    log.warning("search_loop_unanswered", mode="non_streaming")
    return chat_response


async def _stream_response(
    upstream: Any,
    chat_request: dict[str, Any],
    context: Any,
    store: Any,
    original_body: dict[str, Any],
    log: Any,
    auditor: Any,
    search_provider: SearchProvider | None = None,
) -> Any:
    """Stream the upstream response, running search rounds inline.

    The same converter (and therefore the same response id, sequence numbers
    and output indexes) spans every round, so clients see the search items as
    they happen instead of receiving a buffered replay.
    """
    settings = get_settings()
    max_rounds = max(1, int(settings.WEB_SEARCH_MAX_ROUNDS or 1))
    max_results = int(settings.WEB_SEARCH_MAX_RESULTS or 5)

    converter = StreamConverter(
        context,
        auditor=auditor,
        defer_builtin_completion=True,
    )

    current_request = chat_request
    consecutive_failures = 0
    rounds = 0
    forced = False
    iterations = 0
    start = time.monotonic()
    chunk_count = 0

    log.info("stream_started", model=context.model, response_id=context.response_id)

    try:
        # max_rounds search rounds plus (at most) two rounds that are already
        # forced to answer because the search tool was taken away.
        while iterations < max_rounds + 2:
            iterations += 1
            marker = converter.round_item_marker()

            async for raw_chunk in upstream.chat_completions_stream(current_request):
                chunk_count += 1
                auditor.record_upstream_chunk(raw_chunk)
                async for event_type, event_data in converter.process_chunk(raw_chunk):
                    if event_data is None:
                        continue
                    if event_type:
                        yield format_sse_event(event_type, event_data)

            async for event_type, event_data in converter.flush_buffer():
                if event_data is None:
                    continue
                if event_type:
                    yield format_sse_event(event_type, event_data)

            search_calls = converter.collect_search_calls(marker)
            if not search_calls:
                break

            client_calls = converter.collect_client_tool_calls(marker)
            if client_calls:
                log.info(
                    "web_search_mixed_round",
                    mode="streaming",
                    search_calls=len(search_calls),
                    client_tool_calls=len(client_calls),
                )
                # The client owns this round (shell/apply_patch/...), so the
                # search call is handed over as-is. Close the item anyway so
                # no web_search_call is left dangling.
                for call in search_calls:
                    async for event_type, event_data in converter.complete_builtin_item(
                        call.get("_item_id", ""), []
                    ):
                        yield format_sse_event(event_type, event_data)
                break

            # Close the non-search items of this round before searching.
            async for event_type, event_data in converter.finalize_round():
                yield format_sse_event(event_type, event_data)

            round_text = converter.round_assistant_text(marker)
            if forced:
                # Search tools were already stripped: refuse the call instead
                # of spending a query on it, so the turn stays bounded.
                results_map = {
                    call.get("id", ""): format_search_unavailable(
                        "本轮已停止联网搜索，请直接作答"
                    )
                    for call in search_calls
                }
                sources_map: dict[str, list[dict[str, Any]]] = {}
                failures = len(search_calls)
            else:
                rounds += 1
                results_map, sources_map, failures = await _execute_searches(
                    search_calls, search_provider, max_results, log, auditor,
                )

            for call in search_calls:
                item_id = call.get("_item_id", "")
                sources = sources_map.get(call.get("id", ""), [])
                async for event_type, event_data in converter.complete_builtin_item(
                    item_id, sources
                ):
                    yield format_sse_event(event_type, event_data)

            if failures == len(search_calls):
                consecutive_failures += 1
            else:
                consecutive_failures = 0

            current_request = _append_search_exchange(
                current_request, search_calls, results_map, round_text,
            )

            if not forced and (consecutive_failures >= 2 or rounds >= max_rounds):
                current_request, removed = _strip_search_tools(current_request)
                forced = True
                if removed:
                    log.warning(
                        "search_loop_forcing_answer",
                        mode="streaming",
                        round=rounds,
                        max_rounds=max_rounds,
                        consecutive_failures=consecutive_failures,
                    )

            converter.prepare_for_next_round()
        else:
            log.warning(
                "search_loop_unanswered",
                mode="streaming",
                rounds=rounds,
                iterations=iterations,
            )

        async for event_type, event_data in converter.finalize():
            yield format_sse_event(event_type, event_data)

        output_items = converter.current_output_items
        conv_id = _extract_conversation_id(original_body)
        await store.store(
            context.response_id,
            current_request.get("messages", []),
            output_items,
            conversation_id=conv_id,
        )

        duration_ms = round((time.monotonic() - start) * 1000)
        item_types = [item.get("type", "unknown") for item in output_items]

        log.info(
            "stream_completed",
            response_id=context.response_id,
            output_count=len(output_items),
            output_types=item_types,
            chunk_count=chunk_count,
            search_rounds=rounds,
            duration_ms=duration_ms,
        )
        log.debug("conversation_stored", response_id=context.response_id, conversation_id=conv_id)

        auditor.set_output_types(item_types)

    except Exception as e:
        log.error("stream_error", error=str(e), exc_info=True)
        error_event = {
            "type": "response.error",
            "error": {"code": "stream_error", "message": str(e)},
            "sequence_number": converter.next_sequence_number(),
        }
        yield format_sse_event("response.error", error_event)

        failed_event = {
            "type": "response.failed",
            "response": {
                "id": context.response_id,
                "object": "response",
                "status": "failed",
                "error": {"code": "stream_error", "message": str(e)},
            },
            "sequence_number": converter.next_sequence_number(),
        }
        yield format_sse_event("response.failed", failed_event)



def _make_error_response(response_id: str, code: str, message: str) -> dict[str, Any]:
    """Full Response-shaped failure envelope (not just a bare error object)."""
    from codex_rosetta.utils.id_generation import unix_timestamp

    return {
        "id": response_id,
        "object": "response",
        "created_at": unix_timestamp(),
        "completed_at": None,
        "model": "",
        "status": "failed",
        "error": {"code": code, "message": message},
        "incomplete_details": None,
        "output": [],
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def _extract_conversation_id(body: dict[str, Any]) -> str | None:
    conv_param = body.get("conversation")
    if conv_param is None:
        return None
    if isinstance(conv_param, str):
        return conv_param
    if isinstance(conv_param, dict):
        return conv_param.get("id")
    return None


async def _call_upstream_with_retry(
    upstream: Any,
    chat_request: dict[str, Any],
    context: Any,
    log: Any = None,
    max_retries: int = 3,
) -> dict[str, Any]:
    if log is None:
        log = logger

    for attempt in range(max_retries + 1):
        try:
            start = time.monotonic()
            result = await upstream.chat_completions(chat_request)
            duration_ms = round((time.monotonic() - start) * 1000)
            log.debug("upstream_call_completed", attempt=attempt, duration_ms=duration_ms)
            return result
        except httpx.HTTPStatusError as e:
            if context.truncation != "auto":
                raise
            if e.response.status_code != 400:
                raise

            try:
                error_body = e.response.json()
                error_msg = str(error_body.get("error", {})).lower()
            except Exception:
                error_msg = e.response.text.lower()

            context_keywords = ["context_length", "context length", "maximum context",
                                "token limit", "max_tokens", "too many tokens"]
            if not any(kw in error_msg for kw in context_keywords):
                raise

            messages = chat_request.get("messages", [])
            if len(messages) <= 2:
                raise

            system_msgs = [m for m in messages if m.get("role") == "system"]
            non_system = [m for m in messages if m.get("role") != "system"]

            trim_count = max(1, len(non_system) // 4)
            non_system = non_system[trim_count:]

            if not non_system:
                raise

            chat_request["messages"] = system_msgs + non_system

            log.warning(
                "context_overflow_trimming",
                attempt=attempt,
                trimmed_count=trim_count,
                remaining_messages=len(system_msgs) + len(non_system),
            )


async def _responses_mode_response(
    body: dict[str, Any],
    log: Any,
    auditor: Any,
    search_provider: SearchProvider | None,
) -> Any:
    """Forward a Responses request as-is, looping only for Tavily search.

    ``UPSTREAM_API_MODE=responses`` skips the Responses<->Chat conversion
    entirely; the only edit is swapping built-in search tools for the internal
    function tool the upstream can actually execute.
    """
    upstream = get_upstream_client()
    normalized, wants_search = normalize_responses_request(
        body, search_enabled=search_provider is not None
    )
    search_loop = bool(wants_search and search_provider)

    log.info(
        "responses_passthrough",
        model=normalized.get("model"),
        stream=normalized.get("stream", False),
        search_loop=search_loop,
        tool_count=len(normalized.get("tools") or []),
        input_count=len(normalized.get("input") or []),
    )

    if normalized.get("stream", False):
        generator = (
            _stream_responses_search(
                upstream, normalized, log, auditor, search_provider
            )
            if search_loop
            else _stream_responses_verbatim(upstream, normalized, log)
        )
        return StreamingResponse(
            generator,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        if search_loop:
            data = await _responses_search_rounds(
                upstream, normalized, log, auditor, search_provider
            )
        else:
            data = await upstream.responses(normalized)
    except httpx.HTTPStatusError as e:
        log.warning(
            "upstream_error",
            mode="responses",
            status=e.response.status_code,
            error=str(e),
        )
        return JSONResponse(
            status_code=e.response.status_code,
            content=_make_error_response(
                generate_response_id(), "upstream_error", str(e)
            ),
        )

    output_items = data.get("output") or []
    log.info(
        "response_sent",
        mode="responses",
        status=data.get("status"),
        output_count=len(output_items),
        output_types=[item.get("type") for item in output_items if isinstance(item, dict)],
    )
    return JSONResponse(content=data)


async def _stream_responses_verbatim(
    upstream: Any, body: dict[str, Any], log: Any
) -> Any:
    """Forward upstream Responses SSE untouched, reporting errors as events.

    Without this an upstream 400 would surface to the client as a bare
    "stream disconnected" with no reason attached.
    """
    mux = ResponsesStreamMux()
    try:
        async for chunk in upstream.responses_stream(body):
            yield chunk
    except Exception as e:  # noqa: BLE001 - the stream must always terminate
        log.error("stream_error", mode="responses", error=str(e), exc_info=True)
        etype, event = mux.build_failed_event(str(e))
        yield format_sse_event(etype, event)


def _append_responses_search_exchange(
    body: dict[str, Any],
    calls: list[SearchCallState],
    results_map: dict[str, str],
) -> dict[str, Any]:
    """Append the internal search call and its results to the next round input.

    Only the fields every Responses-compatible upstream accepts are emitted:
    strict relays reject unknown ones.
    """
    updated = dict(body)
    items = list(updated.get("input") or [])
    for call in calls:
        items.append({
            "type": "function_call",
            "call_id": call.call_id,
            "name": INTERNAL_SEARCH_NAME,
            "arguments": call.arguments or "{}",
        })
    for call in calls:
        items.append({
            "type": "function_call_output",
            "call_id": call.call_id,
            "output": results_map.get(call.call_id, ""),
        })
    updated["input"] = items
    return updated


def _to_search_states(items: list[dict[str, Any]]) -> list[SearchCallState]:
    return [
        SearchCallState(
            item_id=str(item.get("id") or ""),
            call_id=str(item.get("call_id") or ""),
            output_index=0,
            web_search_id=generate_item_id("web_search"),
            arguments=str(item.get("arguments") or ""),
        )
        for item in items
    ]


async def _responses_search_rounds(
    upstream: Any,
    body: dict[str, Any],
    log: Any,
    auditor: Any,
    search_provider: SearchProvider | None,
) -> dict[str, Any]:
    """Non-streaming Responses search loop, mirroring the streaming rules."""
    settings = get_settings()
    max_rounds = max(1, int(settings.WEB_SEARCH_MAX_ROUNDS or 1))
    max_results = int(settings.WEB_SEARCH_MAX_RESULTS or 5)

    current = body
    accumulated: list[dict[str, Any]] = []
    consecutive_failures = 0
    rounds = 0
    forced = False
    response: dict[str, Any] = {}

    for round_index in range(max_rounds + 2):
        response = await upstream.responses(current)
        output = [i for i in (response.get("output") or []) if isinstance(i, dict)]
        search_items = [i for i in output if is_internal_search_item(i)]
        if not search_items:
            accumulated.extend(output)
            break

        calls = _to_search_states(search_items)
        client_calls = [
            i
            for i in output
            if not is_internal_search_item(i)
            and i.get("type") in {"function_call", "custom_tool_call"}
        ]

        log.info(
            "search_loop_round",
            mode="non_streaming",
            round=round_index + 1,
            search_calls=len(calls),
            forced=forced,
        )

        if forced:
            results_map = {
                call.call_id: format_search_unavailable("本轮已停止联网搜索，请直接作答")
                for call in calls
            }
            sources_map: dict[str, list[dict[str, Any]]] = {}
            failures = len(calls)
        elif client_calls:
            log.info(
                "web_search_mixed_round",
                mode="non_streaming",
                search_calls=len(calls),
                client_tool_calls=len(client_calls),
            )
            results_map, sources_map, failures = {}, {}, 0
        else:
            rounds += 1
            results_map, sources_map, failures = await _execute_searches(
                [to_search_tool_call(call) for call in calls],
                search_provider,
                max_results,
                log,
                auditor,
            )

        for item in output:
            if is_internal_search_item(item):
                call = next(
                    c for c in calls if c.item_id == str(item.get("id") or "")
                )
                accumulated.append(build_web_search_item(
                    call.web_search_id,
                    parse_arguments(call.arguments),
                    sources_map.get(call.call_id),
                ))
            else:
                accumulated.append(item)

        if client_calls:
            break

        if failures == len(calls):
            consecutive_failures += 1
        else:
            consecutive_failures = 0

        current = _append_responses_search_exchange(current, calls, results_map)

        if not forced and (consecutive_failures >= 2 or rounds >= max_rounds):
            current = strip_search_function_tool(current, FORCE_ANSWER_INSTRUCTION)
            forced = True
            log.warning(
                "search_loop_forcing_answer",
                mode="non_streaming",
                round=rounds,
                max_rounds=max_rounds,
                consecutive_failures=consecutive_failures,
            )
    else:
        log.warning("search_loop_unanswered", mode="non_streaming", rounds=rounds)

    merged = dict(response)
    merged["output"] = accumulated
    return merged


async def _stream_responses_search(
    upstream: Any,
    body: dict[str, Any],
    log: Any,
    auditor: Any,
    search_provider: SearchProvider | None,
) -> Any:
    """Stream Responses rounds through one logical response while searching."""
    settings = get_settings()
    max_rounds = max(1, int(settings.WEB_SEARCH_MAX_ROUNDS or 1))
    max_results = int(settings.WEB_SEARCH_MAX_RESULTS or 5)

    mux = ResponsesStreamMux()
    current = body
    consecutive_failures = 0
    rounds = 0
    forced = False
    start = time.monotonic()

    try:
        for round_index in range(max_rounds + 2):
            async for event_type, data in parse_upstream_sse_stream(
                upstream.responses_stream(current)
            ):
                if data is None:
                    continue
                for etype, event in mux.process_event(event_type, data, round_index):
                    yield format_sse_event(etype, event)

            result = mux.finalize_round()
            if result.terminal_event is not None:
                log.warning(
                    "responses_upstream_terminal",
                    event=result.terminal_event[0],
                )
                return
            if not result.search_calls:
                break
            if result.client_calls:
                log.info(
                    "web_search_mixed_round",
                    mode="streaming",
                    search_calls=len(result.search_calls),
                    client_tool_calls=len(result.client_calls),
                )
                for etype, event in mux.complete_search_calls({}, {}):
                    yield format_sse_event(etype, event)
                break

            log.info(
                "search_loop_round",
                mode="streaming",
                round=round_index + 1,
                search_calls=len(result.search_calls),
                forced=forced,
            )

            if forced:
                results_map = {
                    call.call_id: format_search_unavailable(
                        "本轮已停止联网搜索，请直接作答"
                    )
                    for call in result.search_calls
                }
                sources_map: dict[str, list[dict[str, Any]]] = {}
                failures = len(result.search_calls)
            else:
                rounds += 1
                results_map, sources_map, failures = await _execute_searches(
                    [to_search_tool_call(call) for call in result.search_calls],
                    search_provider,
                    max_results,
                    log,
                    auditor,
                )

            for etype, event in mux.complete_search_calls(results_map, sources_map):
                yield format_sse_event(etype, event)

            if failures == len(result.search_calls):
                consecutive_failures += 1
            else:
                consecutive_failures = 0

            current = _append_responses_search_exchange(
                current, result.search_calls, results_map
            )

            if not forced and (consecutive_failures >= 2 or rounds >= max_rounds):
                current = strip_search_function_tool(current, FORCE_ANSWER_INSTRUCTION)
                forced = True
                log.warning(
                    "search_loop_forcing_answer",
                    mode="streaming",
                    round=rounds,
                    max_rounds=max_rounds,
                    consecutive_failures=consecutive_failures,
                )

            mux.start_next_round()
        else:
            log.warning("search_loop_unanswered", mode="streaming", rounds=rounds)

        etype, event = mux.build_completed_event()
        yield format_sse_event(etype, event)

        log.info(
            "stream_completed",
            mode="responses",
            search_rounds=rounds,
            duration_ms=round((time.monotonic() - start) * 1000),
        )
    except Exception as e:  # noqa: BLE001 - the stream must always terminate
        log.error("stream_error", mode="responses", error=str(e), exc_info=True)
        etype, event = mux.build_failed_event(str(e))
        yield format_sse_event(etype, event)
