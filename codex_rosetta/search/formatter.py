from __future__ import annotations

from codex_rosetta.search.base import (
    ERROR_INVALID_KEY,
    ERROR_INVALID_QUERY,
    ERROR_NETWORK_ERROR,
    ERROR_QUOTA_EXCEEDED,
    ERROR_RATE_LIMITED,
    ERROR_UNKNOWN,
    ERROR_UPSTREAM_ERROR,
    SearchResponse,
)

_ERROR_REASONS = {
    ERROR_INVALID_KEY: "API Key 无效或未授权",
    ERROR_QUOTA_EXCEEDED: "搜索服务额度已用尽",
    ERROR_RATE_LIMITED: "触发搜索服务限流",
    ERROR_UPSTREAM_ERROR: "搜索服务返回错误",
    ERROR_NETWORK_ERROR: "搜索服务网络超时或不可达",
    ERROR_INVALID_QUERY: "搜索查询被搜索服务拒绝",
    ERROR_UNKNOWN: "搜索服务未知错误",
}

QUERY_REJECTED_TEMPLATE = (
    "搜索查询被拒绝（原因：{reason}）。搜索服务本身可用，"
    "请改用更普通的检索词重新调用 web_search；若无法改写，"
    "请直接依据已有知识回答，并说明本次未能联网搜索。"
)

UNAVAILABLE_TEMPLATE = (
    "搜索服务暂不可用（原因：{reason}）。不要再用 web_search 重试，"
    "请直接依据已有知识回答，并说明本次未能联网搜索。"
)


def format_search_error(search_response: SearchResponse) -> str:
    """Render a typed provider failure as an instruction for the model."""
    reason = _ERROR_REASONS.get(search_response.error or "", "搜索服务未知错误")
    if search_response.error_detail:
        detail = search_response.error_detail.strip()
        if detail:
            reason = f"{reason}，{detail[:160]}"
    return UNAVAILABLE_TEMPLATE.format(reason=reason)


def format_search_unavailable(detail: str) -> str:
    """Render an infrastructure reason (no credential, empty query, ...)."""
    reason = (detail or "搜索服务未知错误").strip()
    return UNAVAILABLE_TEMPLATE.format(reason=reason)


def format_search_results(search_response: SearchResponse) -> str:
    if search_response.error == ERROR_INVALID_QUERY:
        reason = _ERROR_REASONS[ERROR_INVALID_QUERY]
        if search_response.error_detail:
            reason = f"{reason}，{search_response.error_detail.strip()[:160]}"
        return QUERY_REJECTED_TEMPLATE.format(reason=reason)

    if search_response.error:
        return format_search_error(search_response)

    if not search_response.results:
        return f"搜索 \"{search_response.query}\" 未返回任何结果。"

    lines = [f"搜索 \"{search_response.query}\" 的结果（共 {len(search_response.results)} 条）：\n"]

    for i, result in enumerate(search_response.results, 1):
        lines.append(f"{i}. {result.title}")
        lines.append(f"   {result.url}")
        if result.snippet:
            lines.append(f"   {result.snippet}")
        lines.append("")

    return "\n".join(lines)
