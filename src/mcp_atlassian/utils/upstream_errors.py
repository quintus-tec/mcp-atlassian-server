"""Surface Atlassian HTTP failures that the data layer swallowed.

Many fetcher methods catch ``HTTPError`` and return ``[]``/``{}``, which reaches
the MCP client as a successful empty result. A session response hook records
every Atlassian response of the current tool call so the tool layer can turn a
swallowed failure into an MCP error.
"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, TypeVar

import requests

T = TypeVar("T")

AUTH_STATUSES = (401, 403)
_MAX_MESSAGE_LENGTH = 300


@dataclass
class UpstreamCallTrace:
    """Atlassian responses observed during one tool call."""

    last_failure: tuple[int, str] | None = None
    succeeded: bool = False

    def unreported_failure(self) -> tuple[int, str] | None:
        """The failure a normal tool return would hide, if any.

        The call ended on a failure and either nothing succeeded (the empty-result
        case) or it was an auth/scope rejection. Other failures in a call that also
        succeeded (fallback endpoints, per-item batch errors) stay in its result.
        """
        if self.last_failure is None:
            return None
        if not self.succeeded or self.last_failure[0] in AUTH_STATUSES:
            return self.last_failure
        return None


_current_trace: ContextVar[UpstreamCallTrace | None] = ContextVar(
    "atlassian_upstream_trace", default=None
)


def start_trace() -> tuple[UpstreamCallTrace, Any]:
    """Start recording responses for the current tool call."""
    trace = UpstreamCallTrace()
    return trace, _current_trace.set(trace)


def end_trace(token: Any) -> None:
    _current_trace.reset(token)


def run_untraced(fn: Callable[..., T], *args: Any) -> T:
    """Run ``fn`` without recording, so credential checks don't count as tool calls."""
    token = _current_trace.set(None)
    try:
        return fn(*args)
    finally:
        _current_trace.reset(token)


def record_response(response: requests.Response, *args: Any, **kwargs: Any) -> None:
    """``requests`` response hook; no-op outside a tool call."""
    trace = _current_trace.get()
    if trace is None:
        return
    if response.status_code >= 400:
        trace.last_failure = (response.status_code, extract_error_message(response))
    else:
        trace.succeeded = True
        trace.last_failure = None


def install_response_recorder(session: requests.Session) -> None:
    session.hooks["response"].append(record_response)


def extract_error_message(response: requests.Response | None) -> str:
    """Short Atlassian error message from a response body; never headers or URL."""
    if response is None:
        return ""
    try:
        body = response.json()
    except ValueError:
        return _truncate(getattr(response, "reason", "") or "")
    if not isinstance(body, dict):
        return ""
    parts: list[str] = []
    if isinstance(body.get("message"), str):
        parts.append(body["message"])
    if isinstance(body.get("errorMessage"), str):
        parts.append(body["errorMessage"])
    if isinstance(body.get("errorMessages"), list):
        parts.extend(str(m) for m in body["errorMessages"] if m)
    if isinstance(body.get("errors"), dict):
        parts.extend(f"{k}: {v}" for k, v in body["errors"].items())
    return _truncate("; ".join(parts))


def format_upstream_error(status: int, message: str) -> str:
    """MCP error text for a failed Atlassian call."""
    text = f"Atlassian API error {status}"
    if message:
        text += f": {message}"
    if status in AUTH_STATUSES:
        text += (
            ". Authorization failed: the token is expired, revoked or lacks the "
            "scope for this operation; reconnect the Atlassian account."
        )
    return text


def _truncate(text: str) -> str:
    text = text.strip()
    if len(text) <= _MAX_MESSAGE_LENGTH:
        return text
    return text[: _MAX_MESSAGE_LENGTH - 3] + "..."
