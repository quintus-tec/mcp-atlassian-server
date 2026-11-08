import json
from unittest.mock import MagicMock

import pytest
import requests
from fastmcp.exceptions import ToolError
from requests.exceptions import HTTPError

from mcp_atlassian.exceptions import MCPAtlassianAuthenticationError
from mcp_atlassian.utils.decorators import (
    check_write_access,
    handle_auth_errors,
    handle_tool_errors,
)
from mcp_atlassian.utils.upstream_errors import record_response, run_untraced


class DummyContext:
    def __init__(self, read_only):
        self.request_context = MagicMock()
        self.request_context.lifespan_context = {
            "app_lifespan_context": MagicMock(read_only=read_only)
        }


@pytest.mark.asyncio
async def test_check_write_access_blocks_in_read_only():
    @check_write_access
    async def dummy_tool(ctx, x):
        return x * 2

    ctx = DummyContext(read_only=True)
    with pytest.raises(ToolError) as exc:
        await dummy_tool(ctx, 3)
    assert "read-only mode" in str(exc.value)


@pytest.mark.asyncio
async def test_check_write_access_allows_in_writable():
    @check_write_access
    async def dummy_tool(ctx, x):
        return x * 2

    ctx = DummyContext(read_only=False)
    result = await dummy_tool(ctx, 4)
    assert result == 8


@pytest.mark.asyncio
async def test_handle_tool_errors_wraps_exception_as_tool_error():
    @handle_tool_errors
    async def failing_tool():
        raise ValueError("something went wrong")

    with pytest.raises(ToolError) as exc:
        await failing_tool()
    assert "Error calling tool 'failing_tool'" in str(exc.value)
    assert "something went wrong" in str(exc.value)


@pytest.mark.asyncio
async def test_handle_tool_errors_passes_through_tool_error():
    @handle_tool_errors
    async def tool_with_tool_error():
        raise ToolError("explicit tool error")

    with pytest.raises(ToolError) as exc:
        await tool_with_tool_error()
    assert "explicit tool error" in str(exc.value)


@pytest.mark.asyncio
async def test_handle_tool_errors_preserves_return_value():
    @handle_tool_errors
    async def good_tool():
        return "success"

    result = await good_tool()
    assert result == "success"


def _atlassian_response(
    status_code: int, body: dict | None = None
) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response._content = json.dumps(body or {}).encode()
    response.url = "https://api.atlassian.com/ex/jira/cloud-id/rest/agile/1.0/board"
    return response


SCOPE_401 = {"code": 401, "message": "Unauthorized; scope does not match"}


@pytest.mark.asyncio
async def test_handle_tool_errors_raises_on_swallowed_scope_error():
    @handle_tool_errors
    async def get_agile_boards():
        record_response(_atlassian_response(401, SCOPE_401))
        return "[]"

    with pytest.raises(ToolError) as exc:
        await get_agile_boards()
    message = str(exc.value)
    assert "Atlassian API error 401: Unauthorized; scope does not match" in message
    assert "lacks the scope" in message
    assert "api.atlassian.com" not in message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("statuses", "fails"),
    [
        ([404], True),
        ([500], True),
        ([200, 403], True),
        ([404, 200], False),
        ([200, 400], False),
        ([200], False),
    ],
)
async def test_handle_tool_errors_judges_swallowed_failures(statuses, fails):
    @handle_tool_errors
    async def tool():
        for status in statuses:
            record_response(_atlassian_response(status, {"errorMessages": ["boom"]}))
        return "[]"

    if fails:
        with pytest.raises(ToolError, match=f"Atlassian API error {statuses[-1]}"):
            await tool()
    else:
        assert await tool() == "[]"


@pytest.mark.asyncio
async def test_handle_tool_errors_ignores_untraced_credential_check():
    @handle_tool_errors
    async def tool():
        run_untraced(record_response, _atlassian_response(200))
        record_response(_atlassian_response(404, {"errorMessages": ["No board"]}))
        return "[]"

    with pytest.raises(ToolError, match="Atlassian API error 404: No board"):
        await tool()


@pytest.mark.asyncio
async def test_handle_tool_errors_formats_raised_http_error():
    @handle_tool_errors
    async def tool():
        raise HTTPError(response=_atlassian_response(403, SCOPE_401))

    with pytest.raises(ToolError) as exc:
        await tool()
    assert "Atlassian API error 403" in str(exc.value)
    assert "api.atlassian.com" not in str(exc.value)


# --- handle_auth_errors tests ---


def _make_http_error(status_code: int) -> HTTPError:
    """Create an HTTPError with a mocked response."""
    response = MagicMock()
    response.status_code = status_code
    err = HTTPError(response=response)
    return err


class _FakeService:
    """Dummy class to test the self-bound decorator."""

    @handle_auth_errors("Test API")
    def do_work(self, value: str) -> str:
        return f"ok:{value}"

    @handle_auth_errors("Test API")
    def raise_http_error(self, status_code: int) -> None:
        raise _make_http_error(status_code)

    @handle_auth_errors("Test API")
    def raise_value_error(self) -> None:
        raise ValueError("bad input")


def test_handle_auth_errors_returns_value():
    svc = _FakeService()
    assert svc.do_work("hello") == "ok:hello"


@pytest.mark.parametrize("status_code", [401, 403])
def test_handle_auth_errors_catches_auth_errors(
    status_code: int,
) -> None:
    svc = _FakeService()
    with pytest.raises(MCPAtlassianAuthenticationError) as exc:
        svc.raise_http_error(status_code)
    assert "Authentication failed" in str(exc.value)
    assert str(status_code) in str(exc.value)


def test_handle_auth_errors_passes_through_404():
    svc = _FakeService()
    with pytest.raises(HTTPError) as exc:
        svc.raise_http_error(404)
    assert exc.value.response.status_code == 404


def test_handle_auth_errors_passes_through_non_http_error():
    svc = _FakeService()
    with pytest.raises(ValueError, match="bad input"):
        svc.raise_value_error()


def test_handle_auth_errors_passes_through_no_response():
    """HTTPError with response=None should re-raise."""

    class Svc:
        @handle_auth_errors("Test API")
        def fail(self) -> None:
            raise HTTPError(response=None)

    with pytest.raises(HTTPError):
        Svc().fail()
