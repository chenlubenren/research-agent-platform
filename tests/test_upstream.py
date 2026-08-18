import asyncio

import httpx
import pytest

from research_agent_platform import upstream


@pytest.fixture(autouse=True)
def configured_upstream(monkeypatch):
    monkeypatch.setattr(upstream.config, "upstream_api_key", "test-key")
    monkeypatch.setattr(upstream.config, "upstream_base_url", "https://example.test/v1")


def test_request_retries_one_transient_gateway_failure_for_get(monkeypatch):
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(502, json={"error": {"message": "temporarily unavailable"}})
        return httpx.Response(200, json={"ok": True})

    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        upstream.httpx,
        "AsyncClient",
        lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs),
    )

    result = asyncio.run(upstream._request("GET", "/models"))

    assert result == {"ok": True}
    assert attempts == 2


def test_request_retries_two_transient_gateway_failures_for_get(monkeypatch):
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(502, json={"error": {"message": "temporarily unavailable"}})
        return httpx.Response(200, json={"ok": True})

    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        upstream.httpx,
        "AsyncClient",
        lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs),
    )

    result = asyncio.run(upstream._request("GET", "/models"))

    assert result == {"ok": True}
    assert attempts == 3


def test_request_does_not_retry_chat_post_after_gateway_failure(monkeypatch):
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(502, json={"error": {"message": "temporarily unavailable"}})

    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        upstream.httpx,
        "AsyncClient",
        lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs),
    )

    try:
        asyncio.run(upstream._request("POST", "/chat/completions", {"messages": []}))
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 502
    else:
        raise AssertionError("Expected the failed POST to raise")

    assert attempts == 1
