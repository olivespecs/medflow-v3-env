"""Tests for src.client typed HTTP wrappers."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from src.api import (
    _episodes,
    _episodes_lock,
    _read_rate_limit_lock,
    _read_rate_limit_store,
    _rate_limit_lock,
    _rate_limit_store,
    app,
)
from src.client import APIError, MedicalOpenEnvClient, SyncMedicalOpenEnvClient


@pytest.fixture(autouse=True)
def _clean_state():
    with _episodes_lock:
        _episodes.clear()
    with _rate_limit_lock:
        _rate_limit_store.clear()
    with _read_rate_limit_lock:
        _read_rate_limit_store.clear()
    yield
    with _episodes_lock:
        _episodes.clear()
    with _rate_limit_lock:
        _rate_limit_store.clear()
    with _read_rate_limit_lock:
        _read_rate_limit_store.clear()


def test_async_client_roundtrip_task1():
    async def _run() -> None:
        transport = httpx.ASGITransport(app=app)
        async with MedicalOpenEnvClient(base_url="http://testserver", transport=transport) as client:
            tasks = await client.tasks()
            assert len(tasks["tasks"]) == 5

            contract = await client.contract()
            assert contract["mode"] == "agentic"

            reset = await client.reset(task_id=1, seed=42)
            episode_id = reset["episode_id"]
            records = reset["observation"]["records"]

            step_result = await client.step(episode_id, records=records, is_final=True)
            assert "reward" in step_result
            assert step_result["done"] is True

            state = await client.state(episode_id)
            assert state["episode_id"] == episode_id
            assert state["done"] is True

    asyncio.run(_run())


def test_async_client_task4_payload_path():
    async def _run() -> None:
        transport = httpx.ASGITransport(app=app)
        async with MedicalOpenEnvClient(base_url="http://testserver", transport=transport) as client:
            reset = await client.reset(task_id=4, seed=42)
            episode_id = reset["episode_id"]
            n_records = len(reset["observation"]["records"])

            knowledge = [
                {
                    "entities": [{"text": "hypertension", "type": "Condition", "code": "I10"}],
                    "summary": "Patient has hypertension.",
                }
                for _ in range(n_records)
            ]

            result = await client.step(episode_id, knowledge=knowledge, is_final=True)
            assert "reward" in result
            assert result["done"] is True

    asyncio.run(_run())


def test_sync_client_builds_step_payload_and_query_params():
    captured: dict[str, object] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["episode_id"] = request.url.params.get("episode_id")
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(_handler)
    with SyncMedicalOpenEnvClient(base_url="http://unit.test", transport=transport) as client:
        response = client.step(
            "episode-123",
            records=[{"record_id": "r1"}],
            is_final=True,
        )

    assert response["ok"] is True
    assert captured["method"] == "POST"
    assert captured["path"] == "/step"
    assert captured["episode_id"] == "episode-123"
    assert captured["body"] == {
        "records": [{"record_id": "r1"}],
        "is_final": True,
    }


def test_sync_client_raises_api_error_on_failure():
    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": "invalid payload"})

    transport = httpx.MockTransport(_handler)
    with SyncMedicalOpenEnvClient(base_url="http://unit.test", transport=transport) as client:
        with pytest.raises(APIError) as exc:
            client.reset(task_id=99, seed=0)

    assert exc.value.status_code == 422
    assert "invalid payload" in str(exc.value)
