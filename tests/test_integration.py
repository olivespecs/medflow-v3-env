"""
Integration tests for endpoints added in the observability session:
  - GET /metrics  (Prometheus-style counters)
  - GET /export   (episode snapshot + reward_trend)

The existing test_api.py (9 test classes) already covers the core
episode lifecycle, rate limiting, expiry, and error recovery.
These tests fill the gap for the two new endpoints only.

Uses httpx.ASGITransport so no running server is needed.
Run with:  pytest tests/test_integration.py -v
"""

from __future__ import annotations

import pytest

from src.api import (
    app,
    _episodes,
    _episodes_lock,
    _read_rate_limit_lock,
    _read_rate_limit_store,
    _rate_limit_lock,
    _rate_limit_store,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset episode store and rate-limit store before/after every test."""
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


@pytest.fixture
def client():
    """Synchronous ASGI test client (same pattern as existing test_api.py)."""
    from starlette.testclient import TestClient
    return TestClient(app)


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------


class TestMetricsEndpoint:
    """GET /metrics returns Prometheus-style counters."""

    def test_metrics_returns_expected_keys(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "counters" in data
        assert "capacity" in data
        assert "rate_limit" in data

    def test_metrics_counters_have_required_fields(self, client):
        resp = client.get("/metrics")
        counters = resp.json()["counters"]
        for key in ("episodes_created", "steps_taken", "grades_issued", "grader_errors", "rate_limit_hits"):
            assert key in counters, f"Missing counter key: {key}"

    def test_episodes_created_increments_after_reset(self, client):
        before = client.get("/metrics").json()["counters"]["episodes_created"]
        client.post("/reset", json={"task_id": 1, "seed": 42})
        after = client.get("/metrics").json()["counters"]["episodes_created"]
        assert after == before + 1

    def test_steps_taken_increments_after_step(self, client):
        reset = client.post("/reset", json={"task_id": 1, "seed": 42})
        episode_id = reset.json()["episode_id"]
        before = client.get("/metrics").json()["counters"]["steps_taken"]

        client.post(
            "/step",
            params={"episode_id": episode_id},
            json={"records": [{"record_id": "r1"}], "is_final": False},
        )
        after = client.get("/metrics").json()["counters"]["steps_taken"]
        assert after == before + 1

    def test_capacity_reflects_active_episodes(self, client):
        before = client.get("/metrics").json()["capacity"]["active_episodes"]
        client.post("/reset", json={"task_id": 1, "seed": 42})
        after = client.get("/metrics").json()["capacity"]["active_episodes"]
        assert after == before + 1

    def test_rate_limit_config_matches_env(self, client):
        from src.api import RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS
        rl = client.get("/metrics").json()["rate_limit"]
        assert rl["requests_per_window"] == RATE_LIMIT_REQUESTS
        assert rl["window_seconds"] == RATE_LIMIT_WINDOW_SECONDS


# ---------------------------------------------------------------------------
# /export
# ---------------------------------------------------------------------------


class TestExportEndpoint:
    """GET /export returns full episode snapshot including reward_trend."""

    def test_export_requires_episode_id(self, client):
        resp = client.get("/export")
        assert resp.status_code == 422  # missing required query param

    def test_export_unknown_episode_returns_404(self, client):
        import uuid
        resp = client.get("/export", params={"episode_id": str(uuid.uuid4())})
        assert resp.status_code == 404

    def test_export_before_any_step_has_empty_reward_trend(self, client):
        reset = client.post("/reset", json={"task_id": 1, "seed": 42})
        assert reset.status_code == 200
        episode_id = reset.json()["episode_id"]

        export = client.get("/export", params={"episode_id": episode_id})
        assert export.status_code == 200
        data = export.json()
        assert "reward_trend" in data
        assert isinstance(data["reward_trend"], list)
        assert len(data["reward_trend"]) == 0  # no steps taken yet

    def test_export_reward_trend_grows_with_steps(self, client):
        reset = client.post("/reset", json={"task_id": 1, "seed": 42})
        episode_id = reset.json()["episode_id"]

        for _ in range(3):
            client.post(
                "/step",
                params={"episode_id": episode_id},
                json={"records": [{"record_id": "r1"}], "is_final": False},
            )

        export = client.get("/export", params={"episode_id": episode_id})
        data = export.json()
        assert len(data["reward_trend"]) == 3

    def test_export_reward_trend_entry_shape(self, client):
        reset = client.post("/reset", json={"task_id": 1, "seed": 42})
        episode_id = reset.json()["episode_id"]

        client.post(
            "/step",
            params={"episode_id": episode_id},
            json={"records": [{"record_id": "r1"}], "is_final": True},
        )

        data = client.get("/export", params={"episode_id": episode_id}).json()
        entry = data["reward_trend"][0]
        assert "step" in entry
        assert "score" in entry
        assert "passed" in entry
        assert 0.0 <= entry["score"] <= 1.0

    def test_export_includes_episode_id(self, client):
        reset = client.post("/reset", json={"task_id": 2, "seed": 99})
        episode_id = reset.json()["episode_id"]

        data = client.get("/export", params={"episode_id": episode_id}).json()
        assert data["episode_id"] == episode_id

    def test_export_works_for_all_tasks(self, client):
        for task_id in (1, 2, 3):
            reset = client.post("/reset", json={"task_id": task_id, "seed": 42})
            episode_id = reset.json()["episode_id"]
            resp = client.get("/export", params={"episode_id": episode_id})
            assert resp.status_code == 200, f"Task {task_id} export failed"
