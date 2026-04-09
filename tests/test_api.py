"""
Comprehensive API integration tests for the FastAPI OpenEnv application.

Tests cover:
1. Endpoint integration tests
2. Concurrent episode handling
3. Episode expiry logic
4. Rate limiting
5. Error recovery scenarios
"""

from __future__ import annotations

from collections import deque
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

from src.api import (
    GraderError,
    EPISODE_TTL_SECONDS,
    MAX_EPISODES,
    RATE_LIMIT_ENTRY_TTL_SECONDS,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
    _episodes,
    _episodes_lock,
    _get_cors_origins,
    _is_public_path,
    _read_rate_limit_lock,
    _read_rate_limit_store,
    _rate_limit_lock,
    _rate_limit_store,
    _purge_expired,
    _purge_stale_rate_limits,
    app,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> TestClient:
    """Create a fresh TestClient for each test."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def cleanup_episodes():
    """Clean up global episode store before and after each test."""
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
def episode_id(client: TestClient) -> str:
    """Create a fresh episode and return its ID."""
    response = client.post("/reset", json={"task_id": 1, "seed": 42})
    assert response.status_code == 200
    return response.json()["episode_id"]


@pytest.fixture
def task4_episode_id(client: TestClient) -> str:
    """Create a Task 4 episode for knowledge extraction tests."""
    response = client.post("/reset", json={"task_id": 4, "seed": 42})
    assert response.status_code == 200
    return response.json()["episode_id"]


# ---------------------------------------------------------------------------
# 1. Endpoint Integration Tests
# ---------------------------------------------------------------------------


class TestEndpointIntegration:
    """Test basic endpoint functionality and response schemas."""

    def test_post_mcp_tools_list_returns_jsonrpc(self, client: TestClient):
        """POST /mcp supports tools/list JSON-RPC compatibility for validators."""
        response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": {},
            },
        )
        assert response.status_code == 200

        payload = response.json()
        assert payload["jsonrpc"] == "2.0"
        assert payload["id"] == 1
        assert "result" in payload
        assert "tools" in payload["result"]
        assert len(payload["result"]["tools"]) >= 1

    def test_post_mcp_unknown_method_returns_jsonrpc_error(self, client: TestClient):
        """POST /mcp unknown methods return JSON-RPC method-not-found errors."""
        response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 99,
                "method": "unknown/method",
                "params": {},
            },
        )
        assert response.status_code == 200

        payload = response.json()
        assert payload["jsonrpc"] == "2.0"
        assert payload["id"] == 99
        assert payload["error"]["code"] == -32601

    def test_get_tasks_returns_task_list(self, client: TestClient):
        """GET /tasks returns list of available tasks with correct schema."""
        response = client.get("/tasks")
        assert response.status_code == 200
        
        data = response.json()
        assert "tasks" in data
        assert "action_schema" in data
        assert len(data["tasks"]) == 5
        
        # Verify task structure
        for task in data["tasks"]:
            assert "id" in task
            assert "name" in task
            assert "difficulty" in task
            assert "description" in task
            assert "grader_formula" in task
            assert "pass_bar" in task
            assert "max_steps" in task
            assert task["id"] in (1, 2, 3, 4, 5)

    def test_post_reset_creates_episode(self, client: TestClient):
        """POST /reset creates a new episode and returns initial state."""
        response = client.post("/reset", json={"task_id": 1, "seed": 42})
        assert response.status_code == 200
        
        data = response.json()
        assert "episode_id" in data
        assert "observation" in data
        
        # Verify observation structure
        obs = data["observation"]
        assert obs["task_id"] == 1
        assert obs["step"] == 0
        assert obs["max_steps"] == 10
        assert "records" in obs
        assert "task_description" in obs
        assert "metadata" in obs

    def test_post_reset_different_tasks(self, client: TestClient):
        """POST /reset works for all task IDs."""
        for task_id in (1, 2, 3, 4):
            response = client.post("/reset", json={"task_id": task_id, "seed": 42})
            assert response.status_code == 200
            assert response.json()["observation"]["task_id"] == task_id

    def test_post_step_submits_action(self, client: TestClient, episode_id: str):
        """POST /step submits action and returns observation/reward/done."""
        # Submit with records
        action = {"records": [{"record_id": "test"}], "is_final": False}
        response = client.post("/step", params={"episode_id": episode_id}, json=action)
        assert response.status_code == 200
        
        data = response.json()
        assert "episode_id" in data
        assert "observation" in data
        assert "reward" in data
        assert "done" in data
        assert "info" in data
        assert isinstance(data["reward"], float)
        assert isinstance(data["done"], bool)

    def test_post_step_with_is_final(self, client: TestClient, episode_id: str):
        """POST /step with is_final=True ends the episode."""
        action = {"records": [{"record_id": "test"}], "is_final": True}
        response = client.post("/step", params={"episode_id": episode_id}, json=action)
        assert response.status_code == 200
        assert response.json()["done"] is True

    def test_get_state_returns_episode_state(self, client: TestClient, episode_id: str):
        """GET /state returns current episode state snapshot."""
        response = client.get("/state", params={"episode_id": episode_id})
        assert response.status_code == 200
        
        data = response.json()
        assert data["episode_id"] == episode_id
        assert "task_id" in data
        assert "seed" in data
        assert "step" in data
        assert "max_steps" in data
        assert "done" in data
        assert "passed" in data

    def test_get_grader_regrades_submission(self, client: TestClient, episode_id: str):
        """GET /grader re-grades the last submission."""
        # First make a submission
        action = {"records": [{"record_id": "test"}], "is_final": False}
        client.post("/step", params={"episode_id": episode_id}, json=action)
        
        # Then regrade
        response = client.get("/grader", params={"episode_id": episode_id})
        assert response.status_code == 200
        
        data = response.json()
        assert "episode_id" in data

    def test_get_grader_without_submission(self, client: TestClient, episode_id: str):
        """GET /grader without prior submission returns error."""
        response = client.get("/grader", params={"episode_id": episode_id})
        assert response.status_code == 200
        data = response.json()
        assert "error" in data

    def test_get_schema_returns_schemas(self, client: TestClient):
        """GET /schema returns action/observation/state schemas."""
        response = client.get("/schema")
        assert response.status_code == 200
        
        data = response.json()
        assert "action" in data
        assert "observation" in data
        assert "state" in data
        assert "reward" in data

    def test_get_contract_returns_env_contract(self, client: TestClient):
        """GET /contract returns OpenEnv-style environment contract sections."""
        response = client.get("/contract")
        assert response.status_code == 200

        data = response.json()
        assert data["mode"] == "agentic"
        assert "api_surface" in data
        assert "episode_structure" in data
        assert "schemas" in data
        assert "tasks" in data
        assert len(data["tasks"]) == 5

    def test_get_baseline_covers_all_tasks(self, client: TestClient):
        """GET /baseline runs the hybrid baseline across all 5 tasks."""
        response = client.get("/baseline")
        assert response.status_code == 200

        data = response.json()
        task_ids = {item.get("task_id") for item in data["results"] if "task_id" in item}
        assert task_ids == {1, 2, 3, 4, 5}
        assert "average_score" in data

    def test_get_metadata_returns_info(self, client: TestClient):
        """GET /metadata returns name and description."""
        response = client.get("/metadata")
        assert response.status_code == 200
        
        data = response.json()
        assert "name" in data
        assert "description" in data

    def test_get_mode_returns_agentic(self, client: TestClient):
        """GET /mode returns the operation mode."""
        response = client.get("/mode")
        assert response.status_code == 200
        assert response.json()["mode"] == "agentic"

    def test_get_health_returns_healthy(self, client: TestClient):
        """GET /health returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_invalid_episode_id_returns_404(self, client: TestClient):
        """Requests with invalid episode_id return 404."""
        fake_episode = str(uuid.uuid4())
        
        # Test /state
        response = client.get("/state", params={"episode_id": fake_episode})
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
        
        # Test /step
        response = client.post(
            "/step",
            params={"episode_id": fake_episode},
            json={"records": []}
        )
        assert response.status_code == 404
        
        # Test /grader
        response = client.get("/grader", params={"episode_id": fake_episode})
        assert response.status_code == 404

    def test_malformed_episode_id_returns_422(self, client: TestClient):
        """Malformed episode_id query params are rejected before lookup."""
        malformed = "not-a-uuid"

        response = client.get("/state", params={"episode_id": malformed})
        assert response.status_code == 422

        response = client.get("/grader", params={"episode_id": malformed})
        assert response.status_code == 422

        response = client.post("/step", params={"episode_id": malformed}, json={"records": []})
        assert response.status_code == 422

    def test_invalid_task_id_returns_400(self, client: TestClient):
        """POST /reset with invalid task_id returns 400."""
        response = client.post("/reset", json={"task_id": 99, "seed": 42})
        assert response.status_code == 400
        assert "task_id" in response.json()["detail"].lower()

    def test_task4_requires_knowledge_payload(self, client: TestClient, task4_episode_id: str):
        """Task 4 requires 'knowledge' payload, not 'records'."""
        action = {"records": [{"record_id": "test"}]}
        response = client.post(
            "/step",
            params={"episode_id": task4_episode_id},
            json=action
        )
        assert response.status_code == 422
        assert "knowledge" in response.json()["detail"].lower()

    def test_task4_step_with_knowledge(self, client: TestClient, task4_episode_id: str):
        """Task 4 step works with knowledge payload."""
        # Task 4 expects knowledge objects with entities and summary
        # Entities need 'type', 'name', and 'code' fields for the grader
        action = {
            "knowledge": [
                {
                    "entities": [
                        {"type": "condition", "name": "Diabetes", "code": "E11"}
                    ],
                    "summary": "Patient has diabetes."
                }
            ],
            "is_final": False
        }
        response = client.post(
            "/step",
            params={"episode_id": task4_episode_id},
            json=action
        )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# 2. Concurrent Episode Handling
# ---------------------------------------------------------------------------


class TestConcurrentEpisodes:
    """Test concurrent episode handling and isolation."""

    def test_multiple_episodes_simultaneously(self, client: TestClient):
        """Multiple episodes can run simultaneously."""
        episode_ids = []
        for i in range(5):
            response = client.post("/reset", json={"task_id": 1, "seed": 42 + i})
            assert response.status_code == 200
            episode_ids.append(response.json()["episode_id"])
        
        assert len(set(episode_ids)) == 5  # All unique
        
        # Verify all episodes are accessible
        for eid in episode_ids:
            response = client.get("/state", params={"episode_id": eid})
            assert response.status_code == 200

    def test_parallel_step_requests_different_episodes(self, client: TestClient):
        """Parallel /step requests on different episodes work independently."""
        # Create multiple episodes
        episode_ids = []
        for i in range(3):
            response = client.post("/reset", json={"task_id": 1, "seed": 42 + i})
            episode_ids.append(response.json()["episode_id"])
        
        results: dict[str, Any] = {}
        
        def step_episode(eid: str) -> tuple[str, int]:
            action = {"records": [{"record_id": f"test_{eid}"}], "is_final": False}
            resp = client.post("/step", params={"episode_id": eid}, json=action)
            return eid, resp.status_code
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(step_episode, eid): eid for eid in episode_ids}
            for future in as_completed(futures):
                eid, status = future.result()
                results[eid] = status
        
        # All should succeed
        assert all(status == 200 for status in results.values())

    def test_episode_isolation(self, client: TestClient):
        """Actions on one episode don't affect another."""
        # Create two episodes
        resp1 = client.post("/reset", json={"task_id": 1, "seed": 42})
        resp2 = client.post("/reset", json={"task_id": 2, "seed": 43})
        eid1 = resp1.json()["episode_id"]
        eid2 = resp2.json()["episode_id"]
        
        # Step on episode 1
        action = {"records": [{"record_id": "test1"}], "is_final": False}
        client.post("/step", params={"episode_id": eid1}, json=action)
        
        # Get states
        state1 = client.get("/state", params={"episode_id": eid1}).json()
        state2 = client.get("/state", params={"episode_id": eid2}).json()
        
        # Episode 1 should have step=1, episode 2 should have step=0
        assert state1["step"] == 1
        assert state2["step"] == 0
        assert state1["task_id"] == 1
        assert state2["task_id"] == 2

    def test_concurrent_steps_same_episode(self, client: TestClient, episode_id: str):
        """Concurrent steps on same episode are serialized via lock."""
        results: list[int] = []
        errors: list[Exception] = []
        
        def step_action(step_num: int) -> None:
            try:
                action = {"records": [{"record_id": f"step_{step_num}"}], "is_final": False}
                resp = client.post("/step", params={"episode_id": episode_id}, json=action)
                results.append(resp.status_code)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=step_action, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        # Some may succeed (200), some may fail (400 if episode done)
        # The key is no crashes or race conditions


# ---------------------------------------------------------------------------
# 3. Episode Expiry Logic
# ---------------------------------------------------------------------------


class TestEpisodeExpiry:
    """Test episode expiry and purge mechanisms."""

    def test_episode_expires_after_ttl(self, client: TestClient, episode_id: str):
        """Episodes expire after EPISODE_TTL_SECONDS of inactivity."""
        # Patch the episode's last_used time to simulate expiry
        with _episodes_lock:
            entry = _episodes.get(episode_id)
            assert entry is not None
            # Set last_used to far in the past
            entry.last_used = time.time() - EPISODE_TTL_SECONDS - 10
        
        # Now accessing should return 404
        response = client.get("/state", params={"episode_id": episode_id})
        assert response.status_code == 404
        assert "expired" in response.json()["detail"].lower()

    def test_purge_mechanism_removes_expired(self, client: TestClient):
        """The purge mechanism removes expired episodes."""
        # Create episodes
        episode_ids = []
        for i in range(3):
            resp = client.post("/reset", json={"task_id": 1, "seed": i})
            episode_ids.append(resp.json()["episode_id"])
        
        # Expire first two
        with _episodes_lock:
            for eid in episode_ids[:2]:
                _episodes[eid].last_used = time.time() - EPISODE_TTL_SECONDS - 10
        
        # Trigger purge
        _purge_expired()
        
        # First two should be gone
        with _episodes_lock:
            assert episode_ids[0] not in _episodes
            assert episode_ids[1] not in _episodes
            assert episode_ids[2] in _episodes  # Still valid

    def test_expired_episode_returns_appropriate_error(self, client: TestClient, episode_id: str):
        """Expired episodes return informative error messages."""
        with _episodes_lock:
            _episodes[episode_id].last_used = time.time() - EPISODE_TTL_SECONDS - 10
        
        response = client.get("/state", params={"episode_id": episode_id})
        assert response.status_code == 404
        
        detail = response.json()["detail"]
        assert "expired" in detail.lower()
        assert "reset" in detail.lower()  # Should mention calling /reset

    def test_activity_refreshes_ttl(self, client: TestClient, episode_id: str):
        """Accessing an episode refreshes its TTL."""
        # Set last_used close to expiry threshold
        almost_expired = time.time() - EPISODE_TTL_SECONDS + 60
        with _episodes_lock:
            _episodes[episode_id].last_used = almost_expired
        
        # Access the episode
        response = client.get("/state", params={"episode_id": episode_id})
        assert response.status_code == 200
        
        # last_used should be updated
        with _episodes_lock:
            entry = _episodes[episode_id]
            assert entry.last_used > almost_expired

    def test_edge_case_exact_ttl_boundary(self, client: TestClient, episode_id: str):
        """Test behavior at exact TTL boundary."""
        # Set exactly at boundary
        with _episodes_lock:
            _episodes[episode_id].last_used = time.time() - EPISODE_TTL_SECONDS
        
        # The check is `> EPISODE_TTL_SECONDS`, so exact boundary means it's expired
        response = client.get("/state", params={"episode_id": episode_id})
        assert response.status_code == 404  # Expired at exact boundary


# ---------------------------------------------------------------------------
# 4. Rate Limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """Test per-IP rate limiting on /reset endpoint."""

    def test_rate_limit_triggers_after_max_requests(self, client: TestClient):
        """Rate limiting kicks in after RATE_LIMIT_REQUESTS."""
        # Make RATE_LIMIT_REQUESTS successful requests
        for i in range(RATE_LIMIT_REQUESTS):
            response = client.post("/reset", json={"task_id": 1, "seed": i})
            assert response.status_code == 200, f"Request {i+1} failed unexpectedly"
        
        # Next request should be rate limited
        response = client.post("/reset", json={"task_id": 1, "seed": 999})
        assert response.status_code == 429
        assert "rate limit" in response.json()["detail"].lower()

    def test_rate_limit_429_response_format(self, client: TestClient):
        """Rate limited response has correct format."""
        # Exhaust rate limit
        for i in range(RATE_LIMIT_REQUESTS):
            client.post("/reset", json={"task_id": 1, "seed": i})
        
        response = client.post("/reset", json={"task_id": 1, "seed": 999})
        assert response.status_code == 429
        
        detail = response.json()["detail"]
        assert str(RATE_LIMIT_REQUESTS) in detail
        assert str(RATE_LIMIT_WINDOW_SECONDS) in detail

    def test_retry_after_header(self, client: TestClient):
        """Rate limited response includes Retry-After header."""
        # Exhaust rate limit
        for i in range(RATE_LIMIT_REQUESTS):
            client.post("/reset", json={"task_id": 1, "seed": i})
        
        response = client.post("/reset", json={"task_id": 1, "seed": 999})
        assert response.status_code == 429
        assert "retry-after" in response.headers
        retry_after = int(response.headers["retry-after"])
        assert retry_after > 0
        assert retry_after <= RATE_LIMIT_WINDOW_SECONDS + 1

    def test_different_ips_independent_limits(self):
        """Different IPs have independent rate limits."""
        # Use patched client IPs via mock
        with patch("src.api.Request") as MockRequest:
            # We'll test the rate limiter directly instead
            from src.api import _check_rate_limit, _rate_limit_store
            
            # Clear rate limit store
            with _rate_limit_lock:
                _rate_limit_store.clear()
            
            # IP1 makes requests
            for _ in range(RATE_LIMIT_REQUESTS):
                assert _check_rate_limit("192.168.1.1") is True
            assert _check_rate_limit("192.168.1.1") is False  # Now limited
            
            # IP2 should still be able to make requests
            for _ in range(RATE_LIMIT_REQUESTS):
                assert _check_rate_limit("192.168.1.2") is True
            assert _check_rate_limit("192.168.1.2") is False  # Now limited

    def test_rate_limit_window_expiry(self):
        """Rate limit resets after window expires."""
        from src.api import _check_rate_limit
        
        with _rate_limit_lock:
            _rate_limit_store.clear()
        
        # Exhaust limit for an IP
        for _ in range(RATE_LIMIT_REQUESTS):
            _check_rate_limit("10.0.0.1")
        assert _check_rate_limit("10.0.0.1") is False
        
        # Manually expire the timestamps
        with _rate_limit_lock:
            timestamps = _rate_limit_store["10.0.0.1"]
            # Move all timestamps outside the window
            old_time = time.time() - RATE_LIMIT_WINDOW_SECONDS - 10
            timestamps.clear()
            for _ in range(RATE_LIMIT_REQUESTS):
                timestamps.append(old_time)
        
        # Should be allowed now
        assert _check_rate_limit("10.0.0.1") is True

    def test_inactive_rate_limit_entries_are_pruned(self, monkeypatch: pytest.MonkeyPatch):
        """Inactive IPs older than TTL should be removed from both rate-limit stores."""
        now = 1_700_000_000.0
        monkeypatch.setattr("src.api.time.time", lambda: now)

        old_ts = now - RATE_LIMIT_ENTRY_TTL_SECONDS - 10
        fresh_ts = now - 5

        with _rate_limit_lock:
            _rate_limit_store.clear()
            _rate_limit_store["inactive-ip"] = deque([old_ts])
            _rate_limit_store["active-ip"] = deque([fresh_ts])

        with _read_rate_limit_lock:
            _read_rate_limit_store.clear()
            _read_rate_limit_store["inactive-ip"] = deque([old_ts])
            _read_rate_limit_store["active-ip"] = deque([fresh_ts])

        _purge_stale_rate_limits()

        with _rate_limit_lock:
            assert "inactive-ip" not in _rate_limit_store
            assert "active-ip" in _rate_limit_store

        with _read_rate_limit_lock:
            assert "inactive-ip" not in _read_rate_limit_store
            assert "active-ip" in _read_rate_limit_store

    def test_max_episodes_limit(self, client: TestClient):
        """Test MAX_EPISODES limit prevents unbounded growth."""
        # Fill up to MAX_EPISODES (but we need to be careful about rate limiting)
        # Skip rate limiting by manipulating store
        
        # Create episodes by directly manipulating the store
        from src.api import EpisodeEntry
        from src.environment import MedicalOpenEnv
        
        with _episodes_lock:
            for i in range(MAX_EPISODES):
                eid = f"test-episode-{i}"
                env = MedicalOpenEnv()
                env.reset(task_id=1, seed=i)
                _episodes[eid] = EpisodeEntry(env=env)
        
        # Clear rate limit to allow the request
        with _rate_limit_lock:
            _rate_limit_store.clear()
        
        # Try to create one more - should fail with 429 (too many episodes)
        response = client.post("/reset", json={"task_id": 1, "seed": 999})
        assert response.status_code == 429
        assert "active episodes" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 4b. CORS Configuration
# ---------------------------------------------------------------------------


class TestCORSConfiguration:
    """Validate strict CORS policy behavior."""

    def test_production_defaults_when_origins_unset(self, monkeypatch: pytest.MonkeyPatch):
        """Production mode should fall back to localhost defaults when CORS_ORIGINS is unset."""
        monkeypatch.setattr("src.api.OPENENV_ENV", "production")
        monkeypatch.delenv("CORS_ORIGINS", raising=False)
        assert _get_cors_origins() == ["http://localhost:7860", "http://127.0.0.1:7860"]

    def test_wildcard_origin_is_rejected(self, monkeypatch: pytest.MonkeyPatch):
        """Wildcard origins are never allowed."""
        monkeypatch.setattr("src.api.OPENENV_ENV", "production")
        monkeypatch.setenv("CORS_ORIGINS", "https://example.com,*")

        with pytest.raises(ValueError, match="wildcard"):
            _get_cors_origins()


# ---------------------------------------------------------------------------
# 4c. API Auth Configuration
# ---------------------------------------------------------------------------


class TestAPIAuthConfiguration:
    """Validate configurable API key authentication policy."""

    def test_missing_key_allowed_when_not_required(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """When auth is optional, missing OPENENV_API_KEY should not block requests."""
        monkeypatch.setattr("src.api.OPENENV_API_KEY", "")
        monkeypatch.setattr("src.api.OPENENV_REQUIRE_API_KEY", False)

        response = client.get("/tasks")
        assert response.status_code == 200

    def test_missing_key_rejected_when_required(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """When auth is required, missing OPENENV_API_KEY should return a config error."""
        monkeypatch.setattr("src.api.OPENENV_API_KEY", "")
        monkeypatch.setattr("src.api.OPENENV_REQUIRE_API_KEY", True)

        response = client.get("/tasks")
        assert response.status_code == 503
        assert response.json().get("error_type") == "auth_configuration_error"

    def test_configured_key_is_enforced(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """When a key is configured, protected endpoints require matching credentials."""
        monkeypatch.setattr("src.api.OPENENV_API_KEY", "test-key")
        monkeypatch.setattr("src.api.OPENENV_REQUIRE_API_KEY", True)

        unauthorized = client.get("/tasks")
        assert unauthorized.status_code == 401

        authorized = client.get("/tasks", headers={"X-API-Key": "test-key"})
        assert authorized.status_code == 200

    def test_gradio_paths_marked_public(self):
        """Gradio root/queue/static paths should bypass auth middleware checks."""
        assert _is_public_path("/")
        assert _is_public_path("/gradio_api/queue/join")
        assert _is_public_path("/assets/index.css")
        assert not _is_public_path("/tasks")


# ---------------------------------------------------------------------------
# 5. Error Recovery Scenarios
# ---------------------------------------------------------------------------


class TestErrorRecovery:
    """Test error handling and recovery scenarios."""

    def test_grader_error_structured_response(self, client: TestClient, episode_id: str):
        """GraderError produces structured error response."""
        # We need to trigger a grader error - mock the grading function
        with patch("src.environment.task1_hygiene.grade") as mock_grade:
            mock_grade.side_effect = Exception("Simulated grader failure")
            
            action = {"records": [{"record_id": "test"}], "is_final": False}
            response = client.post("/step", params={"episode_id": episode_id}, json=action)
            
            assert response.status_code == 500
            detail = response.json()["detail"]
            assert detail["error_type"] == "grader_error"
            assert "task_id" in detail
            assert "episode_id" in detail
            assert detail["episode_id"] == episode_id

    def test_invalid_action_payload_422(self, client: TestClient, episode_id: str):
        """Invalid action payloads return 422."""
        # Task 1 requires 'records'
        action = {"records": None}  # Explicitly null
        response = client.post("/step", params={"episode_id": episode_id}, json=action)
        assert response.status_code == 422

    def test_submitting_on_completed_episode(self, client: TestClient, episode_id: str):
        """Submitting actions on completed episodes returns error."""
        # Complete the episode
        action = {"records": [{"record_id": "test"}], "is_final": True}
        response = client.post("/step", params={"episode_id": episode_id}, json=action)
        assert response.status_code == 200
        assert response.json()["done"] is True
        
        # Try to step again
        action = {"records": [{"record_id": "test2"}], "is_final": False}
        response = client.post("/step", params={"episode_id": episode_id}, json=action)
        assert response.status_code == 400
        assert "done" in response.json()["detail"].lower()

    def test_malformed_json_body(self, client: TestClient, episode_id: str):
        """Malformed JSON returns appropriate error."""
        response = client.post(
            "/step",
            params={"episode_id": episode_id},
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_missing_required_fields(self, client: TestClient, episode_id: str):
        """Missing required fields in request body handled properly."""
        # Empty body
        response = client.post(
            "/step",
            params={"episode_id": episode_id},
            json={}
        )
        # FastAPI will use defaults, but records=None means validation fails
        assert response.status_code == 422

    def test_episode_state_consistency_after_error(self, client: TestClient, episode_id: str):
        """Episode state remains consistent after grader error."""
        # Get initial state
        initial_state = client.get("/state", params={"episode_id": episode_id}).json()
        initial_step = initial_state["step"]
        
        # Force a grader error
        with patch("src.environment.task1_hygiene.grade") as mock_grade:
            mock_grade.side_effect = Exception("Simulated failure")
            
            action = {"records": [{"record_id": "test"}], "is_final": False}
            response = client.post("/step", params={"episode_id": episode_id}, json=action)
            assert response.status_code == 500
        
        # State should be restored (step should be rolled back)
        final_state = client.get("/state", params={"episode_id": episode_id}).json()
        # Note: The rollback attempts to restore _current_step
        assert final_state["done"] is False

    def test_grader_error_contains_context(self, client: TestClient, episode_id: str):
        """GraderError response contains useful context for debugging."""
        with patch("src.environment.task1_hygiene.grade") as mock_grade:
            # Use TypeError (not ValueError or RuntimeError) to trigger the generic exception handler
            # ValueError -> 422, RuntimeError -> 400, generic Exception -> 500
            mock_grade.side_effect = TypeError("Grader internal failure")
            
            action = {"records": [{"record_id": "test1"}, {"record_id": "test2"}], "is_final": False}
            response = client.post("/step", params={"episode_id": episode_id}, json=action)
            
            assert response.status_code == 500
            detail = response.json()["detail"]
            assert detail["task_id"] == 1
            assert detail["episode_id"] == episode_id
            # Should have details about the error
            assert detail["details"] is not None

    def test_grader_error_class_to_dict(self):
        """GraderError.to_dict() produces correct structure."""
        original = ValueError("Original error message")
        error = GraderError(
            message="Grading failed",
            task_id=2,
            episode_id="test-123",
            input_summary="records=5 items",
            original_error=original
        )
        
        result = error.to_dict()
        assert result["error_type"] == "grader_error"
        assert result["message"] == "Grading failed"
        assert result["task_id"] == 2
        assert result["episode_id"] == "test-123"
        assert "Original error message" in result["details"]


# ---------------------------------------------------------------------------
# Additional Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test additional edge cases and boundary conditions."""

    def test_reset_with_default_parameters(self, client: TestClient):
        """POST /reset works with default parameters."""
        response = client.post("/reset", json={})
        assert response.status_code == 200
        obs = response.json()["observation"]
        assert obs["task_id"] == 1  # Default
        assert obs["metadata"]["seed"] == 42  # Default

    def test_max_steps_ends_episode(self, client: TestClient, episode_id: str):
        """Episode ends automatically after max_steps."""
        # Submit steps until done
        for i in range(15):  # More than max_steps (10)
            action = {"records": [{"record_id": f"step_{i}"}], "is_final": False}
            response = client.post("/step", params={"episode_id": episode_id}, json=action)
            
            if response.status_code == 400:
                # Episode is done
                break
            
            if response.json().get("done"):
                break
        
        # Verify episode is done
        state = client.get("/state", params={"episode_id": episode_id}).json()
        assert state["done"] is True

    def test_unicode_in_records(self, client: TestClient, episode_id: str):
        """Unicode characters in records are handled correctly."""
        action = {
            "records": [{"record_id": "test", "patient_name": "日本語テスト"}],
            "is_final": False
        }
        response = client.post("/step", params={"episode_id": episode_id}, json=action)
        assert response.status_code == 200

    def test_large_payload(self, client: TestClient, episode_id: str):
        """Large payloads are handled correctly."""
        # Create a moderately large payload
        records = [{"record_id": f"rec_{i}", "data": "x" * 1000} for i in range(50)]
        action = {"records": records, "is_final": False}
        
        response = client.post("/step", params={"episode_id": episode_id}, json=action)
        # Should either succeed or fail with validation, not crash
        assert response.status_code in (200, 422)

    def test_empty_episode_id_parameter(self, client: TestClient):
        """Empty episode_id parameter handled appropriately."""
        response = client.get("/state", params={"episode_id": ""})
        assert response.status_code in (400, 404, 422)

    def test_openapi_json_endpoint(self, client: TestClient):
        """GET /openapi.json returns valid OpenAPI spec."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data

    def test_step_increments_correctly(self, client: TestClient, episode_id: str):
        """Step counter increments correctly with each action."""
        for expected_step in range(1, 4):
            action = {"records": [{"record_id": f"step_{expected_step}"}], "is_final": False}
            response = client.post("/step", params={"episode_id": episode_id}, json=action)
            assert response.status_code == 200
            
            obs = response.json()["observation"]
            assert obs["step"] == expected_step

    def test_reward_score_bounds(self, client: TestClient, episode_id: str):
        """Reward scores stay strictly inside (0, 1) per OpenEnv validation."""
        action = {"records": [{"record_id": "test"}], "is_final": False}
        response = client.post("/step", params={"episode_id": episode_id}, json=action)
        assert response.status_code == 200
        
        reward = response.json()["reward"]
        assert 0.0 < reward < 1.0

    def test_info_contains_breakdown(self, client: TestClient, episode_id: str):
        """Step response info contains breakdown details."""
        action = {"records": [{"record_id": "test"}], "is_final": False}
        response = client.post("/step", params={"episode_id": episode_id}, json=action)
        assert response.status_code == 200
        
        info = response.json()["info"]
        assert "breakdown" in info
        assert "grader_info" in info
