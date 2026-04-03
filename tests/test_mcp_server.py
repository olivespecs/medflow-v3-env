"""Tests for MCP server (src/mcp_server.py)."""

import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mcp_server import (
    MAX_MCP_EPISODES,
    MCP_EPISODE_TTL_SECONDS,
    MCPEpisodeEntry,
    _get_mcp_episode,
    _mcp_episodes,
    _mcp_episodes_lock,
    _purge_mcp_expired,
    mcp,
    reset,
    state,
    step,
)
from src.environment import MedicalOpenEnv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_episodes():
    """Clear MCP episode store before and after each test."""
    with _mcp_episodes_lock:
        _mcp_episodes.clear()
    yield
    with _mcp_episodes_lock:
        _mcp_episodes.clear()


# ===========================================================================
# 1. MCP Tool Registration Tests
# ===========================================================================

class TestMCPToolRegistration:
    """Verify all expected tools are registered with correct metadata."""

    def test_mcp_instance_exists(self):
        """Verify the FastMCP instance is created with correct name."""
        assert mcp is not None
        assert mcp.name == "Medical Records OpenEnv"

    def test_reset_tool_is_registered(self):
        """Verify reset tool is registered."""
        # The reset function should be callable and decorated
        assert callable(reset)
        # Check that it has the expected signature
        import inspect
        sig = inspect.signature(reset)
        params = list(sig.parameters.keys())
        assert "task_id" in params
        assert "seed" in params

    def test_step_tool_is_registered(self):
        """Verify step tool is registered."""
        assert callable(step)
        import inspect
        sig = inspect.signature(step)
        params = list(sig.parameters.keys())
        assert "episode_id" in params
        assert "records" in params
        assert "knowledge" in params
        assert "is_final" in params

    def test_state_tool_is_registered(self):
        """Verify state tool is registered."""
        assert callable(state)
        import inspect
        sig = inspect.signature(state)
        params = list(sig.parameters.keys())
        assert "episode_id" in params

    def test_tool_default_parameters(self):
        """Verify tools have correct default parameter values."""
        import inspect
        
        # reset defaults
        reset_sig = inspect.signature(reset)
        assert reset_sig.parameters["task_id"].default == 1
        assert reset_sig.parameters["seed"].default == 42
        
        # step defaults
        step_sig = inspect.signature(step)
        assert step_sig.parameters["records"].default is None
        assert step_sig.parameters["knowledge"].default is None
        assert step_sig.parameters["is_final"].default is False


# ===========================================================================
# 2. Tool Invocation Tests
# ===========================================================================

class TestResetTool:
    """Test the reset MCP tool."""

    def test_reset_returns_episode_id(self):
        """Reset should return a unique episode_id."""
        result = reset(task_id=1, seed=42)
        assert "episode_id" in result
        assert isinstance(result["episode_id"], str)
        assert len(result["episode_id"]) > 0

    def test_reset_returns_observation(self):
        """Reset should return an observation dict."""
        result = reset(task_id=1, seed=42)
        assert "observation" in result
        obs = result["observation"]
        assert obs["task_id"] == 1
        assert "task_description" in obs
        assert "records" in obs
        assert obs["step"] == 0
        assert obs["max_steps"] == 10

    def test_reset_creates_episode_entry(self):
        """Reset should store the episode in the internal store."""
        result = reset(task_id=2, seed=123)
        episode_id = result["episode_id"]
        with _mcp_episodes_lock:
            assert episode_id in _mcp_episodes
            entry = _mcp_episodes[episode_id]
            assert isinstance(entry, MCPEpisodeEntry)
            assert isinstance(entry.env, MedicalOpenEnv)

    def test_reset_all_task_ids(self):
        """Reset should work for all valid task IDs (1-5)."""
        for task_id in (1, 2, 3, 4, 5):
            result = reset(task_id=task_id, seed=42)
            assert "error" not in result
            obs = result["observation"]
            assert obs["task_id"] == task_id
            assert len(obs["records"]) > 0

    def test_reset_different_seeds_produce_different_data(self):
        """Different seeds should produce different records."""
        result1 = reset(task_id=1, seed=1)
        result2 = reset(task_id=1, seed=999)
        # Records should differ (at minimum record_ids will be different)
        assert result1["observation"]["records"] != result2["observation"]["records"]

    def test_reset_generates_unique_episode_ids(self):
        """Multiple resets should generate unique episode IDs."""
        ids = set()
        for _ in range(10):
            result = reset(task_id=1, seed=42)
            ids.add(result["episode_id"])
        assert len(ids) == 10

    def test_reset_max_episodes_limit(self):
        """Reset should error when max episodes is reached."""
        # Fill up the episode store
        with _mcp_episodes_lock:
            for i in range(MAX_MCP_EPISODES):
                env = MedicalOpenEnv()
                _mcp_episodes[f"fake-episode-{i}"] = MCPEpisodeEntry(env=env)
        
        result = reset(task_id=1, seed=42)
        assert "error" in result
        assert "Too many active MCP episodes" in result["error"]


class TestStepTool:
    """Test the step MCP tool."""

    def test_step_with_valid_episode(self):
        """Step should work with a valid episode_id."""
        reset_result = reset(task_id=1, seed=42)
        episode_id = reset_result["episode_id"]
        records = reset_result["observation"]["records"]
        
        result = step(episode_id=episode_id, records=records)
        assert "error" not in result
        assert "observation" in result
        assert "reward" in result
        assert "done" in result
        assert "info" in result

    def test_step_returns_valid_reward(self):
        """Step should return a reward between 0 and 1."""
        reset_result = reset(task_id=1, seed=42)
        episode_id = reset_result["episode_id"]
        records = reset_result["observation"]["records"]
        
        result = step(episode_id=episode_id, records=records)
        assert 0.0 <= result["reward"] <= 1.0

    def test_step_increments_step_count(self):
        """Step should increment the step count in observation."""
        reset_result = reset(task_id=1, seed=42)
        episode_id = reset_result["episode_id"]
        records = reset_result["observation"]["records"]
        
        result = step(episode_id=episode_id, records=records)
        assert result["observation"]["step"] == 1
        
        result2 = step(episode_id=episode_id, records=records)
        assert result2["observation"]["step"] == 2

    def test_step_is_final_ends_episode(self):
        """Step with is_final=True should end the episode."""
        reset_result = reset(task_id=1, seed=42)
        episode_id = reset_result["episode_id"]
        records = reset_result["observation"]["records"]
        
        result = step(episode_id=episode_id, records=records, is_final=True)
        assert result["done"] is True

    def test_step_with_empty_records(self):
        """Step should accept empty records list."""
        reset_result = reset(task_id=1, seed=42)
        episode_id = reset_result["episode_id"]
        
        result = step(episode_id=episode_id, records=[])
        assert "error" not in result
        # Score should be low but not error
        assert result["reward"] >= 0.0

    def test_step_for_task4_with_knowledge(self):
        """Step for task 4 should accept knowledge parameter."""
        reset_result = reset(task_id=4, seed=42)
        episode_id = reset_result["episode_id"]
        
        knowledge = [
            {"entities": [], "summary": "Test summary"}
            for _ in reset_result["observation"]["records"]
        ]
        
        result = step(episode_id=episode_id, knowledge=knowledge)
        assert "error" not in result
        assert "reward" in result

    def test_step_returns_breakdown(self):
        """Step should return reward breakdown in info."""
        reset_result = reset(task_id=1, seed=42)
        episode_id = reset_result["episode_id"]
        records = reset_result["observation"]["records"]
        
        result = step(episode_id=episode_id, records=records)
        assert "breakdown" in result["info"]


class TestStateTool:
    """Test the state MCP tool."""

    def test_state_returns_episode_info(self):
        """State should return current episode information."""
        reset_result = reset(task_id=1, seed=42)
        episode_id = reset_result["episode_id"]
        
        result = state(episode_id=episode_id)
        assert "error" not in result
        assert result["task_id"] == 1
        assert result["seed"] == 42
        assert result["step"] == 0
        assert result["done"] is False

    def test_state_reflects_step_progress(self):
        """State should reflect progress after steps."""
        reset_result = reset(task_id=1, seed=42)
        episode_id = reset_result["episode_id"]
        records = reset_result["observation"]["records"]
        
        step(episode_id=episode_id, records=records)
        step(episode_id=episode_id, records=records)
        
        result = state(episode_id=episode_id)
        assert result["step"] == 2
        assert result["last_score"] is not None

    def test_state_includes_audit_trail(self):
        """State should include audit trail of all steps."""
        reset_result = reset(task_id=1, seed=42)
        episode_id = reset_result["episode_id"]
        records = reset_result["observation"]["records"]
        
        step(episode_id=episode_id, records=records)
        step(episode_id=episode_id, records=records)
        
        result = state(episode_id=episode_id)
        assert "audit_trail" in result
        assert len(result["audit_trail"]) == 2
        assert result["audit_trail"][0]["step"] == 1
        assert result["audit_trail"][1]["step"] == 2


# ===========================================================================
# 3. Request/Response Handling Tests
# ===========================================================================

class TestResponseFormats:
    """Verify correct response formats for all tools."""

    def test_reset_response_structure(self):
        """Reset response should have the correct structure."""
        result = reset(task_id=1, seed=42)
        
        assert set(result.keys()) == {"episode_id", "observation"}
        obs = result["observation"]
        assert "task_id" in obs
        assert "task_description" in obs
        assert "records" in obs
        assert "step" in obs
        assert "max_steps" in obs
        assert "metadata" in obs

    def test_step_response_structure(self):
        """Step response should have the correct structure."""
        reset_result = reset(task_id=1, seed=42)
        episode_id = reset_result["episode_id"]
        records = reset_result["observation"]["records"]
        
        result = step(episode_id=episode_id, records=records)
        
        assert "observation" in result
        assert "reward" in result
        assert "done" in result
        assert "info" in result
        assert "breakdown" in result["info"]

    def test_state_response_structure(self):
        """State response should have the correct structure."""
        reset_result = reset(task_id=1, seed=42)
        episode_id = reset_result["episode_id"]
        
        result = state(episode_id=episode_id)
        
        expected_keys = {
            "task_id", "seed", "step", "max_steps", "done",
            "last_score", "last_breakdown", "passed", "audit_trail"
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_observation_records_are_dicts(self):
        """Observation records should be dictionaries."""
        result = reset(task_id=1, seed=42)
        records = result["observation"]["records"]
        
        assert isinstance(records, list)
        for rec in records:
            assert isinstance(rec, dict)
            assert "record_id" in rec

    def test_reward_is_float(self):
        """Reward should be a float value."""
        reset_result = reset(task_id=1, seed=42)
        episode_id = reset_result["episode_id"]
        records = reset_result["observation"]["records"]
        
        result = step(episode_id=episode_id, records=records)
        assert isinstance(result["reward"], float)

    def test_done_is_boolean(self):
        """Done should be a boolean value."""
        reset_result = reset(task_id=1, seed=42)
        episode_id = reset_result["episode_id"]
        records = reset_result["observation"]["records"]
        
        result = step(episode_id=episode_id, records=records)
        assert isinstance(result["done"], bool)


class TestEdgeCaseInputs:
    """Test edge case inputs."""

    def test_reset_with_boundary_seeds(self):
        """Reset should work with boundary seed values."""
        for seed in (0, 1, 2**31 - 1):
            result = reset(task_id=1, seed=seed)
            assert "error" not in result

    def test_step_with_none_records(self):
        """Step should handle None records gracefully."""
        reset_result = reset(task_id=1, seed=42)
        episode_id = reset_result["episode_id"]
        
        result = step(episode_id=episode_id, records=None)
        assert "error" not in result

    def test_step_with_partial_record_data(self):
        """Step should handle records with missing optional fields."""
        reset_result = reset(task_id=1, seed=42)
        episode_id = reset_result["episode_id"]
        
        partial_records = [
            {"record_id": "test-1", "mrn": "123", "patient_name": "John",
             "dob": "2000-01-01", "gender": "M"}
        ]
        
        result = step(episode_id=episode_id, records=partial_records)
        assert "error" not in result

    def test_step_with_extra_record_fields(self):
        """Step should handle records with extra fields."""
        reset_result = reset(task_id=1, seed=42)
        episode_id = reset_result["episode_id"]
        records = reset_result["observation"]["records"]
        
        # Add extra field
        for rec in records:
            rec["extra_field"] = "extra_value"
        
        result = step(episode_id=episode_id, records=records)
        assert "error" not in result


# ===========================================================================
# 4. Error Cases Tests
# ===========================================================================

class TestInvalidTaskId:
    """Test invalid task ID handling."""

    def test_reset_with_invalid_task_id_zero(self):
        """Reset should handle task_id=0."""
        with pytest.raises(ValueError, match="task_id must be 1, 2, 3, 4, or 5"):
            reset(task_id=0, seed=42)

    def test_reset_with_invalid_task_id_five(self):
        """Reset should handle task_id=6 and beyond."""
        with pytest.raises(ValueError, match="task_id must be 1, 2, 3, 4, or 5"):
            reset(task_id=6, seed=42)

    def test_reset_with_negative_task_id(self):
        """Reset should handle negative task_id."""
        with pytest.raises(ValueError, match="task_id must be 1, 2, 3, 4, or 5"):
            reset(task_id=-1, seed=42)


class TestInvalidEpisodeId:

    def test_step_with_nonexistent_episode(self):
        """Step should return error for nonexistent episode."""
        result = step(episode_id="11111111-1111-1111-1111-111111111111", records=[])
        assert "error" in result
        assert "not found" in result["error"]

    def test_state_with_nonexistent_episode(self):
        """State should return error for nonexistent episode."""
        result = state(episode_id="11111111-1111-1111-1111-111111111111")
        assert "error" in result
        assert "not found" in result["error"]

    def test_step_with_empty_episode_id(self):
        """Step should handle empty episode_id."""
        result = step(episode_id="", records=[])
        assert "error" in result
        assert "Invalid episode_id format" in result["error"]

    def test_state_with_empty_episode_id(self):
        """State should handle empty episode_id."""
        result = state(episode_id="")
        assert "error" in result
        assert "Invalid episode_id format" in result["error"]


class TestExpiredEpisodes:
    """Test episode expiration handling."""

    def test_expired_episode_returns_error(self):
        """Accessing an expired episode should return an error."""
        reset_result = reset(task_id=1, seed=42)
        episode_id = reset_result["episode_id"]
        
        # Manually expire the episode
        with _mcp_episodes_lock:
            entry = _mcp_episodes[episode_id]
            entry.last_used = time.time() - MCP_EPISODE_TTL_SECONDS - 1
        
        result = step(episode_id=episode_id, records=[])
        assert "error" in result
        assert "expired" in result["error"]

    def test_purge_removes_expired_episodes(self):
        """Purge should remove expired episodes."""
        reset_result = reset(task_id=1, seed=42)
        episode_id = reset_result["episode_id"]
        
        # Manually expire the episode
        with _mcp_episodes_lock:
            entry = _mcp_episodes[episode_id]
            entry.last_used = time.time() - MCP_EPISODE_TTL_SECONDS - 1
        
        _purge_mcp_expired()
        
        with _mcp_episodes_lock:
            assert episode_id not in _mcp_episodes


class TestMCPEpisodeEntry:
    """Test MCPEpisodeEntry internal class."""

    def test_touch_updates_last_used(self):
        """Touch should update last_used timestamp."""
        env = MedicalOpenEnv()
        entry = MCPEpisodeEntry(env=env)
        old_time = entry.last_used
        
        time.sleep(0.01)
        entry.touch()
        
        assert entry.last_used > old_time

    def test_is_expired_false_when_fresh(self):
        """Fresh episode should not be expired."""
        env = MedicalOpenEnv()
        entry = MCPEpisodeEntry(env=env)
        assert entry.is_expired() is False

    def test_is_expired_true_after_ttl(self):
        """Episode should be expired after TTL."""
        env = MedicalOpenEnv()
        entry = MCPEpisodeEntry(env=env)
        entry.last_used = time.time() - MCP_EPISODE_TTL_SECONDS - 1
        assert entry.is_expired() is True


class TestGetMCPEpisode:
    """Test _get_mcp_episode helper function."""

    def test_get_existing_episode(self):
        """Should return entry for existing episode."""
        reset_result = reset(task_id=1, seed=42)
        episode_id = reset_result["episode_id"]
        
        entry = _get_mcp_episode(episode_id)
        assert isinstance(entry, MCPEpisodeEntry)

    def test_get_nonexistent_episode_raises(self):
        """Should raise ValueError for nonexistent episode."""
        with pytest.raises(ValueError, match="not found"):
            _get_mcp_episode("11111111-1111-1111-1111-111111111111")

    def test_get_expired_episode_raises(self):
        """Should raise ValueError for expired episode."""
        reset_result = reset(task_id=1, seed=42)
        episode_id = reset_result["episode_id"]
        
        # Manually expire
        with _mcp_episodes_lock:
            _mcp_episodes[episode_id].last_used = time.time() - MCP_EPISODE_TTL_SECONDS - 1
        
        with pytest.raises(ValueError, match="expired"):
            _get_mcp_episode(episode_id)

    def test_get_episode_touches_entry(self):
        """Getting an episode should update its last_used time."""
        reset_result = reset(task_id=1, seed=42)
        episode_id = reset_result["episode_id"]
        
        with _mcp_episodes_lock:
            entry = _mcp_episodes[episode_id]
            old_time = entry.last_used
        
        time.sleep(0.01)
        _get_mcp_episode(episode_id)
        
        with _mcp_episodes_lock:
            assert _mcp_episodes[episode_id].last_used > old_time


# ===========================================================================
# Integration Tests
# ===========================================================================

class TestFullWorkflow:
    """Test complete MCP workflows."""

    def test_complete_episode_workflow(self):
        """Test a complete episode from reset to done."""
        # Reset
        reset_result = reset(task_id=1, seed=42)
        episode_id = reset_result["episode_id"]
        records = reset_result["observation"]["records"]
        
        # Multiple steps
        for i in range(3):
            step_result = step(episode_id=episode_id, records=records)
            assert step_result["observation"]["step"] == i + 1
            assert step_result["done"] is False
        
        # Check state
        state_result = state(episode_id=episode_id)
        assert state_result["step"] == 3
        assert len(state_result["audit_trail"]) == 3
        
        # Final step
        final_result = step(episode_id=episode_id, records=records, is_final=True)
        assert final_result["done"] is True

    def test_multiple_concurrent_episodes(self):
        """Test multiple episodes running concurrently."""
        episodes = []
        for task_id in (1, 2, 3, 4):
            result = reset(task_id=task_id, seed=task_id * 10)
            episodes.append({
                "id": result["episode_id"],
                "task_id": task_id,
                "records": result["observation"]["records"]
            })
        
        # Step each episode
        for ep in episodes:
            if ep["task_id"] == 4:
                result = step(episode_id=ep["id"], knowledge=[])
            else:
                result = step(episode_id=ep["id"], records=ep["records"])
            assert "error" not in result
        
        # Check states are independent
        for ep in episodes:
            result = state(episode_id=ep["id"])
            assert result["task_id"] == ep["task_id"]
            assert result["step"] == 1

    def test_episode_isolation(self):
        """Episodes should be isolated from each other."""
        # Create two episodes with same task
        result1 = reset(task_id=1, seed=100)
        result2 = reset(task_id=1, seed=200)
        
        id1 = result1["episode_id"]
        id2 = result2["episode_id"]
        records1 = result1["observation"]["records"]
        
        # Step episode 1 multiple times
        for _ in range(3):
            step(episode_id=id1, records=records1)
        
        # Episode 2 should still be at step 0
        state2 = state(episode_id=id2)
        assert state2["step"] == 0
        
        # Episode 1 should be at step 3
        state1 = state(episode_id=id1)
        assert state1["step"] == 3
