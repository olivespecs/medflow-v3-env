"""
Test SQLite episode persistence.

Verifies that episodes survive server restarts and are properly
restored from disk.
"""

import os
import tempfile
import time
from pathlib import Path

import pytest

from src.environment import MedicalOpenEnv
from src.persistence import SQLiteEpisodeStore, get_store, reset_store


@pytest.fixture
def temp_db():
    """Create temporary database file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "episodes.db"
        yield str(db_path)
        # Cleanup handled by TemporaryDirectory


@pytest.fixture
def store(temp_db):
    """Create fresh SQLite store."""
    reset_store()  # Clear any previous instance
    s = SQLiteEpisodeStore(temp_db)
    yield s
    s.close()
    reset_store()


class TestSQLitePersistence:
    """Test SQLite episode persistence layer."""
    
    def test_save_and_load_episode(self, store: SQLiteEpisodeStore):
        """Test basic save/load roundtrip."""
        # Create environment
        env = MedicalOpenEnv()
        env.reset(task_id=1, seed=42)
        
        # Simulate some steps
        env._step = 3
        env._done = False
        env._last_grade = {"score": 0.85, "passed": True}
        
        # Save to database
        episode_id = "test-episode-123"
        store.save_episode(episode_id, env, task_id=1, seed=42)
        
        # Load from database
        result = store.load_episode(episode_id)
        assert result is not None
        
        loaded_env, loaded_task_id, loaded_seed = result
        assert loaded_task_id == 1
        assert loaded_seed == 42
        assert loaded_env._step == 3
        assert loaded_env._done is False
        assert loaded_env._last_grade["score"] == 0.85

    def test_status_reports_errors_and_schema(self, temp_db):
        """Status should report schema_version and last_error after a failure."""
        # Create a bad store by pointing to a directory (will fail to open as file)
        bad_path = os.path.join(temp_db + "_dir")
        Path(bad_path).mkdir(parents=True, exist_ok=True)

        with pytest.raises(Exception):
            SQLiteEpisodeStore(bad_path)._get_connection()

        store = SQLiteEpisodeStore(temp_db)
        status = store.status()
        assert status["schema_version"] == SQLiteEpisodeStore.SCHEMA_VERSION
        # no error expected on healthy path
        assert status["last_error"] is None
    
    def test_delete_episode(self, store: SQLiteEpisodeStore):
        """Test episode deletion."""
        env = MedicalOpenEnv()
        env.reset(task_id=2, seed=123)
        
        episode_id = "to-delete"
        store.save_episode(episode_id, env, task_id=2, seed=123)
        
        # Verify exists
        assert store.load_episode(episode_id) is not None
        
        # Delete
        store.delete_episode(episode_id)
        
        # Verify gone
        assert store.load_episode(episode_id) is None
    
    def test_purge_expired_episodes(self, store: SQLiteEpisodeStore):
        """Test automatic expiration cleanup."""
        env = MedicalOpenEnv()
        env.reset(task_id=1, seed=42)
        
        # Save with old timestamp
        episode_id = "expired-episode"
        store.save_episode(episode_id, env, task_id=1, seed=42)
        
        # Manually update last_used to be old
        conn = store._get_connection()
        conn.execute(
            "UPDATE episodes SET last_used = ? WHERE episode_id = ?",
            (time.time() - 7200, episode_id),  # 2 hours ago
        )
        conn.commit()
        
        # Purge (TTL is 1 hour = 3600s)
        removed_count = store.purge_expired(ttl_seconds=3600)
        assert removed_count == 1
        
        # Verify gone
        assert store.load_episode(episode_id) is None
    
    def test_count_active_episodes(self, store: SQLiteEpisodeStore):
        """Test counting active episodes."""
        # Initially zero
        assert store.count_active() == 0
        
        # Add some episodes
        for i in range(3):
            env = MedicalOpenEnv()
            env.reset(task_id=1, seed=i)
            store.save_episode(f"episode-{i}", env, task_id=1, seed=i)
        
        assert store.count_active() == 3
        
        # Mark one as done
        conn = store._get_connection()
        conn.execute(
            "UPDATE episodes SET done = TRUE WHERE episode_id = ?",
            ("episode-1",),
        )
        conn.commit()
        
        # Should only count non-done
        assert store.count_active() == 2
    
    def test_persistence_across_restarts(self, temp_db):
        """Test that episodes survive creating new store instance."""
        # Create and save with first store
        store1 = SQLiteEpisodeStore(temp_db)
        env = MedicalOpenEnv()
        env.reset(task_id=3, seed=999)
        env._history = [{"step": 1, "score": 0.75}]
        store1.save_episode("persistent-ep", env, task_id=3, seed=999)
        store1.close()
        
        # Create second store (simulating restart)
        reset_store()
        store2 = SQLiteEpisodeStore(temp_db)
        
        # Load episode
        result = store2.load_episode("persistent-ep")
        assert result is not None
        
        loaded_env, task_id, seed = result
        assert task_id == 3
        assert seed == 999
        assert len(loaded_env._history) == 1
        assert loaded_env._history[0]["score"] == 0.75
        
        store2.close()
    
    def test_serialization_preserves_all_fields(self, store: SQLiteEpisodeStore):
        """Test that all environment fields are preserved."""
        from src.models import DirtyRecord, PatientRecord, Vitals
        
        env = MedicalOpenEnv()
        env.reset(task_id=1, seed=42)
        
        # Set various fields with proper types
        env._step = 5
        env._done = True
        env._dirty_records = [
            DirtyRecord(
                record_id="test", mrn="MRN123", patient_name="John Doe",
                dob="1990-01-01", gender="M", injected_flaws=["date_format"]
            )
        ]
        env._clean_truth = [
            PatientRecord(
                record_id="truth", mrn="MRN456", patient_name="Jane Doe",
                dob="1985-05-15", gender="F"
            )
        ]
        env._baseline_ml_scores = [0.8, 0.9]
        env._last_grade = {
            "score": 0.92,
            "breakdown": {"field1": 0.9, "field2": 0.95},
            "passed": True,
        }
        env._last_submitted = [{"submitted": "data"}]
        env._history = [
            {"step": 1, "score": 0.7},
            {"step": 2, "score": 0.8},
        ]
        
        # Save and load
        store.save_episode("complete-test", env, task_id=1, seed=42)
        result = store.load_episode("complete-test")
        
        assert result is not None
        loaded_env, _, _ = result
        
        # Verify all fields
        assert loaded_env._step == 5
        assert loaded_env._done is True
        assert len(loaded_env._dirty_records) == 1
        assert len(loaded_env._clean_truth) == 1
        assert len(loaded_env._baseline_ml_scores) == 2
        assert loaded_env._last_grade["score"] == 0.92
        assert len(loaded_env._history) == 2


class TestGlobalStore:
    """Test global store singleton."""
    
    def test_get_store_singleton(self, temp_db):
        """Test that get_store returns same instance."""
        reset_store()
        
        store1 = get_store(temp_db)
        store2 = get_store(temp_db)
        
        assert store1 is store2
        
        reset_store()
    
    def test_reset_store(self, temp_db):
        """Test store reset."""
        reset_store()
        
        store = get_store(temp_db)
        assert store is not None
        
        reset_store()
        
        # Old reference should be invalid
        new_store = get_store(temp_db)
        # Should be different instance after reset


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
