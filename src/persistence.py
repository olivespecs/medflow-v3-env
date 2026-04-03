"""
SQLite-based episode persistence layer.

Provides durable storage for episodes across server restarts,
essential for Hugging Face Spaces and production deployments.
"""

from __future__ import annotations

import logging
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os

from .environment import MedicalOpenEnv
from .models import AnnotatedRecord, DirtyRecord, PatientRecord

logger = logging.getLogger(__name__)


@dataclass
class PersistedEpisode:
    """Serialized episode data for database storage."""
    episode_id: str
    task_id: int
    seed: int
    step: int
    done: bool
    env_state: bytes  # UTF-8 encoded JSON environment state
    last_used: float
    created_at: float


class SQLiteEpisodeStore:
    """
    Thread-safe SQLite store for episode persistence.
    
    Features:
    - Automatic schema migrations
    - WAL mode for concurrent reads
    - Periodic cleanup of expired episodes
    - Graceful degradation to in-memory if DB unavailable
    """
    
    SCHEMA_VERSION = 1
    ENV_STATE_VERSION = 1

    CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS episodes (
        episode_id TEXT PRIMARY KEY,
        task_id INTEGER NOT NULL,
        seed INTEGER NOT NULL,
        step INTEGER DEFAULT 0,
        done BOOLEAN DEFAULT FALSE,
        env_state BLOB NOT NULL,
        last_used REAL NOT NULL,
        created_at REAL NOT NULL
    );
    
    CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    
    CREATE INDEX IF NOT EXISTS idx_episodes_last_used ON episodes(last_used);
    CREATE INDEX IF NOT EXISTS idx_episodes_done ON episodes(done);
    """
    
    def __init__(self, db_path: str | None = None):
        """
        Initialize SQLite store.
        
        Args:
            db_path: Path to SQLite database. If None, uses in-memory fallback.
        """
        self.db_path = db_path or ":memory:"
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        self._initialized = False
        self._last_error: str | None = None
        
    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection (thread-local)."""
        if self._conn is None:
            try:
                self._conn = sqlite3.connect(
                    self.db_path,
                    check_same_thread=False,  # We manage locking ourselves
                    timeout=30.0
                )
                self._conn.row_factory = sqlite3.Row
                
                # Enable WAL mode for better concurrency + durability
                self._conn.execute("PRAGMA journal_mode=WAL")
                # Prefer FULL to ensure fsync; can be tuned via env if needed
                self._conn.execute("PRAGMA synchronous=FULL")
                self._conn.execute("PRAGMA busy_timeout=5000")
                
                # Initialize schema
                self._initialize_schema()
            except Exception as e:
                self._record_error(f"connection failed: {e}")
                raise
            
        return self._conn
    
    def _initialize_schema(self) -> None:
        """Create tables and indexes."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.executescript(self.CREATE_TABLE_SQL)
                conn.commit()
                self._set_schema_version(conn)
                self._initialized = True
                logger.info(f"SQLite episode store initialized: {self.db_path}")
            except Exception as e:
                self._record_error(f"schema init failed: {e}")
                logger.error(f"Failed to initialize SQLite schema: {e}")
                raise

    def _set_schema_version(self, conn: sqlite3.Connection) -> None:
        """Ensure schema version is set and compatible."""
        existing = self._get_meta(conn, "schema_version")
        if existing is None:
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                ("schema_version", str(self.SCHEMA_VERSION)),
            )
            conn.commit()
            return
        try:
            current = int(existing)
        except ValueError:
            raise RuntimeError(f"Invalid schema_version in DB: {existing}")
        if current != self.SCHEMA_VERSION:
            raise RuntimeError(
                f"Schema version mismatch (db={current}, code={self.SCHEMA_VERSION}). "
                "Manual migration required."
            )

    def _get_meta(self, conn: sqlite3.Connection, key: str) -> str | None:
        cursor = conn.execute("SELECT value FROM meta WHERE key = ?", (key,))
        row = cursor.fetchone()
        return row[0] if row else None

    def _record_error(self, message: str) -> None:
        self._last_error = message
    
    def save_episode(
        self,
        episode_id: str,
        env: MedicalOpenEnv,
        task_id: int,
        seed: int,
    ) -> None:
        """
        Save episode to database.
        
        Args:
            episode_id: Unique episode identifier
            env: MedicalOpenEnv instance to persist
            task_id: Current task ID
            seed: Random seed
        """
        conn = self._get_connection()
        
        # Serialize environment state
        env_state = self._serialize_env(env)
        now = time.time()
        
        with self._lock:
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO episodes 
                    (episode_id, task_id, seed, step, done, env_state, last_used, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM episodes WHERE episode_id = ?), ?))
                    """,
                    (
                        episode_id,
                        task_id,
                        seed,
                        env._step,
                        env._done,
                        env_state,
                        now,
                        episode_id,
                        now,
                    ),
                )
                self._safe_commit(conn)
            except sqlite3.Error as e:
                self._record_error(f"save failed: {e}")
                logger.warning(f"SQLite save failed for {episode_id}: {e}")
                raise
    
    def load_episode(self, episode_id: str) -> tuple[MedicalOpenEnv, int, int] | None:
        """
        Load episode from database.
        
        Returns:
            Tuple of (env, task_id, seed) or None if not found
        """
        conn = self._get_connection()
        
        with self._lock:
            try:
                cursor = conn.execute(
                    "SELECT * FROM episodes WHERE episode_id = ?",
                    (episode_id,),
                )
                row = cursor.fetchone()
                
                if row is None:
                    return None
                
                # Update last_used timestamp
                conn.execute(
                    "UPDATE episodes SET last_used = ? WHERE episode_id = ?",
                    (time.time(), episode_id),
                )
                self._safe_commit(conn)
                
                # Deserialize environment
                env = self._deserialize_env(row["env_state"])
                
                return env, row["task_id"], row["seed"]
            except sqlite3.Error as e:
                self._record_error(f"load failed: {e}")
                logger.warning(f"SQLite load failed for {episode_id}: {e}")
                raise
    
    def delete_episode(self, episode_id: str) -> None:
        """Remove episode from database."""
        conn = self._get_connection()
        
        with self._lock:
            try:
                conn.execute(
                    "DELETE FROM episodes WHERE episode_id = ?",
                    (episode_id,),
                )
                self._safe_commit(conn)
            except sqlite3.Error as e:
                self._record_error(f"delete failed: {e}")
                logger.warning(f"SQLite delete failed for {episode_id}: {e}")
                raise
    
    def purge_expired(self, ttl_seconds: int = 3600) -> int:
        """
        Remove expired episodes.
        
        Args:
            ttl_seconds: Time-to-live in seconds
            
        Returns:
            Number of episodes removed
        """
        conn = self._get_connection()
        cutoff = time.time() - ttl_seconds
        
        with self._lock:
            try:
                cursor = conn.execute(
                    "SELECT episode_id FROM episodes WHERE last_used < ?",
                    (cutoff,),
                )
                expired_ids = [row["episode_id"] for row in cursor.fetchall()]
                
                if expired_ids:
                    placeholders = ",".join("?" * len(expired_ids))
                    conn.execute(
                        f"DELETE FROM episodes WHERE episode_id IN ({placeholders})",
                        expired_ids,
                    )
                    self._safe_commit(conn)
                    logger.info(f"Purged {len(expired_ids)} expired episodes")
                
                return len(expired_ids)
            except sqlite3.Error as e:
                self._record_error(f"purge failed: {e}")
                logger.warning(f"SQLite purge failed: {e}")
                raise
    
    def count_active(self) -> int:
        """Get count of active (non-expired) episodes."""
        conn = self._get_connection()
        
        with self._lock:
            try:
                cursor = conn.execute("SELECT COUNT(*) FROM episodes WHERE done = FALSE")
                return cursor.fetchone()[0]
            except sqlite3.Error as e:
                self._record_error(f"count failed: {e}")
                logger.warning(f"SQLite count failed: {e}")
                raise
    
    def _serialize_env(self, env: MedicalOpenEnv) -> bytes:
        """Serialize environment to UTF-8 JSON bytes with explicit schema."""

        state = {
            "version": self.ENV_STATE_VERSION,
            "task_id": int(env._task_id),
            "seed": int(env._seed),
            "step": int(env._step),
            "done": bool(env._done),
            "generator_id": env._generator_id,
            "correlation_id": env._correlation_id,
            "dirty_records": [r.model_dump() for r in env._dirty_records],
            "clean_truth": [r.model_dump() for r in env._clean_truth],
            "annotated_records": [r.model_dump() for r in env._annotated_records],
            "baseline_ml_scores": [float(v) for v in env._baseline_ml_scores],
            "last_grade": env._last_grade,
            "last_submitted": env._last_submitted,
            "history": env._history,
        }
        return json.dumps(state, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    
    def _deserialize_env(self, data: bytes) -> MedicalOpenEnv:
        """Deserialize environment from UTF-8 JSON bytes with strict validation."""
        try:
            raw = json.loads(data.decode("utf-8"))
        except Exception as e:
            raise ValueError(f"Invalid serialized environment payload: {type(e).__name__}") from e

        if not isinstance(raw, dict):
            raise ValueError("Serialized environment payload must be a JSON object")
        if int(raw.get("version", -1)) != self.ENV_STATE_VERSION:
            raise ValueError(
                f"Unsupported serialized environment version: {raw.get('version')!r}"
            )

        env = MedicalOpenEnv()
        env._task_id = int(raw.get("task_id", 1))
        env._seed = int(raw.get("seed", 42))
        env._step = int(raw.get("step", 0))
        env._done = bool(raw.get("done", False))
        env._generator_id = raw.get("generator_id")
        env._correlation_id = raw.get("correlation_id")
        env._dirty_records = [
            DirtyRecord.model_validate(r) for r in raw.get("dirty_records", [])
        ]
        env._clean_truth = [
            PatientRecord.model_validate(r) for r in raw.get("clean_truth", [])
        ]
        env._annotated_records = [
            AnnotatedRecord.model_validate(r) for r in raw.get("annotated_records", [])
        ]
        env._baseline_ml_scores = [
            float(v) for v in raw.get("baseline_ml_scores", [])
        ]
        env._last_grade = raw.get("last_grade", {})
        env._last_submitted = raw.get("last_submitted", [])
        env._history = raw.get("history", [])
        
        return env
    
    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def status(self) -> dict[str, Any]:
        """Return current persistence status for health checks."""
        return {
            "enabled": self.db_path not in (":memory:", None),
            "db_path": self.db_path,
            "initialized": self._initialized,
            "last_error": self._last_error,
            "schema_version": self.SCHEMA_VERSION,
        }

    def _safe_commit(self, conn: sqlite3.Connection) -> None:
        """Commit and fsync WAL/main files to reduce data loss risk."""
        conn.commit()

        if self.db_path == ":memory:":
            return

        try:
            db_main = Path(self.db_path)
            wal_path = db_main.with_suffix(db_main.suffix + "-wal")
            for path in (db_main, wal_path):
                if path.exists():
                    fd = os.open(path, os.O_RDONLY)
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)
        except Exception as e:
            # Do not fail the request; just log and record the error
            self._record_error(f"fsync failed: {e}")
            logger.warning(f"SQLite fsync failed: {e}")


# Global instance (lazy initialization)
_store: SQLiteEpisodeStore | None = None


def get_store(db_path: str | None = None) -> SQLiteEpisodeStore:
    """Get or create global SQLite store instance."""
    global _store
    if _store is None:
        _store = SQLiteEpisodeStore(db_path)
    return _store


def reset_store() -> None:
    """Reset global store instance (for testing)."""
    global _store
    if _store:
        _store.close()
    _store = None
