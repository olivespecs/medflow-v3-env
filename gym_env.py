"""
Gymnasium-compatible wrapper for Medical Records OpenEnv.

Enables plug-and-play compatibility with:
  - RLlib        (pip install ray[rllib])
  - Stable Baselines 3 (pip install stable-baselines3)
  - CleanRL      (pip install cleanrl)
  - TRL / VERL   (LLM-based RL frameworks)
  - Any other gymnasium-compatible RL framework

Extra dependency (not in requirements.txt — install when using this module):
  pip install gymnasium httpx

Two wrapper classes are provided:

  MedicalRecordsGymEnv      — wraps the HTTP API of a running server.
                              Works with any deployment (local or HF Spaces).

  LocalMedicalRecordsGymEnv — wraps the Python classes directly.
                              No server needed; ideal for local training loops.

Observation space:  gymnasium.spaces.Text  (JSON-serialised observation dict)
Action space:       gymnasium.spaces.Text  (JSON-serialised action dict)

For numeric RL (PPO/DQN on MLPs), subclass either wrapper and override
_encode_obs() to return a fixed-size numpy array extracted from the JSON.

Quick-start — HTTP mode (server already running on localhost:7860):
  env = MedicalRecordsGymEnv(task_id=2, base_url="http://localhost:7860")
  obs, info = env.reset()
  records = json.loads(obs)["records"]
  # ... agent processes records ...
  action = json.dumps({"records": processed_records})
  obs, reward, terminated, truncated, info = env.step(action)
  env.close()

Quick-start — Local mode (no server needed):
  env = LocalMedicalRecordsGymEnv(task_id=1, seed=42)
  obs, info = env.reset()
  # ... same interface as above ...
"""

from __future__ import annotations

import json
import string
from typing import Any

try:
    import gymnasium
    from gymnasium import spaces
except ImportError as _gym_err:
    raise ImportError(
        "gymnasium is required for gym_env.py. "
        "Install it with:  pip install gymnasium"
    ) from _gym_err


# ---------------------------------------------------------------------------
# Shared observation encoding
# ---------------------------------------------------------------------------

def _obs_to_text(obs: dict[str, Any]) -> str:
    """Serialise an observation dict to a JSON string (the Text-space value)."""
    return json.dumps(obs, default=str)


def _parse_action(action: str | dict[str, Any]) -> dict[str, Any]:
    """Accept a JSON string or a raw dict as an action."""
    if isinstance(action, str):
        return json.loads(action)
    if isinstance(action, dict):
        return action
    raise TypeError(f"Action must be a JSON string or dict, got {type(action).__name__}")


# ---------------------------------------------------------------------------
# Shared observation / action spaces
# ---------------------------------------------------------------------------

_TEXT_SPACE = spaces.Text(
    min_length=2,
    max_length=1_000_000,
    charset=string.printable,
)


# ===========================================================================
# HTTP API wrapper
# ===========================================================================

class MedicalRecordsGymEnv(gymnasium.Env):
    """
    Gymnasium wrapper around the Medical Records OpenEnv **HTTP API**.

    Works with any running instance — local dev server or Hugging Face Space.
    Session lifetime is managed automatically (reset → episode_id → step).

    Parameters
    ----------
    task_id  : int   1=Hygiene, 2=Redaction, 3=Anonymisation, 4=Knowledge
    seed     : int   RNG seed passed to /reset
    base_url : str   Base URL of the running server
    timeout  : float HTTP timeout in seconds
    """

    metadata = {"render_modes": ["human"], "name": "MedicalRecordsOpenEnv-v1"}

    def __init__(
        self,
        task_id: int = 1,
        seed: int = 42,
        base_url: str = "http://localhost:7860",
        timeout: float = 60.0,
    ) -> None:
        super().__init__()

        try:
            import httpx
        except ImportError as e:
            raise ImportError(
                "httpx is required for MedicalRecordsGymEnv. "
                "Install it with:  pip install httpx"
            ) from e

        self.task_id = task_id
        self._seed = seed
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(timeout=timeout)
        self._episode_id: str | None = None

        self.observation_space = _TEXT_SPACE
        self.action_space = _TEXT_SPACE

    # ------------------------------------------------------------------
    # gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Start a new episode. Returns (obs_text, info)."""
        if seed is not None:
            self._seed = seed

        resp = self._http.post(
            f"{self.base_url}/reset",
            json={"task_id": self.task_id, "seed": self._seed},
        )
        resp.raise_for_status()
        data = resp.json()

        self._episode_id = data["episode_id"]
        obs = data["observation"]

        return self._encode_obs(obs), {
            "episode_id": self._episode_id,
            "task_description": obs.get("task_description", ""),
        }

    def step(
        self, action: str | dict[str, Any]
    ) -> tuple[str, float, bool, bool, dict[str, Any]]:
        """
        Submit processed records/knowledge and advance the episode.

        Returns (obs, reward, terminated, truncated, info).
        """
        if self._episode_id is None:
            raise RuntimeError("Call reset() before step().")

        action_dict = _parse_action(action)
        action_dict.setdefault("is_final", True)

        resp = self._http.post(
            f"{self.base_url}/step",
            params={"episode_id": self._episode_id},
            json=action_dict,
        )
        resp.raise_for_status()
        data = resp.json()

        obs = data.get("observation", {})
        reward = float(data.get("reward", 0.0))
        done = bool(data.get("done", False))
        info = data.get("info", {})
        info["episode_id"] = self._episode_id
        info["breakdown"] = data.get("info", {}).get("breakdown", {})

        return self._encode_obs(obs), reward, done, False, info

    def render(self) -> None:
        """No visual rendering — see the Gradio UI at /ui instead."""

    def close(self) -> None:
        self._http.close()

    # ------------------------------------------------------------------
    # Override point for numeric RL
    # ------------------------------------------------------------------

    def _encode_obs(self, obs: dict[str, Any]) -> str:
        """
        Encode an observation dict into a Text-space value (JSON string).

        For numeric RL (DQN / PPO on MLPs), subclass and override this method
        to extract a fixed-size numpy array from the observation dict, and
        change self.observation_space to a matching Box space.
        """
        return _obs_to_text(obs)


# ===========================================================================
# Local Python wrapper (no server required)
# ===========================================================================

class LocalMedicalRecordsGymEnv(gymnasium.Env):
    """
    Gymnasium wrapper around the **Python classes** directly.

    No server needed — imports MedicalOpenEnv and Action from src/.
    Ideal for local training loops where HTTP overhead is undesirable.

    Parameters
    ----------
    task_id : int  1=Hygiene, 2=Redaction, 3=Anonymisation, 4=Knowledge
    seed    : int  RNG seed for episode generation
    """

    metadata = {"render_modes": ["human"], "name": "MedicalRecordsOpenEnv-v1-local"}

    def __init__(self, task_id: int = 1, seed: int = 42) -> None:
        super().__init__()

        try:
            from src.environment import MedicalOpenEnv
            from src.models import Action
        except ImportError as e:
            raise ImportError(
                "Could not import src.environment. "
                "Run from the project root directory or install the package with  pip install -e ."
            ) from e

        self._MedicalOpenEnv = MedicalOpenEnv
        self._Action = Action
        self.task_id = task_id
        self._seed = seed
        self._env: Any = None

        self.observation_space = _TEXT_SPACE
        self.action_space = _TEXT_SPACE

    # ------------------------------------------------------------------
    # gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Create a fresh environment and return the first observation."""
        if seed is not None:
            self._seed = seed

        self._env = self._MedicalOpenEnv()
        obs = self._env.reset(task_id=self.task_id, seed=self._seed)

        obs_dict = obs.model_dump()
        return self._encode_obs(obs_dict), {
            "task_description": obs_dict.get("task_description", ""),
        }

    def step(
        self, action: str | dict[str, Any]
    ) -> tuple[str, float, bool, bool, dict[str, Any]]:
        """Submit processed records/knowledge and advance the episode."""
        if self._env is None:
            raise RuntimeError("Call reset() before step().")

        action_dict = _parse_action(action)
        action_dict.setdefault("is_final", True)

        act = self._Action(**action_dict)
        next_obs, reward, done, info = self._env.step(act)

        obs_dict = next_obs.model_dump()
        info["breakdown"] = reward.breakdown

        return self._encode_obs(obs_dict), float(reward.score), done, False, info

    def render(self) -> None:
        """No visual rendering."""

    def close(self) -> None:
        self._env = None

    # ------------------------------------------------------------------
    # Override point for numeric RL
    # ------------------------------------------------------------------

    def _encode_obs(self, obs: dict[str, Any]) -> str:
        """
        JSON-encode the observation dict.

        Override to return a numpy array for numeric RL frameworks.
        """
        return _obs_to_text(obs)


# ===========================================================================
# Gymnasium registration (optional — enables gym.make("MedicalRecords-v1"))
# ===========================================================================

def _register() -> None:
    """Register both wrappers with gymnasium so gym.make() works."""
    try:
        gymnasium.register(
            id="MedicalRecords-v1",
            entry_point="gym_env:MedicalRecordsGymEnv",
            kwargs={"task_id": 1, "base_url": "http://localhost:7860"},
        )
        gymnasium.register(
            id="MedicalRecordsLocal-v1",
            entry_point="gym_env:LocalMedicalRecordsGymEnv",
            kwargs={"task_id": 1},
        )
    except Exception:
        pass  # Already registered or gymnasium not available


_register()


# ===========================================================================
# Smoke test — run directly to verify the wrapper works
#   python gym_env.py              (local mode — no server)
#   python gym_env.py --http       (HTTP mode — requires running server)
# ===========================================================================

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Gym wrapper smoke test")
    parser.add_argument("--http", action="store_true", help="Use HTTP wrapper (requires server)")
    parser.add_argument("--base-url", default="http://localhost:7860")
    parser.add_argument("--task", type=int, default=1, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  Medical Records OpenEnv — gymnasium wrapper smoke test")
    print(f"  mode={'HTTP' if args.http else 'local'}  task={args.task}  seed={args.seed}")
    print(f"{'='*55}\n")

    if args.http:
        env = MedicalRecordsGymEnv(
            task_id=args.task, seed=args.seed, base_url=args.base_url
        )
    else:
        env = LocalMedicalRecordsGymEnv(task_id=args.task, seed=args.seed)

    # Verify gymnasium API compliance
    from gymnasium.utils.env_checker import check_env
    try:
        check_env(env, warn=True, skip_render_check=True)
        print("✅ check_env passed — wrapper is gymnasium-compliant\n")
    except Exception as e:
        print(f"⚠️  check_env warning (expected for Text spaces): {e}\n")

    obs, info = env.reset(seed=args.seed)
    obs_dict = json.loads(obs)
    n_records = len(obs_dict.get("records", []))
    print(f"reset()  → {n_records} records received")
    print(f"         task: {obs_dict.get('task_description', '')[:80]}...")

    # Dummy action: return records unchanged (will score low but tests the loop)
    records = obs_dict.get("records", [])
    if args.task == 4:
        dummy_action = json.dumps({
            "knowledge": [{"entities": [], "summary": "No summary."} for _ in records]
        })
    else:
        dummy_action = json.dumps({"records": records})

    obs2, reward, terminated, truncated, info = env.step(dummy_action)
    print(f"step()   → reward={reward:.4f}  terminated={terminated}")
    print(f"         passed={info.get('passed', False)}")
    bd = info.get("breakdown", {})
    if bd:
        first_k = list(bd.items())[:3]
        print(f"         breakdown (first 3): { {k: round(v, 4) if isinstance(v, float) else v for k, v in first_k} }")

    env.close()
    print(f"\n✅ Smoke test complete\n")
