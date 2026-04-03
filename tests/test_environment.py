"""Tests for MedicalOpenEnv core interface."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment import MedicalOpenEnv
from src.models import Action, Observation, Reward


def test_reset_returns_valid_observation():
    env = MedicalOpenEnv()
    obs = env.reset(task_id=1, seed=42)
    assert isinstance(obs, Observation)
    assert obs.task_id == 1
    assert len(obs.records) > 0
    assert obs.step == 0
    assert obs.max_steps == 10


def test_reset_all_tasks():
    env = MedicalOpenEnv()
    for task_id in (1, 2, 3):
        obs = env.reset(task_id=task_id, seed=0)
        assert obs.task_id == task_id
        assert len(obs.records) >= 4


def test_task4_observation_not_clean_truth_leakage():
    env = MedicalOpenEnv()
    obs = env.reset(task_id=4, seed=42)

    assert len(obs.records) == len(env._clean_truth)
    assert all("injected_flaws" not in record for record in obs.records)
    assert all(
        submitted != truth.model_dump()
        for submitted, truth in zip(obs.records, env._clean_truth)
    )


def test_step_returns_valid_reward():
    env = MedicalOpenEnv()
    obs = env.reset(task_id=1, seed=42)
    action = Action(records=obs.records, is_final=False)
    _, reward, done, info = env.step(action)
    assert isinstance(reward, Reward)
    assert 0.0 <= reward.score <= 1.0
    assert isinstance(done, bool)


def test_episode_ends_on_is_final():
    env = MedicalOpenEnv()
    obs = env.reset(task_id=1, seed=42)
    action = Action(records=obs.records, is_final=True)
    _, _, done, _ = env.step(action)
    assert done is True


def test_episode_ends_after_max_steps():
    env = MedicalOpenEnv()
    obs = env.reset(task_id=2, seed=1)
    done = False
    for _ in range(env._step, 10):
        action = Action(records=obs.records, is_final=False)
        obs, reward, done, info = env.step(action)
        if done:
            break
    assert done is True


def test_state_is_json_serialisable():
    env = MedicalOpenEnv()
    env.reset(task_id=3, seed=7)
    action = Action(records=[], is_final=False)
    try:
        env.step(action)
    except Exception:
        pass
    state = env.state()
    # Should not raise - convert Pydantic model to dict for JSON serialization
    json.dumps(state.model_dump() if hasattr(state, 'model_dump') else state)


def test_reset_clears_previous_state():
    env = MedicalOpenEnv()
    env.reset(task_id=1, seed=10)
    # take a step
    obs = env.reset(task_id=1, seed=10)
    action = Action(records=obs.records, is_final=False)
    env.step(action)
    # re-reset
    obs2 = env.reset(task_id=2, seed=99)
    assert obs2.task_id == 2
    assert env._step == 0
    assert env._done is False


def test_step_after_done_raises():
    env = MedicalOpenEnv()
    obs = env.reset(task_id=1, seed=42)
    action = Action(records=obs.records, is_final=True)
    env.step(action)
    import pytest
    with pytest.raises(RuntimeError):
        env.step(action)


def test_regrade_is_idempotent():
    env = MedicalOpenEnv()
    obs = env.reset(task_id=1, seed=42)
    action = Action(records=obs.records, is_final=False)
    _, reward1, _, _ = env.step(action)
    reward2 = env.regrade()
    assert reward1.score == reward2["score"]


def test_full_episode_loop():
    """End-to-end test of a full episode across all tasks."""
    env = MedicalOpenEnv()
    for task_id in (1, 2, 3, 4):
        obs = env.reset(task_id=task_id, seed=42)
        assert obs.step == 0
        
        # Test multi-step logic
        if task_id == 4:
            action = Action(knowledge=[], is_final=False)
        else:
            action = Action(records=obs.records, is_final=False)
            
        next_obs, reward, done, info = env.step(action)
        assert next_obs.step == 1
        assert 0.0 <= reward.score <= 1.0
        assert not done
        
        # Test final step
        if task_id == 4:
            action = Action(knowledge=[], is_final=True)
        else:
            action = Action(records=obs.records, is_final=True)
            
        _, _, done, _ = env.step(action)
        assert done is True
