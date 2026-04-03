import importlib

from src.environment import MedicalOpenEnv


def test_local_ner_agent_safe_mode(monkeypatch):
    # Safe mode should disable transformer loading and rely on rules only
    monkeypatch.setenv("BASELINE_SAFE_MODE", "1")
    from src.ner_agent import LocalNERAgent

    agent = LocalNERAgent()

    assert agent.nlp is None
    assert agent.safe_mode is True
    assert agent.disabled_reason and "safe mode" in agent.disabled_reason


def test_hybrid_baseline_emits_debug_trace(monkeypatch):
    # Hybrid baseline should surface debug info and honor safe mode
    monkeypatch.setenv("BASELINE_SAFE_MODE", "1")

    import src.baseline_agent as baseline_agent

    importlib.reload(baseline_agent)
    baseline_agent._HYBRID_NER_AGENT = None

    env = MedicalOpenEnv()
    env.reset(task_id=1, seed=1)

    result = baseline_agent.hybrid_baseline(env)
    debug = result["debug"]

    assert debug["ner"]["enabled"] is False
    assert debug["safe_mode"] is True
    assert isinstance(debug["reward_trace"], list)
    assert debug["reward_trace"]
