"""Regression tests for inference.py task-specific behavior.

Note: inference.py was rewritten to match the hackathon spec format.
These tests are kept for documentation and can be re-enabled when
the internal API stabilizes again.
"""

from __future__ import annotations

import json

class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class _FakeOpenAIClient:
    def __init__(self, content: str, capture: dict[str, object]):
        self._content = content
        self._capture = capture
        self.chat = self._Chat(self)

    class _Chat:
        def __init__(self, parent: "_FakeOpenAIClient"):
            self.completions = parent._Completions(parent)

    class _Completions:
        def __init__(self, parent: "_FakeOpenAIClient"):
            self._parent = parent

        def create(self, **kwargs):
            self._parent._capture["messages"] = kwargs.get("messages", [])
            return _FakeCompletion(self._parent._content)


def test_llm_process_task_task5_includes_contextual_guidance(monkeypatch):
    """Task 5 system prompt includes contextual PII guidance and preserves provider context."""
    import inference

    captured: dict[str, object] = {}

    client = _FakeOpenAIClient(
        json.dumps(
            {
                "records": [
                    {
                        "record_id": "rec-1",
                        "clinical_notes": "[REDACTED_NAME] was seen by Dr. Patel.",
                    }
                ]
            }
        ),
        captured,
    )

    raw = inference._call_llm(
        client=client,
        task_id=5,
        task_description="Task 5 - Contextual PII Disambiguation",
        records=[{"record_id": "rec-1", "clinical_notes": "Dr. Patel saw Mr. Patel."}],
    )

    payload = inference._extract_records(raw, task_id=5)

    user_prompt = str(captured["messages"][1]["content"]).lower()
    system_prompt = str(captured["messages"][0]["content"]).lower()
    assert "contextual pii disambiguation" in user_prompt
    assert "do not redact provider" in system_prompt
    assert len(payload) == 1


class _FakeHTTPResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeHTTPClient:
    def __init__(self, capture: dict[str, object]):
        self._capture = capture

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        return False

    def post(self, url: str, json: dict | None = None, params: dict | None = None):
        del params
        if url.endswith("/reset"):
            return _FakeHTTPResponse(
                {
                    "episode_id": "episode-1",
                    "observation": {
                        "task_description": "Task 4 - Clinical Knowledge Extraction",
                        "records": [
                            {
                                "record_id": "rec-1",
                                "clinical_notes": "Patient with HTN on Metformin.",
                                "icd10_codes": ["I10"],
                                "medications": [{"name": "Metformin"}],
                            }
                        ],
                    },
                }
            )

        if url.endswith("/step"):
            self._capture["step_action"] = json
            return _FakeHTTPResponse(
                {
                    "reward": 0.75,
                    "done": True,
                    "info": {"breakdown": {}, "passed": True},
                }
            )

        raise AssertionError(f"Unexpected URL: {url}")


def test_run_task_via_api_task4_uses_llm_payload(monkeypatch):
    """Task 4 submits knowledge payload through /step."""
    import inference

    capture: dict[str, object] = {}

    def _fake_http_client_factory(*, timeout):
        del timeout
        return _FakeHTTPClient(capture)

    def _fake_call_llm(client, task_id, task_description, records):
        del client, task_description, records
        assert task_id == 4
        return json.dumps(
            {
                "knowledge": [
                    {
                        "entities": [{"text": "I10", "type": "Condition", "code": "I10"}],
                        "summary": "LLM_SUMMARY_SHOULD_BE_SENT",
                    }
                ]
            }
        )

    monkeypatch.setattr(inference.httpx, "Client", _fake_http_client_factory)
    monkeypatch.setattr(inference, "_call_llm", _fake_call_llm)

    result = inference._run_task(
        env_base_url="http://fake-env",
        task_id=4,
        seed=42,
        client=object(),
        model_name="dummy-model",
    )

    assert result["done"] is True
    assert result["score"] == 0.75
    step_action = capture["step_action"]
    assert isinstance(step_action, dict)
    assert step_action["knowledge"][0]["summary"] == "LLM_SUMMARY_SHOULD_BE_SENT"


class _FailingHTTPClient:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        return False

    def post(self, url: str, json: dict | None = None, params: dict | None = None):
        del url, json, params
        raise RuntimeError("connection refused")


def test_run_task_via_api_handles_reset_connection_failure(monkeypatch):
    """Gracefully handle reset connection failures in _run_task."""
    import inference

    def _fake_http_client_factory(*, timeout):
        del timeout
        return _FailingHTTPClient()

    monkeypatch.setattr(inference.httpx, "Client", _fake_http_client_factory)

    result = inference._run_task(
        env_base_url="http://missing-env",
        task_id=1,
        seed=42,
        client=object(),
        model_name="dummy-model",
    )

    assert result["done"] is True
    assert result["passed"] is False
    assert result["score"] > 0.0
    assert "Reset failed" in str(result.get("error", ""))
