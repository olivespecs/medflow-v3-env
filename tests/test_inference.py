"""Regression tests for inference.py task-specific behavior."""

from __future__ import annotations

import json

import inference


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


def test_llm_process_task_task5_includes_contextual_guidance(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_create_chat_completion(*, client, model_name, messages):
        del client, model_name
        captured["messages"] = messages
        return _FakeCompletion(
            json.dumps(
                {
                    "records": [
                        {
                            "record_id": "rec-1",
                            "clinical_notes": "[REDACTED_NAME] was seen by Dr. Patel.",
                        }
                    ]
                }
            )
        )

    monkeypatch.setattr(inference, "_create_chat_completion", _fake_create_chat_completion)

    payload, _raw = inference._llm_process_task(
        client=object(),
        model_name="dummy-model",
        task_id=5,
        task_description="Task 5 - Contextual PII Disambiguation",
        records=[{"record_id": "rec-1", "clinical_notes": "Dr. Patel saw Mr. Patel."}],
    )

    prompt_text = str(captured["messages"][1]["content"]).lower()
    assert "contextual pii disambiguation" in prompt_text
    assert "do not redact provider" in prompt_text
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
    capture: dict[str, object] = {}

    def _fake_http_client_factory(*, timeout):
        del timeout
        return _FakeHTTPClient(capture)

    def _fake_llm_process_task(*, client, model_name, task_id, task_description, records):
        del client, model_name, task_description, records
        assert task_id == 4
        return (
            [
                {
                    "entities": [{"text": "I10", "type": "Condition", "code": "I10"}],
                    "summary": "LLM_SUMMARY_SHOULD_BE_SENT",
                }
            ],
            "{...}",
        )

    monkeypatch.setattr(inference.httpx, "Client", _fake_http_client_factory)
    monkeypatch.setattr(inference, "_llm_process_task", _fake_llm_process_task)

    result = inference._run_task_via_api(
        env_base_url="http://fake-env",
        task_id=4,
        seed=42,
        client=object(),
        model_name="dummy-model",
    )

    assert result["task_id"] == 4
    step_action = capture["step_action"]
    assert isinstance(step_action, dict)
    assert step_action["knowledge"][0]["summary"] == "LLM_SUMMARY_SHOULD_BE_SENT"