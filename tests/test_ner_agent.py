import re

from src.ner_agent import LocalNERAgent
from src.record_processors import _redact_contextual_phi


class FakePipeline:
    def __init__(self, returns):
        self.returns = returns
        self.calls = 0

    def __call__(self, text):
        self.calls += 1
        return self.returns


def test_redact_text_triggers_on_uppercase_and_merges_spans():
    agent = LocalNERAgent(safe_mode=True)
    agent.nlp = FakePipeline(
        [
            {"start": 0, "end": 4, "entity_group": "PER", "word": "JOHN", "score": 0.90},
            {"start": 0, "end": 8, "entity_group": "PER", "word": "JOHN DOE", "score": 0.85},
        ]
    )

    text = "JOHN DOE visited the ER."
    redacted = agent.redact_text(text)

    assert agent.nlp.calls == 1
    assert redacted.startswith("[REDACTED_NAME]")
    assert "visited" in redacted
    assert "[REDACTED_NAME][REDACTED_NAME]" not in redacted


def test_provider_allowlist_preserves_facility_names():
    agent = LocalNERAgent(safe_mode=True)
    agent.nlp = FakePipeline(
        [
            {
                "start": 8,
                "end": 24,
                "entity_group": "ORG",
                "word": "General Hospital",
                "score": 0.92,
            }
        ]
    )

    text = "Seen at General Hospital for follow-up."
    redacted = agent.redact_text(text)

    assert "General Hospital" in redacted
    # Ensure no generic redaction was applied to the facility mention
    assert "[REDACTED_ADDRESS]" not in redacted


def test_task5_contextual_redaction_preserves_providers():
    record = {
        "clinical_notes": (
            "Dr. Lee consulted on the case. The patient's daughter, Mary Lee, provided consent. "
            "Lee General Hospital has advanced facilities."
        )
    }

    processed = _redact_contextual_phi(record)
    notes = processed.get("clinical_notes", "")

    assert "Dr. Lee" in notes  # provider preserved
    assert "[REDACTED_PATIENT]" in notes  # patient/family redacted
    assert "Lee General Hospital" in notes  # facility preserved
