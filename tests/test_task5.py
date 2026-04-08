"""Tests for Task 5 — Contextual PII Disambiguation grader."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tasks import task5_reasoning
from src.models import AnnotatedRecord, PHICategory


def _make_truth_records():
    """
    Create ground truth AnnotatedRecord objects for Task 5.
    """
    return task5_reasoning.generate_ambiguous_records()


def test_perfect_disambiguation_passes():
    """An agent that correctly redacts patient info and keeps provider info should pass."""
    truths = _make_truth_records()
    submitted = []
    
    for truth in truths:
        notes = truth.clinical_notes
        # Correctly redact patient identifiers
        for token in truth.phi_tokens:
            if token.category == PHICategory.PATIENT_IDENTIFIER:
                notes = notes.replace(token.value, "[REDACTED]")
            # Do NOT redact provider identifiers
            
        # Handle ambiguous cases according to the grader's logic
        # In the grader, "Dr."/"Nurse" etc are providers (keep), "Mr."/"Mrs." etc are patients (redact)
        for token in truth.phi_tokens:
            if token.category == PHICategory.AMBIGUOUS:
                if any(t in token.value for t in ["Dr.", "Nurse", "MD", "RN", "Prof."]):
                    pass # keep
                elif any(t in token.value for t in ["Mr.", "Mrs.", "Ms.", "Miss", "patient", "family"]):
                    notes = notes.replace(token.value, "[REDACTED]")
        
        submitted.append({"clinical_notes": notes})
        
    result = task5_reasoning.grade(submitted, truths)
    assert result["passed"] is True
    assert result["score"] >= 0.7


def test_no_redaction_fails():
    """An agent that redacts nothing should fail."""
    truths = _make_truth_records()
    submitted = [{"clinical_notes": t.clinical_notes} for t in truths]
    
    result = task5_reasoning.grade(submitted, truths)
    assert result["passed"] is False
    assert result["breakdown"]["patient_phi_score"] == 0.0


def test_over_redaction_fails():
    """An agent that redacts everything should fail (provider score will be low)."""
    truths = _make_truth_records()
    submitted = [{"clinical_notes": "[ALL REDACTED]"} for t in truths]
    
    result = task5_reasoning.grade(submitted, truths)
    assert result["breakdown"]["provider_phi_score"] == 0.0
    # Should fail because provider_phi_score is 0.0 and weighted into overall
    assert result["passed"] is False


def test_mismatched_length_returns_error():
    truths = _make_truth_records()
    result = task5_reasoning.grade([], truths)
    assert result["score"] == 0.0001
    assert "error" in result["breakdown"]


def test_empty_truth_returns_perfect_score():
    result = task5_reasoning.grade([], [])
    assert result["score"] == 0.0001
