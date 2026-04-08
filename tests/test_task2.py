"""Tests for Task 2 — PHI Detection & Redaction grader."""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generator import EHRGenerator
from src.tasks import task2_redaction


def _make_annotated(seed: int = 42, n: int = 4):
    gen = EHRGenerator(seed=seed)
    return gen.make_annotated_records(n=n)


def _fully_redacted(records):
    """Return records with all known PHI fields replaced."""
    from src.baseline_agent import _redact_record
    return [_redact_record(r.model_dump()) for r in records]


def test_full_phi_removal_scores_phi_1():
    """Rule-based redaction (including last-name alias regex) achieves phi_score=1.0."""
    annotated = _make_annotated()
    submitted = _fully_redacted(annotated)
    result = task2_redaction.grade(submitted, annotated)
    bd = result["breakdown"]
    assert bd["phi_score"] >= 0.99, f"Expected phi_score~1.0, got {bd['phi_score']}"


def test_phi_leak_triggers_penalty():
    """Leaking one PHI value should lower the score compared to full redaction."""
    annotated = _make_annotated(n=2)
    full_redact = _fully_redacted(annotated)
    leaked = copy.deepcopy(full_redact)
    # Re-insert a real name into the first record's notes
    real_name = annotated[0].patient_name
    leaked[0]["clinical_notes"] = f"Patient {real_name} " + leaked[0].get("clinical_notes", "")

    full_result = task2_redaction.grade(full_redact, annotated)
    leaked_result = task2_redaction.grade(leaked, annotated)
    assert leaked_result["score"] < full_result["score"], (
        "PHI leak should lower score"
    )


def test_clinical_keywords_preserved_raises_utility():
    """Preserving clinical notes should give utility_score > 0."""
    annotated = _make_annotated(n=2)
    submitted = _fully_redacted(annotated)
    result = task2_redaction.grade(submitted, annotated)
    bd = result["breakdown"]
    assert bd["utility_score"] > 0.0, "Clinical keywords should survive redaction"


def test_pass_bar_requires_full_phi_removal():
    """Partial redaction must not pass the task."""
    annotated = _make_annotated(n=2)
    submitted = [r.model_dump() for r in annotated]  # no redaction at all
    result = task2_redaction.grade(submitted, annotated)
    assert not result["passed"], "Un-redacted records must not pass"


def test_score_between_0_and_1():
    annotated = _make_annotated()
    submitted = [r.model_dump() for r in annotated]
    result = task2_redaction.grade(submitted, annotated)
    assert 0.0 <= result["score"] <= 1.0


def test_breakdown_per_record():
    annotated = _make_annotated(n=3)
    submitted = _fully_redacted(annotated)
    result = task2_redaction.grade(submitted, annotated)
    assert len(result["breakdown"]["per_record"]) == 3


# ============================================================================
# Edge Cases: Empty and Special Notes
# ============================================================================

def test_empty_notes_edge_case():
    """Empty clinical notes should not crash and give valid scores."""
    annotated = _make_annotated(n=2)
    submitted = _fully_redacted(annotated)
    # Clear all clinical notes
    for rec in submitted:
        rec["clinical_notes"] = ""
    result = task2_redaction.grade(submitted, annotated)
    assert 0.0 <= result["score"] <= 1.0
    # Utility should be 0 if notes are empty
    assert result["breakdown"]["utility_score"] == 0.0


def test_notes_with_no_phi():
    """Notes that have no PHI to begin with should score phi=1.0."""
    from src.models import AnnotatedRecord, PHIToken, PHICategory
    
    # Create minimal record with no PHI in notes
    record = AnnotatedRecord(
        record_id="test-no-phi",
        mrn="[REDACTED_MRN]",
        patient_name="[REDACTED_NAME]",
        dob="[REDACTED_DOB]",
        gender="M",
        icd10_codes=["I10"],
        clinical_notes="Patient presents with hypertension. Vitals stable.",
        phi_tokens=[],  # No PHI tokens
        clinical_keywords=["hypertension", "vitals"],
    )
    submitted = [record.model_dump()]
    result = task2_redaction.grade(submitted, [record])
    assert result["breakdown"]["phi_score"] == 1.0, "No PHI means perfect phi_score"


def test_all_phi_notes():
    """Notes consisting mostly of PHI should have low utility after redaction."""
    annotated = _make_annotated(n=1)
    submitted = _fully_redacted(annotated)
    # Replace clinical notes with just redaction tokens
    submitted[0]["clinical_notes"] = "[REDACTED_NAME] [REDACTED_DOB] [REDACTED_MRN]"
    result = task2_redaction.grade(submitted, annotated)
    # Should have low utility since no clinical content remains
    assert result["breakdown"]["utility_score"] < 0.5


# ============================================================================
# Structured Field vs Free-Text Note Redaction
# ============================================================================

def test_structured_field_redaction_only():
    """Redacting only structured fields but leaving notes untouched should fail."""
    annotated = _make_annotated(n=2)
    submitted = []
    for rec in annotated:
        d = rec.model_dump()
        # Redact structured fields only
        d["patient_name"] = "[REDACTED_NAME]"
        d["mrn"] = "[REDACTED_MRN]"
        d["dob"] = "[REDACTED_DOB]"
        d["phone"] = "[REDACTED_PHONE]"
        d["email"] = "[REDACTED_EMAIL]"
        d["address"] = "[REDACTED_ADDRESS]"
        # Leave clinical_notes untouched (contains embedded PHI)
        submitted.append(d)
    result = task2_redaction.grade(submitted, annotated)
    # Should fail because clinical notes still contain PHI
    assert result["breakdown"]["phi_score"] < 1.0
    assert not result["passed"]


def test_freetext_redaction_but_structured_fields_leaked():
    """Redacting notes but leaving structured fields should fail."""
    annotated = _make_annotated(n=2)
    submitted = []
    for rec in annotated:
        d = rec.model_dump()
        # Leave structured PHI fields as-is
        # But redact clinical notes completely
        d["clinical_notes"] = "[REDACTED clinical notes]"
        submitted.append(d)
    result = task2_redaction.grade(submitted, annotated)
    # Structured PHI fields leak -> phi_score < 1
    assert result["breakdown"]["phi_score"] < 1.0
    assert not result["passed"]


# ============================================================================
# Clinical Utility Retention Scoring
# ============================================================================

def test_utility_preserved_with_clinical_content():
    """Redaction that preserves clinical keywords should have high utility."""
    annotated = _make_annotated(n=2)
    submitted = _fully_redacted(annotated)
    result = task2_redaction.grade(submitted, annotated)
    # Baseline redaction should preserve clinical keywords
    assert result["breakdown"]["utility_score"] > 0.5


def test_utility_destroyed_when_all_content_removed():
    """Removing all content destroys utility."""
    annotated = _make_annotated(n=2)
    submitted = _fully_redacted(annotated)
    # Destroy all clinical content
    for rec in submitted:
        rec["clinical_notes"] = "[FULLY REDACTED]"
        rec["icd10_codes"] = []
        rec["medications"] = []
    result = task2_redaction.grade(submitted, annotated)
    assert result["breakdown"]["utility_score"] < 0.5


def test_utility_partial_when_some_keywords_retained():
    """Partial keyword retention gives partial utility credit."""
    annotated = _make_annotated(n=2)
    full_redact = _fully_redacted(annotated)
    
    # Partial: keep some but not all clinical keywords
    partial_redact = copy.deepcopy(full_redact)
    for rec in partial_redact:
        notes = rec.get("clinical_notes", "")
        # Remove half the content
        rec["clinical_notes"] = notes[:len(notes)//2]
    
    full_result = task2_redaction.grade(full_redact, annotated)
    partial_result = task2_redaction.grade(partial_redact, annotated)
    
    # Partial retention should have lower or equal utility
    assert partial_result["breakdown"]["utility_score"] <= full_result["breakdown"]["utility_score"]


# ============================================================================
# Boundary Testing Around Pass Bar
# ============================================================================

def test_pass_bar_boundary_phi_1_utility_exactly_08():
    """Pass bar requires phi_score==1.0 AND utility_score>=0.8."""
    from src.models import AnnotatedRecord, PHIToken, PHICategory
    
    # Create a controlled record where we can achieve exact scores
    annotated = _make_annotated(n=2)
    submitted = _fully_redacted(annotated)
    
    result = task2_redaction.grade(submitted, annotated)
    bd = result["breakdown"]
    
    # If we have perfect phi removal and good utility, should pass
    if bd["phi_score"] >= 1.0 and bd["utility_score"] >= 0.8:
        assert result["passed"] is True
    else:
        assert result["passed"] is False


def test_pass_bar_fails_with_phi_score_below_1():
    """Any PHI leak should fail regardless of utility."""
    annotated = _make_annotated(n=2)
    submitted = _fully_redacted(annotated)
    # Leak one PHI value
    submitted[0]["clinical_notes"] += f" {annotated[0].patient_name}"
    result = task2_redaction.grade(submitted, annotated)
    assert not result["passed"], "PHI leak should cause failure"


def test_pass_bar_fails_with_utility_below_08():
    """High phi_score but low utility should fail."""
    annotated = _make_annotated(n=2)
    submitted = _fully_redacted(annotated)
    # Destroy clinical utility
    for rec in submitted:
        rec["clinical_notes"] = "[REDACTED]"
    result = task2_redaction.grade(submitted, annotated)
    bd = result["breakdown"]
    # Even if phi_score is 1.0, low utility means failure
    if bd["utility_score"] < 0.8:
        assert not result["passed"]


# ============================================================================
# Partial Redaction Penalty Testing
# ============================================================================

def test_multiple_phi_leaks_compound_penalty():
    """Multiple PHI leaks should compound the penalty."""
    annotated = _make_annotated(n=2)
    full_redact = _fully_redacted(annotated)
    
    # Single leak
    single_leak = copy.deepcopy(full_redact)
    single_leak[0]["clinical_notes"] += f" {annotated[0].patient_name}"
    
    # Double leak
    double_leak = copy.deepcopy(full_redact)
    double_leak[0]["clinical_notes"] += f" {annotated[0].patient_name}"
    double_leak[0]["clinical_notes"] += f" {annotated[0].mrn}"
    
    single_result = task2_redaction.grade(single_leak, annotated)
    double_result = task2_redaction.grade(double_leak, annotated)
    
    assert double_result["score"] < single_result["score"], "More leaks = lower score"


def test_phi_leak_categories_tracked():
    """Leaked PHI categories should be tracked in breakdown."""
    annotated = _make_annotated(n=1)
    submitted = _fully_redacted(annotated)
    # Leak a name
    submitted[0]["clinical_notes"] += f" {annotated[0].patient_name}"
    result = task2_redaction.grade(submitted, annotated)
    per_rec = result["breakdown"]["per_record"][0]
    assert len(per_rec["leaked_categories"]) > 0


def test_missing_structured_redaction_token_is_penalized_even_if_token_exists_elsewhere():
    """A token present in notes should not mask missing redaction in the source structured field."""
    annotated = _make_annotated(n=1)
    submitted = _fully_redacted(annotated)

    # Keep note token present, but remove structured redaction in patient_name.
    submitted[0]["patient_name"] = ""
    if "[REDACTED_NAME]" not in submitted[0].get("clinical_notes", ""):
        submitted[0]["clinical_notes"] += " [REDACTED_NAME]"

    result = task2_redaction.grade(submitted, annotated)
    assert result["breakdown"]["phi_score"] < 1.0
    assert not result["passed"]


# ============================================================================
# Scoring Formula Verification
# ============================================================================

def test_score_formula_weighted_correctly():
    """Verify score = phi_score * 0.6 + utility_score * 0.4."""
    annotated = _make_annotated(n=2)
    submitted = _fully_redacted(annotated)
    result = task2_redaction.grade(submitted, annotated)
    bd = result["breakdown"]
    
    expected = max(0.0, min(1.0, bd["phi_score"] * 0.6 + bd["utility_score"] * 0.4))
    
    assert abs(result["score"] - expected) < 0.01, (
        f"Score mismatch: got {result['score']}, expected {expected}"
    )


def test_no_ground_truth_returns_zero():
    """Empty ground truth should return the strict lower bound score."""
    result = task2_redaction.grade([], [])
    assert result["score"] == 0.0001
    assert not result["passed"]
