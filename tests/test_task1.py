"""Tests for Task 1 — Data Hygiene & Standardisation grader."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generator import EHRGenerator
from src.tasks import task1_hygiene


def _make_env(seed: int = 42, n: int = 4):
    gen = EHRGenerator(seed=seed)
    return gen.make_dirty_records(n=n)


def test_perfect_agent_scores_high():
    """Submitting ground-truth records should score >= pass bar."""
    _, truths = _make_env()
    submitted = [r.model_dump() for r in truths]
    result = task1_hygiene.grade(submitted, truths)
    assert result["score"] >= task1_hygiene.PASS_BAR, (
        f"Perfect agent should pass; got {result['score']}"
    )


def test_untouched_dirty_records_score_below_perfect():
    """
    Submitting un-fixed dirty records should score below 1.0 — we can't
    guarantee they score below the 0.85 pass bar because only 1–3 out of
    8 fields are deliberately broken by the generator.
    """
    dirty, truths = _make_env()
    submitted = [r.model_dump() for r in dirty]
    result = task1_hygiene.grade(submitted, truths)
    # At least some fields must be wrong
    assert result["score"] < 1.0, (
        f"Dirty records should not be perfect; got {result['score']}"
    )


def test_all_empty_submission_scores_zero():
    """Submitting all-empty dicts should score near 0."""
    _, truths = _make_env()
    submitted = [{}] * len(truths)
    result = task1_hygiene.grade(submitted, truths)
    assert result["score"] < 0.3, f"Empty submission should score low; got {result['score']}"


def test_partial_fix_gives_partial_credit():
    """Fixing ICD codes only should raise the score vs untouched."""
    # Use more records to increase chance of ICD flaws being present
    dirty, truths = _make_env(seed=0, n=6)
    submitted_untouched = [r.model_dump() for r in dirty]
    submitted_partial = [r.model_dump() for r in dirty]

    # Fix ICD codes and DOB in all records (guaranteed to change something)
    for i, truth in enumerate(truths):
        submitted_partial[i]["icd10_codes"] = truth.icd10_codes
        submitted_partial[i]["dob"] = truth.dob

    result_partial = task1_hygiene.grade(submitted_partial, truths)
    result_untouched = task1_hygiene.grade(submitted_untouched, truths)
    # Partial fix either equals (if no flaws injected in those fields) or beats untouched
    assert result_partial["score"] >= result_untouched["score"], (
        "Partially fixed records should not score lower than untouched"
    )


def test_breakdown_contains_per_record():
    """Result breakdown must contain per-record entries."""
    dirty, truths = _make_env(n=3)
    submitted = [r.model_dump() for r in truths]
    result = task1_hygiene.grade(submitted, truths)
    assert "per_record" in result["breakdown"]
    assert len(result["breakdown"]["per_record"]) == 3


def test_score_between_0_and_1():
    dirty, truths = _make_env()
    submitted = [r.model_dump() for r in dirty]
    result = task1_hygiene.grade(submitted, truths)
    assert 0.0 <= result["score"] <= 1.0


def test_grade_empty_submission():
    """Submitting empty list should return 0.0 score without crashing."""
    _, truths = _make_env(n=2)
    result = task1_hygiene.grade([], truths)
    assert result["score"] == 0.0 or result["score"] < task1_hygiene.PASS_BAR


def test_longitudinal_consistency_gender_flip():
    """
    Test the longitudinal majority-vote logic for gender inconsistencies.
    
    Scenario: Same MRN has gender "M" in Visit 1 and "F" in Visit 2.
    Agent must return majority value (or consistent value) across visits.
    """
    from src.models import PatientRecord
    
    # Create two records with same MRN but different genders
    base_dob = "1980-05-15"
    mrn = "MRN123456"
    
    # Visit 1: gender M
    visit1_dict = {
        "record_id": "visit-1",
        "mrn": mrn,
        "patient_name": "John Smith",
        "dob": base_dob,
        "gender": "M",
        "phone": "555-1234",
        "email": "john@example.com",
        "address": "123 Main St",
        "icd10_codes": ["I10"],
        "vitals": {"heart_rate_bpm": 72.0},
        "medications": [],
        "clinical_notes": "Patient presents for checkup",
    }
    
    # Visit 2: gender F (inconsistent - should be resolved by agent)
    visit2_dict = {
        "record_id": "visit-2",
        "mrn": mrn,
        "patient_name": "John Smith",
        "dob": base_dob,
        "gender": "F",  # Inconsistent!
        "phone": "555-1234",
        "email": "john@example.com",
        "address": "123 Main St",
        "icd10_codes": ["I10"],
        "vitals": {"heart_rate_bpm": 75.0},
        "medications": [],
        "clinical_notes": "Follow-up visit",
    }
    
    # Convert to PatientRecord objects for ground truth
    visit1 = PatientRecord(**visit1_dict)
    visit2 = PatientRecord(**visit2_dict)
    truths = [visit1, visit2]
    
    # Test 1: Agent keeps inconsistency (should fail longitudinal check)
    submitted_inconsistent = [visit1_dict.copy(), visit2_dict.copy()]
    result = task1_hygiene.grade(submitted_inconsistent, truths)
    
    # Should have low longitudinal consistency score
    assert "longitudinal_consistency" in result["breakdown"]
    assert result["breakdown"]["longitudinal_consistency"] < 1.0, (
        f"Inconsistent submission should not get perfect longitudinal score; "
        f"got {result['breakdown']['longitudinal_consistency']}"
    )
    
    # Test 2: Agent fixes to consistent value (all M)
    submitted_fixed_m = [
        visit1_dict.copy(),
        {**visit1_dict, "record_id": "visit-2"},
    ]
    result_fixed = task1_hygiene.grade(submitted_fixed_m, truths)
    
    assert result_fixed["breakdown"]["longitudinal_consistency"] == 1.0, (
        f"Consistent submission should get perfect longitudinal score; "
        f"got {result_fixed['breakdown']['longitudinal_consistency']}"
    )
    
    # Test 3: Agent fixes to consistent value (all F)
    submitted_fixed_f = [
        {**visit2_dict, "record_id": "visit-1"},
        visit2_dict.copy(),
    ]
    result_fixed_f = task1_hygiene.grade(submitted_fixed_f, truths)
    
    assert result_fixed_f["breakdown"]["longitudinal_consistency"] == 1.0, (
        f"Consistent submission should get perfect longitudinal score; "
        f"got {result_fixed_f['breakdown']['longitudinal_consistency']}"
    )


def test_longitudinal_consistency_dob_shift():
    """
    Test longitudinal consistency for DOB inconsistencies.
    
    Scenario: Same MRN has DOB shifted by 1 year between visits.
    Agent must normalize to consistent DOB across visits.
    """
    from src.models import PatientRecord
    
    mrn = "MRN789012"
    
    # Visit 1: correct DOB
    visit1_dict = {
        "record_id": "visit-1",
        "mrn": mrn,
        "patient_name": "Jane Doe",
        "dob": "1990-03-22",
        "gender": "F",
        "phone": "555-5678",
        "email": "jane@example.com",
        "address": "456 Oak Ave",
        "icd10_codes": ["E11.9"],
        "vitals": {"heart_rate_bpm": 68.0},
        "medications": [],
        "clinical_notes": "Diabetes checkup",
    }
    
    # Visit 2: DOB shifted by 1 year (common OCR/scanning error)
    visit2_dict = {
        "record_id": "visit-2",
        "mrn": mrn,
        "patient_name": "Jane Doe",
        "dob": "1991-03-22",  # Shifted by 1 year!
        "gender": "F",
        "phone": "555-5678",
        "email": "jane@example.com",
        "address": "456 Oak Ave",
        "icd10_codes": ["E11.9"],
        "vitals": {"heart_rate_bpm": 70.0},
        "medications": [],
        "clinical_notes": "Follow-up",
    }
    
    # Convert to PatientRecord objects
    visit1 = PatientRecord(**visit1_dict)
    visit2 = PatientRecord(**visit2_dict)
    truths = [visit1, visit2]
    
    # Agent keeps inconsistency
    submitted_inconsistent = [visit1_dict.copy(), visit2_dict.copy()]
    result = task1_hygiene.grade(submitted_inconsistent, truths)
    
    assert "longitudinal_consistency" in result["breakdown"]
    assert result["breakdown"]["longitudinal_consistency"] < 1.0, (
        f"Inconsistent DOB should not get perfect longitudinal score; "
        f"got {result['breakdown']['longitudinal_consistency']}"
    )
    
    # Agent normalizes to first visit DOB
    submitted_normalized = [
        visit1_dict.copy(),
        {**visit1_dict, "record_id": "visit-2"},
    ]
    result_normalized = task1_hygiene.grade(submitted_normalized, truths)
    
    assert result_normalized["breakdown"]["longitudinal_consistency"] == 1.0, (
        f"Normalized DOB should get perfect longitudinal score; "
        f"got {result_normalized['breakdown']['longitudinal_consistency']}"
    )
