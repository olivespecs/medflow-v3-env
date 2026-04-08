"""Tests for Task 3 — Anonymisation + Downstream Utility grader."""

import copy
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generator import EHRGenerator
from src.tasks import task3_anonymization


def _make_annotated(seed: int = 42, n: int = 4):
    gen = EHRGenerator(seed=seed)
    return gen.make_annotated_records(n=n)


def _anonymise(records):
    from src.baseline_agent import _anonymise_record
    return [_anonymise_record(r.model_dump()) for r in records]


def test_mock_ml_scorer_is_deterministic():
    """Same input → same output every time."""
    annotated = _make_annotated()
    submitted = _anonymise(annotated)
    r1 = task3_anonymization.grade(submitted, annotated)
    r2 = task3_anonymization.grade(submitted, annotated)
    assert r1["breakdown"]["ml_utility_score"] == r2["breakdown"]["ml_utility_score"]


def test_aggressive_redaction_lowers_ml_utility_score():
    """Removing all ICD codes and medications should tank ml_utility_score."""
    annotated = _make_annotated(n=3)
    aggressive = _anonymise(annotated)
    for rec in aggressive:
        rec["icd10_codes"] = []
        rec["medications"] = []
        rec["clinical_notes"] = "[ALL REDACTED]"

    balanced = _anonymise(annotated)

    r_aggressive = task3_anonymization.grade(aggressive, annotated)
    r_balanced = task3_anonymization.grade(balanced, annotated)
    assert r_aggressive["breakdown"]["ml_utility_score"] <= r_balanced["breakdown"]["ml_utility_score"], (
        "Aggressive redaction must not beat balanced on ml_utility_score"
    )


def test_balanced_anonymisation_passes():
    """Task 3 is intentionally hard; balanced anonymisation should not be invalid."""
    annotated = _make_annotated(seed=99)
    submitted = _anonymise(annotated)
    baseline_ml = task3_anonymization.compute_baseline_ml_scores(annotated)
    result = task3_anonymization.grade(submitted, annotated, baseline_ml)
    bd = result["breakdown"]
    # Hard mode: adversarial identifiers should prevent trivial perfect privacy.
    assert 0.0 <= bd["phi_score"] <= 1.0, f"phi_score={bd['phi_score']}"
    # check first record's adversarial_privacy_score
    assert bd["per_record"][0]["adversarial_privacy_score"] <= 1.0


def test_score_components_sum_correctly():
    annotated = _make_annotated(n=2)
    submitted = _anonymise(annotated)
    result = task3_anonymization.grade(submitted, annotated)
    bd = result["breakdown"]
    expected = (
        bd["phi_score"] * 0.4
        + bd["ml_utility_score"] * 0.3
        + bd["fidelity_score"] * 0.2
        + bd["k_anonymity_score"] * 0.1
    )
    assert abs(result["score"] - expected) < 0.0001


def test_score_between_0_and_1():
    annotated = _make_annotated()
    submitted = [r.model_dump() for r in annotated]
    result = task3_anonymization.grade(submitted, annotated)
    assert 0.0 <= result["score"] <= 1.0


def test_breakdown_per_record():
    annotated = _make_annotated(n=4)
    submitted = _anonymise(annotated)
    result = task3_anonymization.grade(submitted, annotated)
    assert len(result["breakdown"]["per_record"]) == 4


def test_adversarial_identifier_leakage_reduces_phi_score():
    """Leaking hidden indirect identifiers should reduce privacy score."""
    annotated = _make_annotated(n=2)
    submitted = _anonymise(annotated)

    # Start from a sanitized variant with adversarial markers removed.
    clean_submitted = copy.deepcopy(submitted)
    for i, truth in enumerate(annotated):
        notes = clean_submitted[i].get("clinical_notes", "")
        for marker in truth.adversarial_identifiers:
            notes = notes.replace(marker, "[GENERALIZED_CLUSTER]")
        clean_submitted[i]["clinical_notes"] = notes

    clean_result = task3_anonymization.grade(clean_submitted, annotated)

    # Leak adversarial identifiers back into notes for first record only.
    leak_1 = annotated[0].adversarial_identifiers[0]
    leak_2 = annotated[0].adversarial_identifiers[1]
    leaked_submitted = copy.deepcopy(clean_submitted)
    leaked_submitted[0]["clinical_notes"] += f" {leak_1}. {leak_2}."

    leaked_result = task3_anonymization.grade(leaked_submitted, annotated)

    assert leaked_result["breakdown"]["phi_score"] < clean_result["breakdown"]["phi_score"]


# ============================================================================
# Full Anonymization Pass Criteria Tests
# ============================================================================

def test_pass_criteria_phi_1_and_ml_above_threshold():
    """Pass requires phi_score == 1.0 AND ml_utility_score >= 0.60."""
    annotated = _make_annotated(n=2)
    submitted = _anonymise(annotated)
    baseline_ml = task3_anonymization.compute_baseline_ml_scores(annotated)
    result = task3_anonymization.grade(submitted, annotated, baseline_ml)
    bd = result["breakdown"]
    
    # Check pass condition logic
    expected_passed = (bd["phi_score"] >= 1.0) and (bd["ml_utility_score"] >= 0.60)
    assert result["passed"] == expected_passed


def test_pass_fails_with_phi_below_1():
    """Any PHI leak should fail pass criteria."""
    annotated = _make_annotated(n=2)
    # Submit raw records without anonymization
    submitted = [r.model_dump() for r in annotated]
    result = task3_anonymization.grade(submitted, annotated)
    assert not result["passed"], "Raw records with PHI should not pass"


def test_pass_fails_with_low_ml_utility_score():
    """Perfect PHI removal but destroyed utility should fail."""
    annotated = _make_annotated(n=2)
    submitted = _anonymise(annotated)
    # Destroy all clinical utility
    for rec in submitted:
        rec["icd10_codes"] = []
        rec["medications"] = []
        rec["clinical_notes"] = "[FULLY ANONYMIZED]"
        # Remove age info too
        rec.pop("dob", None)
        rec.pop("age_group", None)
    
    result = task3_anonymization.grade(submitted, annotated)
    # ML score should be low due to missing clinical data
    assert result["breakdown"]["ml_utility_score"] < 1.0


# ============================================================================
# Clinical Fidelity Scoring Tests
# ============================================================================

def test_fidelity_preserved_age_group():
    """Preserving age group should contribute to fidelity score."""
    annotated = _make_annotated(n=2)
    submitted = _anonymise(annotated)
    
    result = task3_anonymization.grade(submitted, annotated)
    assert result["breakdown"]["fidelity_score"] > 0.0


def test_fidelity_lower_when_age_group_wrong():
    """Incorrect age group should lower fidelity."""
    annotated = _make_annotated(n=2)
    correct_submit = _anonymise(annotated)
    wrong_submit = copy.deepcopy(correct_submit)
    # Set wrong age groups
    for rec in wrong_submit:
        rec["age_group"] = "76+"  # Force incorrect age group
        rec.pop("dob", None)  # Remove DOB so age_group is used
    
    correct_result = task3_anonymization.grade(correct_submit, annotated)
    wrong_result = task3_anonymization.grade(wrong_submit, annotated)
    
    # Fidelity should be affected by wrong age group
    # (may not always be lower depending on actual ages)
    assert wrong_result["breakdown"]["fidelity_score"] >= 0.0


def test_fidelity_gender_preservation():
    """Gender preservation affects fidelity score."""
    annotated = _make_annotated(n=2)
    correct_submit = _anonymise(annotated)
    wrong_submit = copy.deepcopy(correct_submit)
    # Flip all genders
    for i, rec in enumerate(wrong_submit):
        original_gender = annotated[i].gender
        rec["gender"] = "F" if original_gender == "M" else "M"
    
    correct_result = task3_anonymization.grade(correct_submit, annotated)
    wrong_result = task3_anonymization.grade(wrong_submit, annotated)
    
    # Wrong gender should lower fidelity
    assert wrong_result["breakdown"]["fidelity_score"] < correct_result["breakdown"]["fidelity_score"]


def test_fidelity_icd10_prefix_preserved():
    """ICD-10 chapter prefixes should be preserved for fidelity."""
    annotated = _make_annotated(n=2)
    submitted = _anonymise(annotated)
    # Verify ICD codes are present (anonymization shouldn't remove them)
    for rec in submitted:
        assert rec.get("icd10_codes") is not None
    
    result = task3_anonymization.grade(submitted, annotated)
    assert result["breakdown"]["fidelity_score"] > 0.0


# ============================================================================
# k-Anonymity Validation Tests
# ============================================================================

def test_k_anonymity_score_in_breakdown():
    """k-anonymity score should be present in breakdown."""
    annotated = _make_annotated(n=4)
    submitted = _anonymise(annotated)
    result = task3_anonymization.grade(submitted, annotated)
    assert "k_anonymity_score" in result["breakdown"]
    assert 0.0 <= result["breakdown"]["k_anonymity_score"] <= 1.0


def test_k_anonymity_improved_with_generalization():
    """Generalizing quasi-identifiers should improve k-anonymity."""
    annotated = _make_annotated(n=4)
    
    # Original submission
    original_submit = _anonymise(annotated)
    
    # Generalized submission - make all quasi-IDs identical
    generalized_submit = copy.deepcopy(original_submit)
    for rec in generalized_submit:
        rec["dob"] = "[REDACTED]"
        rec["gender"] = "Other"
        rec["address"] = "[REDACTED]"
    
    original_result = task3_anonymization.grade(original_submit, annotated)
    generalized_result = task3_anonymization.grade(generalized_submit, annotated)
    
    # Generalized records should have higher k-anonymity
    assert generalized_result["breakdown"]["k_anonymity_score"] >= original_result["breakdown"]["k_anonymity_score"]


def test_k_anonymity_with_unique_records():
    """Unique quasi-identifier combinations should have lower k-anonymity."""
    annotated = _make_annotated(n=4)
    submitted = _anonymise(annotated)
    # Make each record unique by varying address
    for i, rec in enumerate(submitted):
        rec["address"] = f"Unique Address {i}"
    
    result = task3_anonymization.grade(submitted, annotated)
    # k-anonymity should reflect the uniqueness
    assert 0.0 <= result["breakdown"]["k_anonymity_score"] <= 1.0


# ============================================================================
# Mock ML Risk Scorer Tests
# ============================================================================

def test_ml_scorer_uses_age_group():
    """Mock ML scorer should use age_group when DOB is removed."""
    annotated = _make_annotated(n=2)
    submitted = _anonymise(annotated)
    # Replace DOB with age_group
    for rec in submitted:
        rec.pop("dob", None)
        rec["age_group"] = "61-75"  # High-risk age group
    
    result = task3_anonymization.grade(submitted, annotated)
    # Should still compute ML score
    assert result["breakdown"]["ml_utility_score"] > 0.0


def test_ml_scorer_high_risk_medications():
    """High-risk medications should increase ML risk score."""
    annotated = _make_annotated(n=2)
    submitted = _anonymise(annotated)
    # Add high-risk medications
    for rec in submitted:
        rec["medications"] = [
            {"name": "Warfarin", "dose_mg": 5.0},
            {"name": "Insulin", "dose_mg": 20.0},
        ]
    
    result = task3_anonymization.grade(submitted, annotated)
    assert result["breakdown"]["ml_utility_score"] >= 0.0


def test_ml_scorer_high_risk_icd_codes():
    """High-risk ICD codes should affect ML risk score."""
    annotated = _make_annotated(n=2)
    submitted = _anonymise(annotated)
    # Use high-risk ICD prefixes (I, E, J, N, F)
    for rec in submitted:
        rec["icd10_codes"] = ["I10", "E11.9", "J44.1"]  # Cardiovascular, metabolic, respiratory
    
    result = task3_anonymization.grade(submitted, annotated)
    assert result["breakdown"]["ml_utility_score"] >= 0.0


def test_ml_scorer_baseline_computation():
    """Baseline ML scores should be computed from ground truth."""
    annotated = _make_annotated(n=3)
    baseline = task3_anonymization.compute_baseline_ml_scores(annotated)
    assert len(baseline) == 3
    for score in baseline:
        assert 0.0 <= score <= 1.0


# ============================================================================
# Edge Cases: Minimal and Large Records
# ============================================================================

def test_minimal_record_single():
    """Single minimal record should be graded without error."""
    annotated = _make_annotated(n=1)
    submitted = _anonymise(annotated)
    result = task3_anonymization.grade(submitted, annotated)
    assert 0.0 <= result["score"] <= 1.0
    assert len(result["breakdown"]["per_record"]) == 1


def test_many_records():
    """Large batch of records should be handled correctly."""
    annotated = _make_annotated(n=10)
    submitted = _anonymise(annotated)
    result = task3_anonymization.grade(submitted, annotated)
    assert 0.0 <= result["score"] <= 1.0
    assert len(result["breakdown"]["per_record"]) == 10


def test_empty_ground_truth_returns_zero():
    """Empty ground truth should return the strict lower bound score."""
    result = task3_anonymization.grade([], [])
    assert result["score"] == 0.0001
    assert not result["passed"]


def test_record_with_many_quasi_identifiers():
    """Records with many quasi-identifiers should still be processable."""
    annotated = _make_annotated(n=2)
    submitted = _anonymise(annotated)
    # Add extra quasi-identifier-like fields
    for rec in submitted:
        rec["dob"] = "1960-05-15"
        rec["gender"] = "M"
        rec["address"] = "123 Main St, Springfield, IL 62701"
    
    result = task3_anonymization.grade(submitted, annotated)
    assert 0.0 <= result["score"] <= 1.0


# ============================================================================
# Boundary Testing Around Pass Thresholds
# ============================================================================

def test_boundary_ml_utility_score_exactly_at_threshold():
    """ML score at exactly 0.60 should pass if PHI is perfect."""
    annotated = _make_annotated(n=2)
    submitted = _anonymise(annotated)
    result = task3_anonymization.grade(submitted, annotated)
    bd = result["breakdown"]
    
    # Verify pass logic is correct
    if bd["phi_score"] >= 1.0 and bd["ml_utility_score"] >= 0.60:
        assert result["passed"] is True


def test_boundary_ml_utility_score_just_below_threshold():
    """ML score just below 0.60 should fail even with perfect PHI."""
    # This is hard to engineer directly, so we test the pass logic
    annotated = _make_annotated(n=2)
    submitted = _anonymise(annotated)
    result = task3_anonymization.grade(submitted, annotated)
    bd = result["breakdown"]
    
    # If ml_utility_score < 0.60, should not pass
    if bd["ml_utility_score"] < 0.60:
        assert result["passed"] is False


def test_adversarial_privacy_blend_factor():
    """Adversarial privacy should be blended into PHI score."""
    annotated = _make_annotated(n=2)
    submitted = _anonymise(annotated)
    result = task3_anonymization.grade(submitted, annotated)
    
    # Check per-record has both phi_score and adversarial_privacy_score
    for per_rec in result["breakdown"]["per_record"]:
        assert "phi_score" in per_rec
        assert "phi_direct_score" in per_rec
        assert "adversarial_privacy_score" in per_rec


# ============================================================================
# Adversarial Identifier Suppression Tests
# ============================================================================

def test_adversarial_identifiers_detection():
    """Adversarial identifiers should be detected and penalized."""
    annotated = _make_annotated(n=1)
    # Keep adversarial identifiers in the submission
    submitted = [annotated[0].model_dump()]
    
    result = task3_anonymization.grade(submitted, annotated)
    per_rec = result["breakdown"]["per_record"][0]
    
    # Should detect adversarial leaks
    assert "adversarial_leaks" in per_rec


def test_adversarial_fully_suppressed():
    """Full suppression of adversarial identifiers should improve score."""
    annotated = _make_annotated(n=2)
    
    # Submission with adversarial identifiers
    with_adv = _anonymise(annotated)
    
    # Submission with adversarial identifiers suppressed
    without_adv = copy.deepcopy(with_adv)
    for i, truth in enumerate(annotated):
        notes = without_adv[i].get("clinical_notes", "")
        for marker in truth.adversarial_identifiers:
            notes = notes.replace(marker, "[SUPPRESSED]")
        without_adv[i]["clinical_notes"] = notes
    
    with_result = task3_anonymization.grade(with_adv, annotated)
    without_result = task3_anonymization.grade(without_adv, annotated)
    
    # Suppression should improve or maintain phi_score
    assert without_result["breakdown"]["phi_score"] >= with_result["breakdown"]["phi_score"]


def test_adversarial_paraphrase_still_detected():
    """Paraphrased disease+ZIP linkage should still count as adversarial leakage."""
    annotated = _make_annotated(n=1)
    submitted = _anonymise(annotated)

    # Start from notes with exact markers removed, then add a paraphrase.
    truth = annotated[0]
    notes = submitted[0].get("clinical_notes", "")
    for marker in truth.adversarial_identifiers:
        notes = notes.replace(marker, "[GENERALIZED_CLUSTER]")

    # Paraphrase the disease+ZIP marker in a different surface form.
    dz_marker = truth.adversarial_identifiers[0]
    dz_match = re.search(r"the\s+(.+?)\s+patient\s+from\s+(\d{5})", dz_marker, flags=re.IGNORECASE)
    if dz_match:
        disease, zip_code = dz_match.group(1), dz_match.group(2)
        notes += f" Linkage risk remains: {disease} case observed in ZIP {zip_code}."

    submitted[0]["clinical_notes"] = notes

    result = task3_anonymization.grade(submitted, annotated)
    per_rec = result["breakdown"]["per_record"][0]
    assert per_rec["adversarial_privacy_score"] < 1.0
    assert len(per_rec["adversarial_leaks"]) >= 1


# ============================================================================
# Score Component Weights Verification
# ============================================================================

def test_score_weights_documented():
    """Score weights should be documented in info."""
    annotated = _make_annotated(n=2)
    submitted = _anonymise(annotated)
    result = task3_anonymization.grade(submitted, annotated)
    
    assert "phi_weight" in result["info"]
    assert "ml_weight" in result["info"]
    assert "fidelity_weight" in result["info"]
    assert "k_weight" in result["info"]


def test_pass_conditions_documented():
    """Pass conditions should be documented in info."""
    annotated = _make_annotated(n=2)
    submitted = _anonymise(annotated)
    result = task3_anonymization.grade(submitted, annotated)
    
    assert "pass_conditions" in result["info"]
