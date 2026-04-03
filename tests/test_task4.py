"""Tests for Task 4 — Clinical Knowledge Extraction & Summarization grader."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generator import EHRGenerator
from src.tasks import task4_knowledge


def _make_env(seed: int = 42, n: int = 4):
    gen = EHRGenerator(seed=seed)
    # Task 4 uses clean records for extraction. 
    # EHRGenerator has make_clean_record() for a single record.
    return [gen.make_clean_record() for _ in range(n)]


def _ref_summary(truth):
    return task4_knowledge._build_reference_summary(truth)


def test_perfect_agent_scores_high():
    """Submitting perfect entities and summaries should score high."""
    truths = _make_env(n=2)
    submitted = []
    for truth in truths:
        # Extract perfect entities using the same logic as the grader
        entities = task4_knowledge._extract_ground_truth_entities(truth)
        # Use clinical notes as a perfect summary for this test
        submitted.append({
            "entities": entities,
            "summary": _ref_summary(truth)
        })
    
    result = task4_knowledge.grade(submitted, truths)
    assert result["score"] >= task4_knowledge.PASS_BAR, (
        f"Perfect agent should pass; got {result['score']}"
    )


def test_empty_submission_scores_low():
    """Submitting empty entities and summaries should score low."""
    truths = _make_env(n=2)
    submitted = [{"entities": [], "summary": ""}] * len(truths)
    result = task4_knowledge.grade(submitted, truths)
    assert result["score"] < 0.2, f"Empty submission should score low; got {result['score']}"


def test_partial_extraction_partial_credit():
    """Fixing entities only should give a partial score."""
    truths = _make_env(n=2)
    
    # Untouched/Empty
    submitted_empty = [{"entities": [], "summary": ""}] * len(truths)
    result_empty = task4_knowledge.grade(submitted_empty, truths)
    
    # Correct entities, empty summary
    submitted_partial = []
    for truth in truths:
        entities = task4_knowledge._extract_ground_truth_entities(truth)
        submitted_partial.append({
            "entities": entities,
            "summary": ""
        })
    
    result_partial = task4_knowledge.grade(submitted_partial, truths)
    assert result_partial["score"] > result_empty["score"], (
        f"Partial extraction should beat empty; {result_partial['score']} vs {result_empty['score']}"
    )


def test_score_between_0_and_1():
    truths = _make_env(n=2)
    submitted = [{"entities": [], "summary": "Some random text"}] * len(truths)
    result = task4_knowledge.grade(submitted, truths)
    assert 0.0 <= result["score"] <= 1.0


def test_grade_empty_list():
    """Submitting empty list should return 0.0 score without crashing."""
    truths = _make_env(n=2)
    result = task4_knowledge.grade([], truths)
    assert result["score"] == 0.0


# ============================================================================
# Entity Extraction Scoring Tests (ICD-10 Codes + Medications)
# ============================================================================

def test_icd10_code_extraction_scores():
    """Correct ICD-10 code extraction should score high."""
    truths = _make_env(n=2)
    submitted = []
    for truth in truths:
        entities = []
        # Extract only ICD-10 codes
        for code in truth.icd10_codes:
            entities.append({"text": code, "type": "Condition", "code": code})
        submitted.append({"entities": entities, "summary": _ref_summary(truth)})
    
    result = task4_knowledge.grade(submitted, truths)
    assert result["breakdown"]["avg_code_precision"] >= 0.5


def test_medication_extraction_scores():
    """Correct medication extraction should contribute to entity score."""
    truths = _make_env(n=2)
    submitted = []
    for truth in truths:
        entities = []
        # Extract only medications
        for med in truth.medications:
            entities.append({"text": med.name, "type": "Medication", "code": med.name})
        submitted.append({"entities": entities, "summary": _ref_summary(truth)})
    
    result = task4_knowledge.grade(submitted, truths)
    # Should get partial credit for medications
    assert result["breakdown"]["avg_entity_extraction"] > 0.0


def test_mixed_entity_extraction():
    """Both ICD-10 and medications should be scored together."""
    truths = _make_env(n=2)
    submitted = []
    for truth in truths:
        entities = task4_knowledge._extract_ground_truth_entities(truth)
        submitted.append({"entities": entities, "summary": _ref_summary(truth)})
    
    result = task4_knowledge.grade(submitted, truths)
    assert result["breakdown"]["avg_entity_extraction"] >= 0.75


def test_wrong_entity_types_penalized():
    """Wrong entity types should lower the score."""
    truths = _make_env(n=2)
    # Perfect extraction
    perfect_submitted = []
    for truth in truths:
        entities = task4_knowledge._extract_ground_truth_entities(truth)
        perfect_submitted.append({"entities": entities, "summary": _ref_summary(truth)})
    
    # Wrong types
    wrong_submitted = []
    for truth in truths:
        entities = []
        for code in truth.icd10_codes:
            # Use wrong type ("Medication" instead of "Condition")
            entities.append({"text": code, "type": "Medication", "code": code})
        wrong_submitted.append({"entities": entities, "summary": _ref_summary(truth)})
    
    perfect_result = task4_knowledge.grade(perfect_submitted, truths)
    wrong_result = task4_knowledge.grade(wrong_submitted, truths)
    
    assert wrong_result["breakdown"]["avg_entity_extraction"] < perfect_result["breakdown"]["avg_entity_extraction"]


def test_extra_entities_affect_score():
    """Extra entities beyond ground truth should affect precision."""
    truths = _make_env(n=2)
    
    # Exact entities
    exact_submitted = []
    for truth in truths:
        entities = task4_knowledge._extract_ground_truth_entities(truth)
        exact_submitted.append({"entities": entities, "summary": _ref_summary(truth)})
    
    # Extra entities
    extra_submitted = []
    for truth in truths:
        entities = task4_knowledge._extract_ground_truth_entities(truth)
        # Add fake extra entities
        entities.append({"text": "FakeCode", "type": "Condition", "code": "Z99.99"})
        entities.append({"text": "FakeMed", "type": "Medication", "code": "FakeMed"})
        extra_submitted.append({"entities": entities, "summary": _ref_summary(truth)})
    
    exact_result = task4_knowledge.grade(exact_submitted, truths)
    extra_result = task4_knowledge.grade(extra_submitted, truths)
    
    # Extra entities should lower the score (uses max of truth/submitted count)
    assert extra_result["breakdown"]["avg_entity_extraction"] <= exact_result["breakdown"]["avg_entity_extraction"]


# ============================================================================
# Summary Semantic Fidelity Scoring Tests
# ============================================================================

def test_perfect_summary_scores_high():
    """Perfect summary (same as clinical notes) should score high."""
    truths = _make_env(n=2)
    submitted = []
    for truth in truths:
        entities = task4_knowledge._extract_ground_truth_entities(truth)
        submitted.append({"entities": entities, "summary": _ref_summary(truth)})
    
    result = task4_knowledge.grade(submitted, truths)
    assert result["breakdown"]["avg_summary_fidelity"] >= 0.8


def test_empty_summary_scores_low():
    """Empty summary should score low on fidelity."""
    truths = _make_env(n=2)
    submitted = []
    for truth in truths:
        entities = task4_knowledge._extract_ground_truth_entities(truth)
        submitted.append({"entities": entities, "summary": ""})
    
    result = task4_knowledge.grade(submitted, truths)
    assert result["breakdown"]["avg_summary_fidelity"] < 0.5


def test_partial_summary_partial_credit():
    """Partial summary content should get partial credit."""
    truths = _make_env(n=2)
    
    # Full summary
    full_submitted = []
    for truth in truths:
        entities = task4_knowledge._extract_ground_truth_entities(truth)
        full_submitted.append({"entities": entities, "summary": _ref_summary(truth)})
    
    # Partial summary
    partial_submitted = []
    for truth in truths:
        entities = task4_knowledge._extract_ground_truth_entities(truth)
        # Use only first half of notes
        ref = _ref_summary(truth)
        partial_submitted.append({"entities": entities, "summary": ref[:len(ref)//2]})
    
    full_result = task4_knowledge.grade(full_submitted, truths)
    partial_result = task4_knowledge.grade(partial_submitted, truths)
    
    # Partial should score lower than full
    assert partial_result["breakdown"]["avg_summary_fidelity"] <= full_result["breakdown"]["avg_summary_fidelity"]


def test_irrelevant_summary_scores_low():
    """Completely irrelevant summary should score low."""
    truths = _make_env(n=2)
    submitted = []
    for truth in truths:
        entities = task4_knowledge._extract_ground_truth_entities(truth)
        submitted.append({"entities": entities, "summary": "The quick brown fox jumps over the lazy dog."})
    
    result = task4_knowledge.grade(submitted, truths)
    assert result["breakdown"]["avg_summary_fidelity"] < 0.8


# ============================================================================
# Pass Bar Tests
# ============================================================================

def test_pass_bar_entity_and_summary():
    """Pass requires avg_entity_extraction >= ENTITY_PASS_BAR AND avg_summary_fidelity >= SUMMARY_PASS_BAR."""
    truths = _make_env(n=2)
    submitted = []
    for truth in truths:
        entities = task4_knowledge._extract_ground_truth_entities(truth)
        submitted.append({"entities": entities, "summary": _ref_summary(truth)})
    
    result = task4_knowledge.grade(submitted, truths)
    bd = result["breakdown"]
    
    expected_passed = (
        bd["avg_entity_extraction"] >= task4_knowledge.ENTITY_PASS_BAR
        and bd["avg_summary_fidelity"] >= task4_knowledge.SUMMARY_PASS_BAR
    )
    assert result["passed"] == expected_passed


def test_pass_fails_with_low_entity_extraction():
    """Low entity extraction should fail pass bar."""
    truths = _make_env(n=2)
    # Submit only partial entities
    submitted = []
    for truth in truths:
        # Only include first entity (if any)
        full_entities = task4_knowledge._extract_ground_truth_entities(truth)
        entities = full_entities[:1] if full_entities else []
        submitted.append({"entities": entities, "summary": _ref_summary(truth)})
    
    result = task4_knowledge.grade(submitted, truths)
    # If entity extraction is below threshold, should not pass
    if result["breakdown"]["avg_entity_extraction"] < 0.75:
        assert not result["passed"]


def test_pass_fails_with_low_summary_fidelity():
    """Low summary fidelity should fail pass bar."""
    truths = _make_env(n=2)
    submitted = []
    for truth in truths:
        entities = task4_knowledge._extract_ground_truth_entities(truth)
        # Use poor summary
        submitted.append({"entities": entities, "summary": "Poor summary."})
    
    result = task4_knowledge.grade(submitted, truths)
    # If summary fidelity is below threshold, should not pass
    if result["breakdown"]["avg_summary_fidelity"] < task4_knowledge.SUMMARY_PASS_BAR:
        assert not result["passed"]


def test_pass_bar_values_documented():
    """Pass bar thresholds should be documented in breakdown."""
    truths = _make_env(n=2)
    submitted = [{"entities": [], "summary": ""}] * len(truths)
    result = task4_knowledge.grade(submitted, truths)
    
    assert "entity_pass_bar" in result["breakdown"]
    assert "summary_pass_bar" in result["breakdown"]
    assert result["breakdown"]["entity_pass_bar"] == task4_knowledge.ENTITY_PASS_BAR
    assert result["breakdown"]["summary_pass_bar"] == task4_knowledge.SUMMARY_PASS_BAR


# ============================================================================
# Edge Cases: Perfect, Empty, Partial Extractions
# ============================================================================

def test_perfect_extraction_all_entities():
    """Perfect extraction of all entities should score 1.0 on entity score."""
    truths = _make_env(n=2)
    submitted = []
    for truth in truths:
        entities = task4_knowledge._extract_ground_truth_entities(truth)
        submitted.append({"entities": entities, "summary": _ref_summary(truth)})
    
    result = task4_knowledge.grade(submitted, truths)
    # Perfect entity extraction
    assert result["breakdown"]["avg_entity_extraction"] == 1.0


def test_empty_entities_and_summary():
    """Empty entities and summary should score very low."""
    truths = _make_env(n=2)
    submitted = [{"entities": [], "summary": ""}] * len(truths)
    result = task4_knowledge.grade(submitted, truths)
    assert result["score"] < 0.2


def test_partial_icd_match():
    """Matching some but not all ICD codes should give partial credit."""
    truths = _make_env(n=2)
    submitted = []
    for truth in truths:
        # Only include first ICD code
        entities = []
        if truth.icd10_codes:
            code = truth.icd10_codes[0]
            entities.append({"text": code, "type": "Condition", "code": code})
        submitted.append({"entities": entities, "summary": _ref_summary(truth)})
    
    result = task4_knowledge.grade(submitted, truths)
    # Should have some entity credit but not full
    assert 0.0 < result["breakdown"]["avg_entity_extraction"] < 1.0


def test_case_insensitive_entity_matching():
    """Entity matching should handle case differences."""
    truths = _make_env(n=2)
    submitted = []
    for truth in truths:
        entities = []
        # Use lowercase codes
        for code in truth.icd10_codes:
            entities.append({"text": code.lower(), "type": "Condition", "code": code.lower()})
        for med in truth.medications:
            entities.append({"text": med.name.lower(), "type": "Medication", "code": med.name.lower()})
        submitted.append({"entities": entities, "summary": _ref_summary(truth)})
    
    result = task4_knowledge.grade(submitted, truths)
    # Case-insensitive matching should work
    assert result["breakdown"]["avg_entity_extraction"] > 0.0


# ============================================================================
# Handling Malformed or Unexpected Input
# ============================================================================

def test_malformed_entities_missing_fields():
    """Entities with missing fields should be handled gracefully."""
    truths = _make_env(n=2)
    submitted = []
    for truth in truths:
        # Malformed entities missing some fields
        entities = [
            {"text": "SomeText"},  # Missing type and code
            {"type": "Condition"},  # Missing text and code
            {"code": "I10"},  # Missing text and type
        ]
        submitted.append({"entities": entities, "summary": _ref_summary(truth)})
    
    # Should not crash
    result = task4_knowledge.grade(submitted, truths)
    assert 0.0 <= result["score"] <= 1.0


def test_none_entities_list():
    """None entities list should be handled."""
    truths = _make_env(n=2)
    submitted = [{"entities": None, "summary": "Some summary"}] * len(truths)
    
    # Should not crash
    result = task4_knowledge.grade(submitted, truths)
    assert 0.0 <= result["score"] <= 1.0


def test_missing_summary_key():
    """Missing summary key should be handled."""
    truths = _make_env(n=2)
    submitted = [{"entities": []}] * len(truths)  # No "summary" key
    
    # Should not crash
    result = task4_knowledge.grade(submitted, truths)
    assert 0.0 <= result["score"] <= 1.0


def test_extra_fields_ignored():
    """Extra fields in submission should be ignored."""
    truths = _make_env(n=2)
    submitted = []
    for truth in truths:
        entities = task4_knowledge._extract_ground_truth_entities(truth)
        submitted.append({
            "entities": entities,
            "summary": _ref_summary(truth),
            "extra_field": "should be ignored",
            "another_extra": 123,
        })
    
    result = task4_knowledge.grade(submitted, truths)
    # Should work normally ignoring extra fields
    assert result["score"] >= task4_knowledge.PASS_BAR


def test_fewer_submissions_than_truths():
    """Fewer submissions than ground truths should be handled."""
    truths = _make_env(n=4)
    # Only submit 2 records
    submitted = []
    for truth in truths[:2]:
        entities = task4_knowledge._extract_ground_truth_entities(truth)
        submitted.append({"entities": entities, "summary": _ref_summary(truth)})
    
    result = task4_knowledge.grade(submitted, truths)
    # Should handle mismatch gracefully
    assert 0.0 <= result["score"] <= 1.0


def test_empty_ground_truth():
    """Empty ground truth should return zero score."""
    result = task4_knowledge.grade([{"entities": [], "summary": "test"}], [])
    assert result["score"] == 0.0
    assert not result["passed"]


# ============================================================================
# Scoring Formula Verification
# ============================================================================

def test_score_weights_correct():
    """Verify final_score = entity*0.4 + code*0.3 + summary*0.3."""
    truths = _make_env(n=2)
    submitted = []
    for truth in truths:
        entities = task4_knowledge._extract_ground_truth_entities(truth)
        submitted.append({"entities": entities, "summary": _ref_summary(truth)})
    
    result = task4_knowledge.grade(submitted, truths)
    bd = result["breakdown"]
    
    expected_score = (
        bd["avg_entity_extraction"] * 0.4 +
        bd["avg_code_precision"] * 0.3 +
        bd["avg_summary_fidelity"] * 0.3
    )
    
    assert abs(result["score"] - expected_score) < 0.01


def test_per_record_breakdown():
    """Per-record breakdown should be present and correct."""
    truths = _make_env(n=3)
    submitted = []
    for truth in truths:
        entities = task4_knowledge._extract_ground_truth_entities(truth)
        submitted.append({"entities": entities, "summary": _ref_summary(truth)})
    
    result = task4_knowledge.grade(submitted, truths)
    assert "per_record" in result
    assert len(result["per_record"]) == 3
    
    for per_rec in result["per_record"]:
        assert "record_id" in per_rec
        assert "entity_score" in per_rec
        assert "code_score" in per_rec
        assert "summary_score" in per_rec
