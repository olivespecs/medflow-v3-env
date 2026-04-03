"""
Task 4 Grader — Clinical Knowledge Extraction & Summarization.

Reference summary is a structured abstract generated from the record's
structured fields (ICD-10 codes, medications, vitals), NOT the raw
clinical_notes. This prevents the baseline from achieving a perfect
score simply by copying the notes verbatim.
"""

from __future__ import annotations

from typing import Any
import re
from ..models import PatientRecord
from ..utils import scan_phi, semantic_similarity_score
from .common import normalize_record_list, validate_length_or_error

# Weights
ENTITY_WEIGHT = 0.4
CODE_WEIGHT = 0.3
SUMMARY_WEIGHT = 0.3

ENTITY_PASS_BAR = 0.75
SUMMARY_PASS_BAR = 0.50  # Must match environment.py TASK_DESCRIPTIONS, README, and openenv.yaml
SUMMARY_PHI_PENALTY_PER_MATCH = 0.1
SUMMARY_PHI_PENALTY_CAP = 0.6

# Deprecated: use ENTITY_PASS_BAR and SUMMARY_PASS_BAR for specific checks
# Maintained for backward compatibility with existing tests
PASS_BAR = ENTITY_PASS_BAR

def _extract_ground_truth_entities(truth: PatientRecord) -> list[dict[str, str]]:
    """
    Extract standardized entities from the PatientRecord (ICD-10 codes and Medications).
    """
    entities = []
    
    # Conditions from ICD-10
    for code in truth.icd10_codes:
        entities.append({
            "text": code,
            "type": "Condition",
            "code": code
        })
        
    # Medications
    for med in truth.medications:
        entities.append({
            "text": med.name,
            "type": "Medication",
            "code": med.name
        })
        
    return entities


def _build_reference_summary(truth: PatientRecord) -> str:
    """Structured abstract for fidelity scoring (avoids raw-note leakage)."""

    parts: list[str] = []

    # Conditions (ICD-10)
    if truth.icd10_codes:
        parts.append("Conditions: " + ", ".join(sorted(truth.icd10_codes)))

    # Medications with dose/frequency when available
    if truth.medications:
        meds_fmt: list[str] = []
        for med in truth.medications:
            detail = med.name
            dose = f" {med.dose_mg}mg" if med.dose_mg is not None else ""
            freq = f" @ {med.frequency}" if med.frequency else ""
            meds_fmt.append(detail + dose + freq)
        parts.append("Medications: " + ", ".join(meds_fmt))

    # Vitals (include only available fields to keep concise)
    vitals = truth.vitals
    vitals_parts: list[str] = []
    if vitals.heart_rate_bpm is not None:
        vitals_parts.append(f"HR {vitals.heart_rate_bpm} bpm")
    if vitals.systolic_bp_mmhg is not None and vitals.diastolic_bp_mmhg is not None:
        vitals_parts.append(f"BP {vitals.systolic_bp_mmhg}/{vitals.diastolic_bp_mmhg} mmHg")
    if vitals.temperature_c is not None:
        vitals_parts.append(f"Temp {vitals.temperature_c} C")
    if vitals.weight_kg is not None:
        vitals_parts.append(f"Weight {vitals.weight_kg} kg")
    if vitals.height_cm is not None:
        vitals_parts.append(f"Height {vitals.height_cm} cm")
    if vitals_parts:
        parts.append("Vitals: " + ", ".join(vitals_parts))

    if not parts:
        return "No structured data available."

    return "; ".join(parts)

def _score_entity_text(submitted: list[dict[str, Any]] | None, truth: list[dict[str, Any]]) -> float:
    """Partial credit: correct type+code even if text differs."""
    # Handle None or non-list submitted safely (LLM hallucination protection)
    if not isinstance(submitted, list):
        submitted = []
    if not truth:
        return 1.0 if not submitted else 0.0
    truth_codes = {f"{e['type']}|{e['code']}".lower() for e in truth}
    # Safely extract codes, skipping malformed entities
    sub_codes = set()
    for e in submitted:
        if isinstance(e, dict) and 'type' in e and 'code' in e:
            sub_codes.add(f"{e['type']}|{e['code']}".lower())
    return len(truth_codes & sub_codes) / max(len(truth_codes), len(sub_codes), 1)

def _score_code_precision(submitted: list[dict[str, Any]] | None, truth: list[dict[str, Any]]) -> float:
    """Did the agent get the exact ICD-10 codes right?"""
    # Handle None or non-list submitted safely (LLM hallucination protection)
    if not isinstance(submitted, list):
        submitted = []
    truth_codes = {e['code'].upper() for e in truth if e.get('type') == 'Condition'}
    # Safely extract codes, skipping malformed entities
    sub_codes = set()
    for e in submitted:
        if isinstance(e, dict) and e.get('type') == 'Condition' and e.get('code'):
            sub_codes.add(e['code'].upper())
    if not truth_codes:
        return 1.0
    return len(truth_codes & sub_codes) / len(truth_codes)

def grade(
    submitted_knowledge: list[dict[str, Any]],
    ground_truth: list[PatientRecord]
) -> dict[str, Any]:
    """
    Grade clinical knowledge extraction and summarization.

    Alignment: submitted_knowledge[i] is scored against ground_truth[i] (index / list
    order only; knowledge dicts do not carry record_id).
    
    Robustness: Handles submissions with or without record_id field — uses index alignment
    regardless. Extra fields in submission are ignored.
    """
    if not ground_truth:
        return {"score": 0.0, "passed": False, "info": {"error": "No ground truth"}}

    submitted_knowledge = normalize_record_list(submitted_knowledge)
    ok_len, err = validate_length_or_error(submitted_knowledge, len(ground_truth), "task4_knowledge")
    if not ok_len:
        return err
        
    per_record = []
    entity_scores = []
    code_scores = []
    summary_scores = []
    total_phi_leaks = 0
    
    for i, truth in enumerate(ground_truth):
        # Handle both formats: with or without record_id
        sub = submitted_knowledge[i] if i < len(submitted_knowledge) else {"entities": [], "summary": ""}
        # Note: record_id field is safely ignored if present in submission
        
        # Ground truth entities
        truth_entities = _extract_ground_truth_entities(truth)
        
        # Score entities and codes
        e_score = _score_entity_text(sub.get("entities", []), truth_entities)
        c_score = _score_code_precision(sub.get("entities", []), truth_entities)
        entity_scores.append(e_score)
        code_scores.append(c_score)
        
        # Score summary against a structured reference abstract (NOT raw clinical_notes).
        # This prevents the circularity where copying notes yields a perfect score.
        reference_summary = _build_reference_summary(truth)
        summary_text = sub.get("summary", "")
        s_score = semantic_similarity_score(reference_summary, summary_text)

        # Penalize PHI leaks in summaries to prevent copy-paste exfiltration.
        phi_found = scan_phi(summary_text)
        phi_leak_count = sum(len(matches) for matches in phi_found.values())
        total_phi_leaks += phi_leak_count
        if phi_leak_count > 0:
            phi_penalty = min(
                SUMMARY_PHI_PENALTY_CAP,
                phi_leak_count * SUMMARY_PHI_PENALTY_PER_MATCH,
            )
            s_score = max(0.0, s_score - phi_penalty)

        summary_scores.append(s_score)
        
        per_record.append({
            "record_id": truth.record_id,
            "entity_score": round(e_score, 4),
            "code_score": round(c_score, 4),
            "summary_score": round(s_score, 4),
            "phi_leak_count": phi_leak_count,
            "phi_categories": sorted(phi_found.keys()),
        })
        
    avg_e = sum(entity_scores) / len(entity_scores)
    avg_c = sum(code_scores) / len(code_scores)
    avg_s = sum(summary_scores) / len(summary_scores)
    
    final_score = (avg_e * ENTITY_WEIGHT) + (avg_c * CODE_WEIGHT) + (avg_s * SUMMARY_WEIGHT)
    passed = (avg_e >= ENTITY_PASS_BAR) and (avg_s >= SUMMARY_PASS_BAR)
    
    return {
        "score": round(final_score, 4),
        "passed": passed,
        "per_record": per_record,  # top-level for environment.py and direct consumers
        "breakdown": {
            "avg_entity_extraction": round(avg_e, 4),
            "avg_code_precision": round(avg_c, 4),
            "avg_summary_fidelity": round(avg_s, 4),
            "summary_phi_leaks": total_phi_leaks,
            "entity_pass_bar": ENTITY_PASS_BAR,
            "summary_pass_bar": SUMMARY_PASS_BAR,
            "per_record": per_record,  # also kept in breakdown for backward compat
        },
        "info": {
            "pass_conditions": "avg_entity_extraction >= 0.75 AND avg_summary_fidelity >= 0.50"
        }
    }
