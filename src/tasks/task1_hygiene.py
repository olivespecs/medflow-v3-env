"""
Task 1 Grader — Data Hygiene & Standardisation.

score = correct_fields / total_fields  →  0.0–1.0
Pass bar: score >= 0.85
"""

from __future__ import annotations

from typing import Any

from ..models import PatientRecord
from ..utils import is_valid_icd10, normalize_date, align_submitted_to_truth
from .common import normalize_record_list, validate_length_or_error


# Fields evaluated (order preserved for stable breakdown dicts)
GRADED_FIELDS = [
    "dob",
    "gender",
    "icd10_codes",
    "vitals",
    "medications",
    "phone",
    "email",
    "address",
]

PASS_BAR = 0.85


def _score_dob(submitted: str | None, truth: str) -> float:
    """1.0 if submitted normalises to the same ISO date as truth."""
    if not submitted:
        return 0.0
    normalised = normalize_date(submitted)
    return 1.0 if normalised == truth else 0.0


def _score_text_field(submitted: str | None, truth: str | None) -> float:
    """Score text fields with character-level tolerance for OCR fixes."""
    if truth is None:
        return 1.0 if submitted is None else 0.0
    if submitted is None:
        return 0.0
    
    s_clean = str(submitted).strip().lower()
    t_clean = str(truth).strip().lower()
    
    if s_clean == t_clean:
        return 1.0
    
    # Simple character-level similarity for OCR correction credit
    # If the agent fixed "Jchn" to "John", we give credit
    from difflib import SequenceMatcher
    ratio = SequenceMatcher(None, s_clean, t_clean).ratio()
    return 1.0 if ratio > 0.8 else 0.0


def _score_icd(submitted: list[str], truth: list[str]) -> float:
    """
    Partial credit per code:
    - Submitted codes must be valid ICD-10 to count
    - Score = overlap(valid_submitted, truth) / max(|truth|, |valid_submitted|)
    - All-invalid submissions score 0.0 — no free credit for garbage output
    """
    if not truth:
        return 1.0 if not submitted else 0.0

    # Normalize truth and submitted; invalid codes are silently excluded
    truth_norm = {c.strip().upper() for c in truth}
    sub_norm = {c.strip().upper() for c in submitted if is_valid_icd10(c)}

    # sub_norm empty → no valid codes submitted → 0 overlap → 0.0 (correct)
    overlap = len(sub_norm & truth_norm)
    return overlap / max(len(truth_norm), len(sub_norm), 1)


def _score_vitals(submitted: dict[str, Any], truth: dict[str, Any]) -> float:
    """Field-level partial credit on vitals sub-dict."""
    keys = [k for k in truth if truth[k] is not None]
    if not keys:
        return 1.0
    correct = sum(
        1 for k in keys
        if submitted.get(k) is not None
        and abs(float(submitted.get(k, 0)) - float(truth[k])) < 0.5
    )
    return correct / len(keys)


def _score_medications(submitted: list[dict], truth: list[dict]) -> float:
    """Partial credit: match by medication name + approximate dose."""
    if not truth:
        return 1.0 if not submitted else 0.5
    truth_names = {m["name"].lower() for m in truth}
    sub_names = {m["name"].lower() for m in submitted}
    name_overlap = len(truth_names & sub_names) / max(len(truth_names), 1)
    return name_overlap


def _check_longitudinal_consistency(submitted_records: list[dict[str, Any]]) -> float:
    """
    Check if fields that should be constant for a patient (MRN -> DOB, Gender, Name)
    are actually consistent across all records with the same MRN.
    """
    mrn_map: dict[str, dict[str, Any]] = {}
    total_checks = 0
    consistent_checks = 0
    
    for rec in submitted_records:
        mrn = rec.get("mrn")
        if not mrn:
            # Missing MRN prevents longitudinal linkage and is treated as a consistency failure.
            total_checks += 1
            continue
        
        if mrn not in mrn_map:
            mrn_map[mrn] = {
                "dob": rec.get("dob"),
                "gender": rec.get("gender"),
                "patient_name": rec.get("patient_name")
            }
        else:
            base = mrn_map[mrn]
            # Check DOB
            total_checks += 1
            if normalize_date(rec.get("dob")) == normalize_date(base["dob"]):
                consistent_checks += 1
            
            # Check Gender
            total_checks += 1
            if rec.get("gender") == base["gender"]:
                consistent_checks += 1
                
            # Check Name
            total_checks += 1
            if rec.get("patient_name") == base["patient_name"]:
                consistent_checks += 1
                
    if total_checks == 0:
        return 1.0
    return consistent_checks / total_checks


def grade_record(submitted: dict[str, Any], truth: PatientRecord) -> dict[str, float]:
    """Return a per-field score dict for a single record."""
    truth_d = truth.model_dump()
    scores: dict[str, float] = {}

    scores["dob"] = _score_dob(submitted.get("dob"), truth_d["dob"])
    scores["gender"] = 1.0 if submitted.get("gender") == truth_d["gender"] else 0.0
    scores["icd10_codes"] = _score_icd(
        submitted.get("icd10_codes", []), truth_d["icd10_codes"]
    )
    scores["vitals"] = _score_vitals(
        submitted.get("vitals") or {}, truth_d.get("vitals") or {}
    )
    scores["medications"] = _score_medications(
        submitted.get("medications") or [], truth_d.get("medications") or []
    )
    scores["phone"] = _score_text_field(submitted.get("phone"), truth_d.get("phone"))
    scores["email"] = _score_text_field(submitted.get("email"), truth_d.get("email"))
    scores["address"] = _score_text_field(submitted.get("address"), truth_d.get("address"))

    return scores


def grade(
    submitted_records: list[dict[str, Any]],
    ground_truth: list[PatientRecord],
) -> dict[str, Any]:
    """
    Grade a full batch of submitted records against ground truth.

    Returns
    -------
    {
        "score": float,          # episode score 0.0–1.0
        "breakdown": {...},      # per-record, per-field scores
        "passed": bool,
        "info": {...},
    }
    """
    if not ground_truth:
        return {"score": 0.0001, "breakdown": {}, "passed": False, "info": {"error": "No ground truth"}}

    submitted_records = normalize_record_list(submitted_records)
    ok_len, err = validate_length_or_error(submitted_records, len(ground_truth), "task1_hygiene")
    if not ok_len:
        return err

    # [P4] Align submitted records to truth order by record_id
    truth_ids = [t.record_id for t in ground_truth]
    submitted_records = align_submitted_to_truth(submitted_records, truth_ids)

    per_record: list[dict] = []
    all_field_scores: list[float] = []

    for i, truth in enumerate(ground_truth):
        submitted = submitted_records[i]
        field_scores = grade_record(submitted, truth)
        record_score = sum(field_scores.values()) / len(field_scores)
        all_field_scores.extend(field_scores.values())
        per_record.append({
            "record_id": truth.record_id,
            "record_score": round(record_score, 4),
            "field_scores": {k: round(v, 4) for k, v in field_scores.items()},
        })

    if not all_field_scores:
        return {"score": 0.0001, "breakdown": {"per_record": per_record}, "passed": False, "info": {"pass_bar": PASS_BAR, "total_fields_evaluated": 0}}

    # Calculate average per-field scores
    avg_per_field = {
        field: sum(rec["field_scores"][field] for rec in per_record) / len(per_record)
        for field in GRADED_FIELDS
    }

    # Add longitudinal consistency score
    consistency_score = _check_longitudinal_consistency(submitted_records)

    # Weight per-field and consistency: 80% per-field, 20% consistency
    field_avg = sum(avg_per_field.values()) / len(GRADED_FIELDS)
    final_score = (field_avg * 0.8) + (consistency_score * 0.2)

    passed = final_score >= PASS_BAR

    # Ensure score is strictly in (0, 1) - validation requirement
    clamped_score = max(0.0001, min(0.9999, final_score))

    return {
        "score": round(clamped_score, 4),
        "passed": passed,
        "breakdown": {
            "per_record": per_record,
            **{f"avg_{k}": round(v, 4) for k, v in avg_per_field.items()},
            "longitudinal_consistency": round(consistency_score, 4),
            "pass_bar": PASS_BAR,
        },
        "info": {
            "pass_bar": PASS_BAR,
            "fields_graded": GRADED_FIELDS,
        }
    }
