"""
Task 3 Grader — Full Anonymisation + Downstream Utility.

final_score = (avg_phi * 0.4) + (avg_ml_utility * 0.3) + (avg_fid * 0.2) + (k_score * 0.1)
Pass bar:  phi_score == 1.0  AND  ml_utility_score >= 0.60

ml_utility_score measures risk-score *fidelity*: how close the anonymised
record's predicted disease-risk is to the original ground-truth record's
predicted risk (closeness to baseline), NOT the raw risk magnitude.
A score of 1.0 means the anonymisation preserved all risk-relevant signals;
0.0 means the risk prediction diverged by ±0.5 or more from baseline.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ..models import AnnotatedRecord
from ..tasks.task2_redaction import _check_phi_redacted, _flatten_record, _strip_metadata
from ..utils import clinical_utility_score, align_submitted_to_truth
from .common import normalize_record_list, validate_length_or_error

logger = logging.getLogger(__name__)


_PHI_W = 0.4
_ML_W = 0.3
_FIDELITY_W = 0.2
_K_W = 0.1
ADVERSARIAL_PRIVACY_BLEND = 0.25
ADVERSARIAL_LEAK_PENALTY = 0.10

# Baseline mock-ML score that the anonymised output must beat
BASELINE_ML_SCORE = 0.60

# ─────────────────────────────────────────────────────────────────────────────
# Mock Disease-Risk Scorer
# ─────────────────────────────────────────────────────────────────────────────
# A fully deterministic, stateless scorer that estimates disease-risk probability
# from structured record fields.  No ML model is actually loaded; the logic is a
# hand-crafted decision function that is stable across Python versions.

AGE_RISK_THRESHOLDS = [
    (18, 40, 0.15),
    (40, 60, 0.40),
    (60, 75, 0.65),
    (75, 200, 0.85),
]

HIGH_RISK_ICD_PREFIXES = {"I", "E", "J", "N", "F"}  # cardiovascular, metabolic, …

HIGH_RISK_MEDS = {
    "insulin", "warfarin", "furosemide", "metformin", "lisinopril",
    "metoprolol", "atorvastatin",
}


def _mock_risk_score(record: dict[str, Any]) -> float:
    """
    Deterministic risk scorer.  Higher output = higher modelled disease risk.
    Inputs used: age-group token OR dob (for age), icd10_codes, medications.
    """
    score = 0.0

    # Age contribution
    age_group = record.get("age_group")          # pseudonymised field
    dob = record.get("dob", "")                  # may still be present
    age_score = _age_group_to_score(age_group, dob)
    score += age_score * 0.4

    # ICD-10 contribution
    icd_codes: list[str] = record.get("icd10_codes") or []
    high_risk_count = sum(
        1 for c in icd_codes
        if c and c[0].upper() in HIGH_RISK_ICD_PREFIXES
    )
    icd_score = min(1.0, high_risk_count / max(len(icd_codes), 1))
    score += icd_score * 0.35

    # Medication contribution
    meds: list[dict] = record.get("medications") or []
    med_names = {(m.get("name") or "").lower() for m in meds}
    high_risk_med_count = len(med_names & HIGH_RISK_MEDS)
    med_score = min(1.0, high_risk_med_count / max(len(meds), 1))
    score += med_score * 0.25

    return round(min(1.0, score), 4)


def _age_group_to_score(age_group: str | None, dob: str | None) -> float:
    """Convert an age_group token or DOB to a risk score."""
    if age_group:
        mapping = {"18-40": 0.15, "41-60": 0.40, "61-75": 0.65, "76+": 0.85}
        return mapping.get(age_group, 0.30)
    if dob:
        try:
            from datetime import date, datetime
            birth = datetime.strptime(dob, "%Y-%m-%d").date()
            age = (date.today() - birth).days / 365.25
            for lo, hi, risk in AGE_RISK_THRESHOLDS:
                if lo <= age < hi:
                    return risk
        except Exception:
            pass
    return 0.30   # default when age unknown


def compute_baseline_ml_scores(ground_truth_records: list[AnnotatedRecord]) -> list[float]:
    """
    Run the mock scorer on the *clean* ground-truth records to establish
    a per-record baseline that anonymised output must at least match.
    """
    return [_mock_risk_score(r.model_dump()) for r in ground_truth_records]


# ─────────────────────────────────────────────────────────────────────────────
# Fidelity scorer
# ─────────────────────────────────────────────────────────────────────────────

def _age_group_from_dob(dob: str) -> str:
    """Map ISO dob string to age-group bucket."""
    try:
        from datetime import date, datetime
        birth = datetime.strptime(dob, "%Y-%m-%d").date()
        age = (date.today() - birth).days / 365.25
        if age < 40:
            return "18-40"
        if age < 60:
            return "41-60"
        if age < 75:
            return "61-75"
        return "76+"
    except Exception:
        return "unknown"


def _fidelity_score_record(submitted: dict[str, Any], truth: AnnotatedRecord) -> float:
    """
    Score clinical fidelity of one pseudonymised record.
    Checks: age-group, gender, ICD-10 prefix category, medication presence.
    """
    sub_scores: list[float] = []

    # 1. Age-group preserved (accept DOB → age-group mapping)
    truth_age_group = _age_group_from_dob(truth.dob)
    sub_age_group = submitted.get("age_group") or _age_group_from_dob(submitted.get("dob", ""))
    sub_scores.append(1.0 if sub_age_group == truth_age_group else 0.0)

    # 2. Gender preserved (or generalised as "Other" — still partial credit)
    truth_gender = truth.gender
    sub_gender = submitted.get("gender", "")
    if sub_gender == truth_gender:
        sub_scores.append(1.0)
    elif sub_gender in ("Other", "Unknown", ""):
        sub_scores.append(0.5)
    else:
        sub_scores.append(0.0)

    # 3. ICD-10 chapter prefix preserved
    truth_prefixes = {c[0] for c in truth.icd10_codes if c}
    sub_codes: list[str] = submitted.get("icd10_codes") or []
    sub_prefixes = {c[0] for c in sub_codes if c}
    if truth_prefixes:
        prefix_overlap = len(truth_prefixes & sub_prefixes) / len(truth_prefixes)
    else:
        prefix_overlap = 1.0
    sub_scores.append(prefix_overlap)

    # 4. Clinical keywords still present in notes
    # Harden against None/missing notes
    sub_notes = submitted.get("clinical_notes") or ""
    utility = clinical_utility_score(truth.clinical_keywords, sub_notes)
    sub_scores.append(utility)

    return sum(sub_scores) / len(sub_scores)


def _check_adversarial_identifiers(
    submitted_record: dict[str, Any],
    adversarial_identifiers: list[str],
) -> tuple[float, list[str]]:
    """Detect leaked indirect identifiers that enable linkage attacks."""
    if not adversarial_identifiers:
        return 1.0, []

    clean_record = _strip_metadata(submitted_record)
    submitted_flat = _flatten_record(clean_record).lower()
    leaked: list[str] = []
    for marker in adversarial_identifiers:
        if not marker:
            continue

        marker_l = marker.lower()

        # Exact leakage (most common case)
        if marker_l in submitted_flat:
            leaked.append(marker)
            continue

        # Robust check 1: disease + ZIP linkage survives paraphrasing
        dz_match = re.search(r"the\s+(.+?)\s+patient\s+from\s+(\d{5})", marker_l)
        if dz_match:
            disease = dz_match.group(1).strip()
            zip_code = dz_match.group(2).strip()
            if disease and zip_code and disease in submitted_flat and re.search(rf"\b{re.escape(zip_code)}\b", submitted_flat):
                leaked.append(marker)
                continue

        # Robust check 2: family-linkage cue with same surname
        fam_match = re.search(r"sibling\s+of\s+([a-z\-']+)", marker_l)
        if fam_match:
            surname = fam_match.group(1)
            if re.search(rf"\b(sibling|brother|sister|relative|family\s+member)\s+of\s+{re.escape(surname)}\b", submitted_flat):
                leaked.append(marker)

    base_score = max(0.0, (len(adversarial_identifiers) - len(leaked)) / len(adversarial_identifiers))
    penalty = ADVERSARIAL_LEAK_PENALTY * len(leaked)
    return max(0.0, base_score - penalty), leaked


# ─────────────────────────────────────────────────────────────────────────────
# Main grader
# ─────────────────────────────────────────────────────────────────────────────

def grade(
    submitted_records: list[dict[str, Any]],
    annotated_ground_truth: list[AnnotatedRecord],
    baseline_ml_scores: list[float] | None = None,
) -> dict[str, Any]:
    """
    Grade a full anonymisation submission.

    Returns
    -------
    {
        "score": float,
        "breakdown": {...},
        "passed": bool,
        "info": {...},
    }
    """
    if not annotated_ground_truth:
        return {"score": 0.0001, "breakdown": {}, "passed": False, "info": {"error": "No ground truth"}}

    submitted_records = normalize_record_list(submitted_records)
    ok_len, err = validate_length_or_error(submitted_records, len(annotated_ground_truth), "task3_anonymization")
    if not ok_len:
        return err

    # [P4] Align submitted records to truth order by record_id
    truth_ids = [t.record_id for t in annotated_ground_truth]
    submitted_records = align_submitted_to_truth(submitted_records, truth_ids)

    if baseline_ml_scores is None:
        baseline_ml_scores = compute_baseline_ml_scores(annotated_ground_truth)

    phi_scores: list[float] = []
    adversarial_privacy_scores: list[float] = []
    fidelity_scores: list[float] = []
    ml_scores: list[float] = []
    per_record: list[dict] = []

    for i, truth in enumerate(annotated_ground_truth):
        submitted = submitted_records[i]

        # PHI
        base_phi_s, leaked = _check_phi_redacted(submitted, truth.phi_tokens)
        adv_priv_s, adversarial_leaks = _check_adversarial_identifiers(
            submitted,
            truth.adversarial_identifiers,
        )
        phi_s = max(
            0.0,
            (1.0 - ADVERSARIAL_PRIVACY_BLEND) * base_phi_s
            + ADVERSARIAL_PRIVACY_BLEND * adv_priv_s,
        )
        phi_scores.append(phi_s)
        adversarial_privacy_scores.append(adv_priv_s)

        # Fidelity
        fid_s = _fidelity_score_record(submitted, truth)
        fidelity_scores.append(fid_s)
        
        # Experimental metrics (safe — never crash grading)
        try:
            from ..utils import semantic_similarity_score, redaction_robustness_score
            semantic_sim = semantic_similarity_score(truth.clinical_notes, submitted.get("clinical_notes", ""))
            robustness = redaction_robustness_score(submitted.get("clinical_notes", ""))
        except Exception as _exp_err:
            logger.debug("Experimental metrics failed (non-fatal, record %s): %s", truth.record_id, _exp_err)
            semantic_sim, robustness = 0.5, 1.0

        # ML utility score: measures how well the anonymised record preserves
        # the disease-risk signal compared to the original ground-truth record.
        # This is a *fidelity* metric (closeness to baseline), NOT a raw risk score.
        sub_ml = _mock_risk_score(submitted)
        base_ml = baseline_ml_scores[i] if i < len(baseline_ml_scores) else BASELINE_ML_SCORE
        diff = abs(sub_ml - base_ml)
        ml_s = max(0.0, 1.0 - diff / 0.5)  # 1.0 = identical risk, 0.0 = diverged by >=0.5
        ml_scores.append(ml_s)

        per_record.append({
            "record_id": truth.record_id,
            "phi_score": round(phi_s, 4),
            "phi_direct_score": round(base_phi_s, 4),
            "adversarial_privacy_score": round(adv_priv_s, 4),
            "fidelity_score": round(fid_s, 4),
            "semantic_similarity_exp": round(semantic_sim, 4),
            "redaction_robustness_exp": round(robustness, 4),
            "ml_utility_score": round(ml_s, 4),
            "leaked_categories": leaked,
            "adversarial_leaks": adversarial_leaks,
            "submitted_risk": round(sub_ml, 4),
            "baseline_risk": round(base_ml, 4),
        })

    # k-anonymity check on quasi-identifiers
    from ..utils import check_k_anonymity
    quasi_ids = ["age_group", "gender", "address"]
    k_score = check_k_anonymity(submitted_records, quasi_ids, k=2)

    # Average scores
    avg_phi = sum(phi_scores) / len(phi_scores)
    avg_fid = sum(fidelity_scores) / len(fidelity_scores)
    avg_ml = sum(ml_scores) / len(ml_scores)
    
    # Weight everything: PHI is critical, then ML utility, then Fidelity, then k-anonymity
    # final = PHI(0.4) + ML(0.3) + FID(0.2) + K-ANONYMITY(0.1)
    final_score = (
        (avg_phi * _PHI_W) + 
        (avg_ml * _ML_W) + 
        (avg_fid * _FIDELITY_W) + 
        (k_score * _K_W)
    )
    
    # PASS criteria
    # Must have perfect PHI removal AND meet baseline ML score
    passed = (avg_phi >= 1.0) and (avg_ml >= BASELINE_ML_SCORE)

    # Ensure score is strictly in (0, 1) - validation requirement
    clamped_score = max(0.0001, min(0.9999, final_score))

    return {
        "score": round(clamped_score, 4),
        "passed": passed,
        "breakdown": {
            "phi_score": round(avg_phi, 4),
            "ml_utility_score": round(avg_ml, 4),
            "fidelity_score": round(avg_fid, 4),
            "k_anonymity_score": round(k_score, 4),
            "per_record": per_record,
        },
        "info": {
            "phi_weight": _PHI_W,
            "ml_weight": _ML_W,
            "fidelity_weight": _FIDELITY_W,
            "k_weight": _K_W,
            "baseline_ml_score": BASELINE_ML_SCORE,
            "pass_conditions": "phi_score == 1.0 AND ml_utility_score >= 0.60 (fidelity to baseline risk)",
        },
    }
