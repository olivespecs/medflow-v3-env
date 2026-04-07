"""
Task 2 Grader — PHI Detection & Redaction.

final_score = phi_score × 0.6 + utility_score × 0.4
Pass bar:  phi_score == 1.0  AND  utility_score >= 0.8

Note: phi_score is computed per-record as (tokens_ok / total_tokens). A single
leaked PHI token in a 5-token record reduces phi_score from 1.0 to 0.8.  The
per-record leak counts are reported in the breakdown for diagnostics but do NOT
apply an additional flat penalty on top of the phi_score reduction.
"""

from __future__ import annotations

import logging
from typing import Any

from ..models import AnnotatedRecord, PHIToken
from ..utils import clinical_utility_score, align_submitted_to_truth
from .common import normalize_record_list, validate_length_or_error

logger = logging.getLogger(__name__)


PHI_WEIGHT = 0.6
UTILITY_WEIGHT = 0.4
UTILITY_PASS_BAR = 0.8


# Keys that are part of AnnotatedRecord ground truth — must NEVER appear in submissions
_GRADER_ONLY_KEYS = {
    "phi_tokens",
    "clinical_keywords",
    "adversarial_identifiers",
    "injected_flaws",
}


def _strip_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """Remove ground-truth-only keys from a submitted record before grading."""
    return {k: v for k, v in record.items() if k not in _GRADER_ONLY_KEYS}


def _flatten_value(value: Any) -> str:
    """Flatten an arbitrary value into text for field-scoped token checks."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return _flatten_record(value)
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(_flatten_record(item))
        return " ".join(parts)
    return ""


def _check_phi_redacted(
    submitted_record: dict[str, Any],
    phi_tokens: list[PHIToken],
) -> tuple[float, list[str]]:
    """
    For each annotated PHI token check whether its value has been replaced
    by its expected redaction_token in the submitted record.

    Returns (phi_score, list_of_leaked_token_categories).
    """
    if not phi_tokens:
        return 1.0, []

    leaked: list[str] = []
    missing: list[str] = []

    # Strip grader metadata before flattening so phi_tokens annotations
    # don't make it look like PHI is still present.
    clean_record = _strip_metadata(submitted_record)
    submitted_str = _flatten_record(clean_record).lower()

    for token in phi_tokens:
        # The raw PHI value must NOT appear anywhere in the submitted record.
        value = (token.value or "").lower()
        redaction_token = (token.redaction_token or "").lower()
        field_text = _flatten_value(clean_record.get(token.field, "")).lower()

        if value and value in submitted_str:
            leaked.append(token.category.value)
            continue

        if redaction_token:
            # For free-text notes we also accept a generic [REDACTED] marker,
            # because rule-based baselines may first scrub literal values before
            # category-specific pattern replacement.
            if token.field == "clinical_notes":
                has_any_redaction = (redaction_token in field_text) or ("[redacted]" in field_text)
            else:
                has_any_redaction = redaction_token in field_text

            if not has_any_redaction:
                # Not leaked globally but also not replaced in the originating field.
                missing.append(f"{token.category.value}:missing_redaction")

    total_issues = len(leaked) + len(missing)
    phi_score = max(0.0, (len(phi_tokens) - total_issues) / len(phi_tokens))
    return phi_score, leaked + missing


def _flatten_record(record: dict[str, Any]) -> str:
    """Recursively flatten all string values in a dict to a single string."""
    parts: list[str] = []
    for v in record.values():
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, dict):
            parts.append(_flatten_record(v))
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(_flatten_record(item))
    return " ".join(parts)


def grade(
    submitted_records: list[dict[str, Any]],
    annotated_ground_truth: list[AnnotatedRecord],
) -> dict[str, Any]:
    """
    Grade a batch of redacted records.

    Returns
    -------
    {
        "score": float,
        "breakdown": {
            "phi_score": float,
            "utility_score": float,
            "per_record": [...],
        },
        "passed": bool,
        "info": {...},
    }
    """
    if not annotated_ground_truth:
        return {"score": 0.0001, "breakdown": {}, "passed": False, "info": {"error": "No ground truth"}}

    submitted_records = normalize_record_list(submitted_records)
    ok_len, err = validate_length_or_error(submitted_records, len(annotated_ground_truth), "task2_redaction")
    if not ok_len:
        return err

    # [P4] Align submitted records to truth order by record_id
    truth_ids = [t.record_id for t in annotated_ground_truth]
    submitted_records = align_submitted_to_truth(submitted_records, truth_ids)

    total_phi: list[float] = []
    total_utility: list[float] = []
    total_leaked: list[str] = []
    per_record: list[dict] = []

    for i, truth in enumerate(annotated_ground_truth):
        submitted = submitted_records[i]

        phi_score, leaked = _check_phi_redacted(submitted, truth.phi_tokens)
        total_phi.append(phi_score)
        total_leaked.extend(leaked)

        # Build the submitted clinical notes for utility check
        submitted_text = submitted.get("clinical_notes", "")
        utility = clinical_utility_score(truth.clinical_keywords, submitted_text)
        total_utility.append(utility)
        
        # Experimental metrics (safe — never crash grading)
        try:
            from ..utils import semantic_similarity_score, redaction_robustness_score
            semantic_sim = semantic_similarity_score(truth.clinical_notes, submitted_text)
            robustness = redaction_robustness_score(submitted_text)
        except Exception as _exp_err:
            logger.debug("Experimental metrics failed (non-fatal, record %s): %s", truth.record_id, _exp_err)
            semantic_sim, robustness = 0.5, 1.0

        per_record.append({
            "record_id": truth.record_id,
            "phi_score": round(phi_score, 4),
            "utility_score": round(utility, 4),
            "semantic_similarity_exp": round(semantic_sim, 4),
            "redaction_robustness_exp": round(robustness, 4),
            "leaked_categories": leaked,
        })

    avg_phi = sum(total_phi) / len(total_phi)
    avg_utility = sum(total_utility) / len(total_utility)

    # Score is the straightforward weighted sum — no additional flat penalty.
    # Leaks are already penalised proportionally via the per-record phi_score.
    final_score = max(0.0001, min(0.9999, avg_phi * PHI_WEIGHT + avg_utility * UTILITY_WEIGHT))

    passed = (avg_phi >= 1.0) and (avg_utility >= UTILITY_PASS_BAR)

    return {
        "score": round(final_score, 4),
        "breakdown": {
            "phi_score": round(avg_phi, 4),
            "utility_score": round(avg_utility, 4),
            "per_record": per_record,
        },
        "passed": passed,
        "info": {
            "phi_weight": PHI_WEIGHT,
            "utility_weight": UTILITY_WEIGHT,
            "utility_pass_bar": UTILITY_PASS_BAR,
            "total_phi_tokens_evaluated": sum(len(t.phi_tokens) for t in annotated_ground_truth),
            "total_leaked": len(total_leaked),
            "pass_conditions": "phi_score == 1.0 AND utility_score >= 0.8",
        },
    }
