"""
Task 5 — Contextual PII Disambiguation (Hard)

Challenge: Agent must decide which mentions are PII based on context.
Example: "Dr. Smith saw Mr. Smith" — only "Mr. Smith" is patient PII, "Dr. Smith" is provider.

This task tests nuanced understanding of:
- Patient vs. provider identifiers
- Family member mentions (incidental vs. relevant)
- Contextual PHI redaction decisions
"""

from __future__ import annotations

from typing import Any
from random import Random

from ..models import AnnotatedRecord, PHICategory, PHIToken


def grade(
    submitted_records: list[dict[str, Any]],
    ground_truth: list[AnnotatedRecord],
) -> dict[str, Any]:
    """
    Grade contextual PII disambiguation.
    
    Alignment: submitted_records[i] is scored against ground_truth[i].
    
    Scoring:
    - patient_phi_score: Correct redaction of patient identifiers
    - provider_phi_score: Correct preservation of provider identifiers  
    - contextual_accuracy: Proper handling of ambiguous cases
    - overall_score: Weighted combination
    """
    if len(submitted_records) != len(ground_truth):
        return {
            "score": 0.0001,
            "breakdown": {
                "error": f"Expected {len(ground_truth)} records, got {len(submitted_records)}",
            },
            "passed": False,
        }

    # Validate payload upfront to avoid grader crashes and give actionable errors
    for i, submitted in enumerate(submitted_records):
        if not isinstance(submitted, dict):
            return {
                "score": 0.0001,
                "breakdown": {"error": f"Record {i} is not a JSON object"},
                "passed": False,
            }
        if "clinical_notes" not in submitted:
            return {
                "score": 0.0001,
                "breakdown": {"error": f"Record {i} missing 'clinical_notes'"},
                "passed": False,
            }
        if not isinstance(submitted.get("clinical_notes"), str):
            return {
                "score": 0.0001,
                "breakdown": {"error": f"Record {i} clinical_notes must be a string"},
                "passed": False,
            }
    
    patient_scores = []
    provider_scores = []
    contextual_scores = []
    
    for i, truth in enumerate(ground_truth):
        submitted = submitted_records[i] if i < len(submitted_records) else {}
        
        # Extract fields from phi_tokens list
        patient_phi_truth = {
            t.value for t in truth.phi_tokens
            if t.category == PHICategory.PATIENT_IDENTIFIER
        }
        provider_phi_truth = {
            t.value for t in truth.phi_tokens
            if t.category == PHICategory.PROVIDER_IDENTIFIER
        }
        ambiguous_truth = {
            t.value for t in truth.phi_tokens
            if t.category == PHICategory.AMBIGUOUS
        }
        
        submitted_text = submitted.get("clinical_notes", "")
        
        # Score patient identifier redaction
        patient_score = _score_patient_phi(patient_phi_truth, submitted_text)
        patient_scores.append(patient_score)
        
        # Score provider identifier preservation
        provider_score = _score_provider_phi(provider_phi_truth, submitted_text)
        provider_scores.append(provider_score)
        
        # Score contextual disambiguation
        contextual_score = _score_contextual_accuracy(
            ambiguous_truth, submitted_text, patient_phi_truth, provider_phi_truth
        )
        contextual_scores.append(contextual_score)
    
    avg_patient = sum(patient_scores) / len(patient_scores) if patient_scores else 0.0
    avg_provider = sum(provider_scores) / len(provider_scores) if provider_scores else 0.0
    avg_contextual = sum(contextual_scores) / len(contextual_scores) if contextual_scores else 0.0
    
    # Weighted overall score
    overall = (avg_patient * 0.5 + avg_provider * 0.3 + avg_contextual * 0.2)
    
    passed = overall >= 0.70 and avg_patient >= 0.80
    
    # Ensure score is strictly in (0, 1) - validation requirement
    clamped_score = max(0.0001, min(0.9999, overall))

    return {
        "score": round(clamped_score, 4),
        "breakdown": {
            "patient_phi_score": round(avg_patient, 4),
            "provider_phi_score": round(avg_provider, 4),
            "contextual_accuracy": round(avg_contextual, 4),
        },
        "passed": passed,
        "info": {
            "records_processed": len(submitted_records),
            "threshold": 0.70,
        },
    }


def _score_patient_phi(
    patient_identifiers: set[str],
    submitted_text: str,
) -> float:
    """Score patient identifier redaction."""
    if not patient_identifiers:
        return 1.0
    if not isinstance(submitted_text, str):
        return 0.0

    redacted_count = 0
    for identifier in patient_identifiers:
        if not _contains(submitted_text, identifier):
            # Identifier is fully absent — correctly redacted.
            redacted_count += 1
        else:
            # Identifier still present — only give credit if a redaction token
            # appears immediately adjacent (within 40 chars), meaning the agent
            # partially replaced it in-place rather than leaving it wholesale.
            if _looks_like_redacted(submitted_text, identifier):
                redacted_count += 1

    return redacted_count / len(patient_identifiers)


def _score_provider_phi(
    provider_identifiers: set[str],
    submitted_text: str,
) -> float:
    """Score provider identifier preservation (should NOT be redacted)."""
    if not provider_identifiers:
        return 1.0
    if not isinstance(submitted_text, str):
        return 0.0
    
    preserved_count = 0
    for identifier in provider_identifiers:
        present = _contains(submitted_text, identifier)
        if present and not _looks_like_redacted(submitted_text, identifier):
            preserved_count += 1
    
    return preserved_count / len(provider_identifiers)


def _score_contextual_accuracy(
    ambiguous_mentions: set[str],
    submitted_text: str,
    patient_ids: set[str],
    provider_ids: set[str],
) -> float:
    """
    Score proper handling of ambiguous mentions.
    
    Examples:
    - "Dr. Smith" (provider) → should keep
    - "Mr. Smith" (patient family) → should redact
    - "Smith Clinic" (facility) → depends on task
    """
    if not ambiguous_mentions:
        return 1.0
    if not isinstance(submitted_text, str):
        return 0.0
    
    correct_decisions = 0
    
    for mention in ambiguous_mentions:
        # Check if it's a provider title
        is_provider = any(
            title.lower() in mention.lower()
            for title in ["Dr.", "Doctor", "Nurse", "MD", "RN", "Prof."]
        )
        
        # Check if it's a patient/family member
        is_patient = any(
            title.lower() in mention.lower()
            for title in ["Mr.", "Mrs.", "Ms.", "Miss", "patient", "family"]
        )
        
        if is_provider:
            # Should preserve provider mentions
            if _contains(submitted_text, mention) and not _looks_like_redacted(submitted_text, mention):
                correct_decisions += 1
        elif is_patient:
            # Should redact patient/family mentions
            if not _contains(submitted_text, mention) or _looks_like_redacted(submitted_text, mention):
                correct_decisions += 1
        else:
            # Neutral mentions (e.g., just surname "Li", "Smith"): require contextual consistency
            # Check if this ambiguous surname is part of a provider or patient identifier
            is_part_of_provider = any(mention.lower() in prov_id.lower() for prov_id in provider_ids)
            is_part_of_patient = any(mention.lower() in pat_id.lower() for pat_id in patient_ids)
            
            if _contains(submitted_text, mention):
                # Mention is present - give credit if it's preserved and associated with provider
                if is_part_of_provider and not _looks_like_redacted(submitted_text, mention):
                    correct_decisions += 1
            else:
                # Mention is absent - give credit only if it's primarily a patient identifier
                # Don't give credit if it's also part of provider IDs (those should be preserved)
                if is_part_of_patient and not is_part_of_provider:
                    correct_decisions += 1
    
    return correct_decisions / len(ambiguous_mentions)


def _contains(text: str, needle: str) -> bool:
    return needle.lower() in text.lower()


def _has_redaction_token(text: str) -> bool:
    lowered = text.lower()
    return "[redacted" in lowered or "<redacted" in lowered


def _looks_like_redacted(text: str, mention: str) -> bool:
    """
    Return True only if a redaction token appears within ~40 characters of
    where *mention* sits (or sat) in *text*.

    This prevents a single unrelated [REDACTED_PATIENT] anywhere in the
    document from making every other identifier 'look redacted'.
    """
    idx = text.lower().find(mention.lower())
    if idx == -1:
        # Mention is absent — use the start of text as reference; the agent
        # may have placed the redaction token near position 0 after removing it.
        # Scan the whole text but require at least one token to be present.
        return _has_redaction_token(text)
    # Mention is present — inspect only the window around it.
    window_start = max(0, idx - 40)
    window_end = idx + len(mention) + 40
    vicinity = text[window_start:window_end]
    return _has_redaction_token(vicinity)


def generate_ambiguous_records(seed: int = 42) -> list[AnnotatedRecord]:
    """
    Generate typed AnnotatedRecord objects for Task 5 with contextual PHI tokens.

    Each record contains:
    - patient_identifier tokens (redact)
    - provider_identifier tokens (preserve)
    - ambiguous tokens that require contextual handling
    """
    rng = Random(seed)

    base_cases: list[dict[str, Any]] = [
        {
            "clinical_notes": """
                Dr. Sarah Johnson examined the patient. Mr. Johnson (patient's father) 
                reported symptoms. The Johnson family has history of diabetes. 
                Referral made to Johnson Medical Center.
            """,
            "patient_identifier": ["Mr. Johnson"],
            "provider_identifier": ["Dr. Sarah Johnson", "Johnson Medical Center"],
            "ambiguous": ["Johnson"],
        },
        {
            "clinical_notes": """
                Doctor Maria Silva consulted on the case. The patient's cousin, Ana Silva, 
                provided history. Silva Clinic specializes in endocrinology.
            """,
            "patient_identifier": ["Ana Silva"],
            "provider_identifier": ["Doctor Maria Silva", "Silva Clinic"],
            "ambiguous": ["Silva"],
        },
        {
            "clinical_notes": """
                Surgeon K. Patel operated successfully. Patient's brother, Ravi Patel, 
                signed consent. Patel Center reported the outcome.
            """,
            "patient_identifier": ["Ravi Patel"],
            "provider_identifier": ["Surgeon K. Patel", "Patel Center"],
            "ambiguous": ["Patel"],
        },
        {
            "clinical_notes": """
                Nurse practitioner Smith assessed vital signs. Patient's wife, Mrs. Smith, 
                expressed concern about medication. Dr. Brown will follow up. 
                Smith Pharmaceuticals developed the medication.
            """,
            "patient_identifier": ["Mrs. Smith"],
            "provider_identifier": ["Nurse practitioner Smith", "Dr. Brown", "Smith Pharmaceuticals"],
            "ambiguous": ["Smith"],
        },
        {
            "clinical_notes": """
                Dr. Lee consulted on the case. The patient's daughter, Mary Lee, 
                provided consent. Lee General Hospital has advanced facilities.
            """,
            "patient_identifier": ["Mary Lee"],
            "provider_identifier": ["Dr. Lee", "Lee General Hospital"],
            "ambiguous": ["Lee"],
        },
        {
            "clinical_notes": """
                Prof. Mueller reviewed labs. Patient's mother, Anna Mueller, attended. 
                Mueller Research Group published the findings.
            """,
            "patient_identifier": ["Anna Mueller"],
            "provider_identifier": ["Prof. Mueller", "Mueller Research Group"],
            "ambiguous": ["Mueller"],
        },
        {
            "clinical_notes": """
                Prof. Martinez reviewed the imaging. Patient's brother, Carlos Martinez, 
                was contacted. Martinez Research Institute approved the study.
            """,
            "patient_identifier": ["Carlos Martinez"],
            "provider_identifier": ["Prof. Martinez", "Martinez Research Institute"],
            "ambiguous": ["Martinez"],
        },
        {
            "clinical_notes": """
                Dr Jean Dupont assessed the patient. His uncle, Paul Dupont, accompanied him. 
                Dupont Centre Hospitalier provided the room.
            """,
            "patient_identifier": ["Paul Dupont"],
            "provider_identifier": ["Dr Jean Dupont", "Dupont Centre Hospitalier"],
            "ambiguous": ["Dupont"],
        },
        {
            "clinical_notes": """
                Dr. Chen performed the procedure. Patient's guardian, Wei Chen, 
                signed consent forms. Chen Foundation funded the research.
            """,
            "patient_identifier": ["Wei Chen"],
            "provider_identifier": ["Dr. Chen", "Chen Foundation"],
            "ambiguous": ["Chen"],
        },
        {
            "clinical_notes": """
                RN Williams administered medication. The patient, James Williams, 
                complained of side effects. Williams Memorial Hospital noted the incident.
            """,
            "patient_identifier": ["James Williams"],
            "provider_identifier": ["RN Williams", "Williams Memorial Hospital"],
            "ambiguous": ["Williams"],
        },
        {
            "clinical_notes": """
                Attending Li oversaw care. The patient's sister, Mei Li, requested an update. 
                Li & Partners Clinic documented the visit.
            """,
            "patient_identifier": ["Mei Li"],
            "provider_identifier": ["Attending Li", "Li & Partners Clinic"],
            "ambiguous": ["Li"],
        },
    ]

    # Deterministic shuffle across Python versions
    base_cases = rng.sample(base_cases, k=len(base_cases))

    annotated: list[AnnotatedRecord] = []

    for idx, case in enumerate(base_cases):
        phi_tokens: list[PHIToken] = []

        for val in sorted(case.get("patient_identifier", [])):
            phi_tokens.append(PHIToken(
                category=PHICategory.PATIENT_IDENTIFIER,
                value=val,
                field="clinical_notes",
                redaction_token="[REDACTED_PATIENT]",
            ))

        for val in sorted(case.get("provider_identifier", [])):
            phi_tokens.append(PHIToken(
                category=PHICategory.PROVIDER_IDENTIFIER,
                value=val,
                field="clinical_notes",
                redaction_token="",  # providers should be preserved; no redaction hint
            ))

        for val in sorted(case.get("ambiguous", [])):
            phi_tokens.append(PHIToken(
                category=PHICategory.AMBIGUOUS,
                value=val,
                field="clinical_notes",
                redaction_token="[REDACTED_AMBIGUOUS]",
            ))

        annotated.append(AnnotatedRecord(
            record_id=f"TASK5_{idx}",
            mrn=f"MRN_TASK5_{idx}",
            patient_name="Task 5 Patient",
            dob="1970-01-01",
            gender="U",
            clinical_notes=case["clinical_notes"],
            phi_tokens=phi_tokens,
        ))

    return annotated
