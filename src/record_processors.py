"""
Core record processing functions for the Medical Records OpenEnv.
These are shared utilities used by both baseline_agent and ner_agent to avoid circular imports.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Protocol, runtime_checkable
import os
import re

from .utils import PHI_PATTERNS, is_valid_icd10, normalize_date

# NER confidence threshold for Task 3 adversarial scrubbing — mirrors NER_CONFIDENCE_ADVERSARIAL
# in ner_agent.py. Read directly from env to avoid importing from ner_agent here
# (record_processors must stay free of ner_agent imports to prevent circular imports).
_NER_CONF_ADVERSARIAL = float(os.getenv("NER_CONFIDENCE_THRESHOLD_ADVERSARIAL", "0.3"))


# ---------------------------------------------------------------------------
# Optional NER Agent Protocol (for type hints without circular imports)
# ---------------------------------------------------------------------------

@runtime_checkable
class NERAgentProtocol(Protocol):
    """Protocol for NER agents that can redact text."""
    nlp: Any
    def redact_text(self, text: str, confidence_threshold: float = 0.4) -> str: ...


# ---------------------------------------------------------------------------
# Task 1 — Hygiene rule-based fixes
# ---------------------------------------------------------------------------

_VALID_ICD10_FALLBACK = "I10"   # hypertension — safe generic replacement


def _normalize_vitals(vitals: dict[str, Any]) -> dict[str, Any]:
    """Ensure all vitals fields are floats if present."""
    v = deepcopy(vitals)
    for k, val in v.items():
        if val is not None:
            try:
                v[k] = float(val)
            except (ValueError, TypeError):
                v[k] = None
    return v


def _fix_record_task1(record: dict[str, Any]) -> dict[str, Any]:
    """Apply deterministic cleaning rules to a single record, including OCR noise fix."""
    rec = deepcopy(record)

    # 0. OCR Noise Fix (Basic substitutions)
    def fix_ocr(text: str) -> str:
        if not text: return text
        subs = {
            "0": "o", "1": "l", "5": "s", "2": "z", "8": "B", "3": "E", "@": "a"
        }
        # Only fix if it looks like noise (e.g. "J0hn" or "E11.9")
        # For names, we expect mostly letters
        chars = list(text)
        for i, char in enumerate(chars):
            if char in subs:
                # Heuristic: if surrounded by letters, it's likely noise
                prev_is_alpha = i > 0 and chars[i-1].isalpha()
                next_is_alpha = i < len(chars)-1 and chars[i+1].isalpha()
                if prev_is_alpha or next_is_alpha:
                    chars[i] = subs[char]
        return "".join(chars)

    if rec.get("patient_name"):
        rec["patient_name"] = fix_ocr(rec["patient_name"])

    # 1. Fix date of birth (DOB)
    if rec.get("dob"):
        normalised = normalize_date(rec["dob"])
        if normalised:
            rec["dob"] = normalised

    # 2. Fix gender (normalize to M/F/Other)
    gender = str(rec.get("gender") or "").upper().strip()
    if gender in ("MALE", "M", "MAN", "BOY"):
        rec["gender"] = "M"
    elif gender in ("FEMALE", "F", "WOMAN", "GIRL"):
        rec["gender"] = "F"
    elif gender in ("OTHER", "NON-BINARY", "NB", "O"):
        rec["gender"] = "Other"
    else:
        # Ground truth is usually M/F in this generator, but we'll be safe
        rec["gender"] = "Other"

    # 3. Fix ICD-10 codes (ensure validity and format)
    codes = rec.get("icd10_codes") or []
    fixed_codes = []
    for c in codes:
        if not c: continue
        c_clean = str(c).strip().upper()
        if is_valid_icd10(c_clean):
            fixed_codes.append(c_clean)
        else:
            # Try to fix common formatting issues (e.g., missing dot)
            if len(c_clean) > 3 and "." not in c_clean:
                fixed = f"{c_clean[:3]}.{c_clean[3:]}"
                if is_valid_icd10(fixed):
                    fixed_codes.append(fixed)
                    continue
            fixed_codes.append(_VALID_ICD10_FALLBACK)
    rec["icd10_codes"] = fixed_codes

    # 4. Fix vitals (ensure floats and handle units)
    if rec.get("vitals"):
        vitals = rec["vitals"] if isinstance(rec.get("vitals"), dict) else {}
        rec["vitals"] = _normalize_vitals(vitals)

    # 5. Fix medication names and units
    meds = rec.get("medications") or []
    if not isinstance(meds, list):
        meds = []
    rec["medications"] = meds
    for med in meds:
        # Standardize name (Capital Case)
        if med.get("name"):
            med["name"] = med["name"].strip().capitalize()

        # Fix units in frequency field (Task 1 specific injected flaw)
        freq = med.get("frequency") or ""
        # Remove patterns like "500 mcg " from the start of frequency
        cleaned = re.sub(r"^\d+\s*(mcg|µg|g|ug|mg)\s*", "", freq, flags=re.IGNORECASE)
        med["frequency"] = cleaned.strip()

    # 6. Ensure required text fields are non-None
    for field in ("phone", "email", "address"):
        if rec.get(field) is None or str(rec.get(field)).lower() in ("none", "null", "n/a", ""):
            rec[field] = f"[UNKNOWN_{field.upper()}]"

    return rec


# ---------------------------------------------------------------------------
# Task 2 — PHI redaction rule-based
# ---------------------------------------------------------------------------

REDACTION_MAP = {
    "patient_name": "[REDACTED_NAME]",
    "mrn": "[REDACTED_MRN]",
    "dob": "[REDACTED_DOB]",
    "phone": "[REDACTED_PHONE]",
    "email": "[REDACTED_EMAIL]",
    "address": "[REDACTED_ADDRESS]",
}

NOTES_PHI_FIELDS = ["patient_name", "mrn", "dob", "phone", "email", "address"]


def _redact_record(record: dict[str, Any], ner_agent: NERAgentProtocol | None = None) -> dict[str, Any]:
    """Redact all known PHI fields and scrub notes.
    
    Args:
        record: The medical record to redact
        ner_agent: Optional NER agent for ML-powered redaction pass
    """
    rec = deepcopy(record)
    phi_values: list[str] = []
    last_name: str = ""
    first_name: str = ""

    # 1. Collect structured PHI values for literal replacement
    raw_name = rec.get("patient_name") or ""
    if raw_name:
        parts = raw_name.split()
        if len(parts) >= 2:
            first_name = parts[0]
            last_name = parts[-1]
        else:
            last_name = raw_name

    for field, token in REDACTION_MAP.items():
        val = rec.get(field)
        if val and val not in (None, ""):
            phi_values.append(str(val))
            rec[field] = token

    # 2. Scrub PHI from clinical notes
    notes = rec.get("clinical_notes") or ""

    # Literal replacement for all structured PHI values collected
    # Sort by length (longest first) to avoid partial replacement bugs
    for val in sorted(phi_values, key=len, reverse=True):
        if len(val) > 2:  # Avoid redacting tiny strings
            pattern = re.compile(re.escape(val), re.IGNORECASE)
            notes = pattern.sub("[REDACTED]", notes)

    # Informal name aliases (e.g., "Pt. Smith", "Pt Smith", "Smith, J")
    if last_name and len(last_name) > 2:
        # "Pt. Smith" or "Patient Smith"
        notes = re.sub(
            rf"\b(Pt\.?|Patient)\s+{re.escape(last_name)}\b",
            r"\1 [REDACTED_NAME]",
            notes,
            flags=re.IGNORECASE,
        )
        # "Mr./Ms. Smith"
        notes = re.sub(
            rf"\b(Mr\.?|Ms\.?|Mrs\.?|Dr\.?)\s+{re.escape(last_name)}\b",
            r"\1 [REDACTED_NAME]",
            notes,
            flags=re.IGNORECASE,
        )

    if first_name and len(first_name) > 2:
        # First name mentions
        notes = re.sub(
            rf"\b{re.escape(first_name)}\b",
            "[REDACTED_NAME]",
            notes,
            flags=re.IGNORECASE,
        )

    # Initials (e.g., "J.S.")
    if first_name and last_name:
        initials = rf"{first_name[0]}\.?\s*{last_name[0]}\.?"
        notes = re.sub(
            rf"\b{initials}\b",
            "[REDACTED_NAME]",
            notes,
            flags=re.IGNORECASE,
        )

    # 3. Use global patterns from utils.py for broad scrubbing (Email, Phone, MRN, etc.)
    for category, pattern in PHI_PATTERNS.items():
        token = f"[REDACTED_{category}]"
        notes = pattern.sub(token, notes)

    # 4. ML-powered pass using NER agent (Hybrid approach)
    if ner_agent is not None and ner_agent.nlp is not None:
        notes = ner_agent.redact_text(notes)

    rec["clinical_notes"] = notes
    return rec


# ---------------------------------------------------------------------------
# Task 3 — Anonymisation rule-based
# ---------------------------------------------------------------------------

def _age_group(dob: str | None) -> str:
    if not dob:
        return "unknown"
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


def _anonymise_record(record: dict[str, Any], ner_agent: NERAgentProtocol | None = None) -> dict[str, Any]:
    """Pseudonymise record: redact PHI, bucket age, preserve clinical fields.
    
    Args:
        record: The medical record to anonymize
        ner_agent: Optional NER agent for ML-powered redaction pass
    """
    rec = _redact_record(record, ner_agent=ner_agent)  # first pass — remove all raw PHI (includes hybrid NER pass)
    dob_raw = record.get("dob")
    rec["age_group"] = _age_group(normalize_date(dob_raw) if dob_raw else None)
    
    # 2. Scrub indirect adversarial identifiers (Task 3 specific)
    notes = rec.get("clinical_notes") or ""
    
    # Pattern 1: the <anything> patient from <zip>
    notes = re.sub(
        r"the [^.]+ patient from \d{5}",
        "[REDACTED_TAG]",
        notes,
        flags=re.IGNORECASE,
    )
    
    # Pattern 2: sibling of <last_name>
    raw_name = record.get("patient_name") or ""
    if raw_name:
        # Collect all parts of the name as potential last names or identifiers
        name_parts = re.findall(r"\w+", raw_name)
        for part in name_parts:
            if len(part) > 2:
                notes = re.sub(
                    rf"sibling of {re.escape(part)}",
                    "sibling of [REDACTED_NAME]",
                    notes,
                    flags=re.IGNORECASE,
                )

    # Pattern 3: Case structurally resembles ...
    notes = re.sub(r"Case structurally resembles [^.]+", "Case structurally resembles [REDACTED]", notes, flags=re.IGNORECASE)
    
    # 3. Second-pass PHI scrubber using NER agent if available
    # Targeted sweep for adversarial identifiers (e.g. clinic names, doctor names)
    if ner_agent is not None and ner_agent.nlp is not None:
        # Lower threshold for Task 3 scrubber to catch more potential linkage markers
        notes = ner_agent.redact_text(notes, confidence_threshold=_NER_CONF_ADVERSARIAL)
    
    rec["clinical_notes"] = notes
    return rec


# ---------------------------------------------------------------------------
# Task 5 — Contextual PII Disambiguation
# ---------------------------------------------------------------------------

def _redact_contextual_phi(record: dict[str, Any]) -> dict[str, Any]:
    """
    Decide which mentions are PII based on context for Task 5.
    - Patient/family identifiers (redact): "Mr. Johnson", "Mrs. Smith", etc.
    - Provider identifiers (keep): "Dr. Johnson", "Nurse Smith", etc.
    """
    rec = deepcopy(record)
    notes = rec.get("clinical_notes") or ""

    # Protect provider mentions so we don't redact them in downstream passes
    provider_patterns = [
        r"\bDr\.\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?",
        r"\bDoctor\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?",
        r"\bNurse\s+[A-Z][a-z]+",
        r"\bRN\s+[A-Z][a-z]+",
        r"\bMD\s+[A-Z][a-z]+",
        r"\bProf\.\s+[A-Z][a-z]+",
        r"\bSurgeon\s+[A-Z][a-z]+",
        r"\bAttending\s+[A-Z][a-z]+",
    ]

    protected: dict[str, str] = {}
    for idx, pattern in enumerate(provider_patterns):
        for match in re.finditer(pattern, notes):
            placeholder = f"__PROVIDER_{idx}_{len(protected)}__"
            protected[placeholder] = match.group(0)
            notes = notes.replace(match.group(0), placeholder)

    # 1. Identify and redact patient/family identifiers
    patient_titles = ["Mr.", "Mrs.", "Ms.", "Miss", "patient", "family"]
    family_terms = ["brother", "sister", "mother", "father", "daughter", "son", "cousin", "uncle", "aunt", "guardian", "wife", "husband"]

    # Title + Name
    for title in patient_titles:
        pattern = rf"\b{re.escape(title)}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?"
        notes = re.sub(pattern, "[REDACTED_PATIENT]", notes)

    # Family-member constructs (e.g., "Smith family", "Johnson brother")
    notes = re.sub(
        rf"\b([A-Z][a-z]+)\s+(?:family|{'|'.join(family_terms)})\b",
        "[REDACTED_PATIENT]",
        notes,
    )

    # Possessive patient references with family term and name (e.g., "patient's daughter, Mary Lee")
    notes = re.sub(
        rf"patient['’]s\s+(?:{'|'.join(family_terms)})\s*,?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?",
        "[REDACTED_PATIENT]",
        notes,
        flags=re.IGNORECASE,
    )

    # 2. Restore protected provider mentions
    for placeholder, value in protected.items():
        notes = notes.replace(placeholder, value)

    rec["clinical_notes"] = notes
    return rec


# ---------------------------------------------------------------------------
# Task 4 — Clinical Knowledge Extraction rule-based
# ---------------------------------------------------------------------------

def _extract_knowledge_rule_based(record: dict[str, Any], ner_agent: NERAgentProtocol | None = None) -> dict[str, Any]:
    """Extract entities and generate summary using rule + ML hybrid approach.
    
    Args:
        record: The medical record to extract knowledge from
        ner_agent: Optional NER agent for ML-powered entity extraction
    """
    # 1. Rule-based extraction from structured fields
    entities = []
    for code in record.get("icd10_codes", []):
        entities.append({"text": code, "type": "Condition", "code": code})
    
    for med in record.get("medications", []):
        name = med.get("name", "Unknown")
        entities.append({"text": name, "type": "Medication", "code": name})
        
    # 2. ML-powered extraction from clinical notes (Task 4 is usually clean but dense)
    if ner_agent is not None and ner_agent.nlp is not None:
        notes = record.get("clinical_notes", "")
        # Use the pipeline directly to get raw entities for extraction
        raw_entities = ner_agent.nlp(notes)
        for ent in raw_entities:
            # We filter for medical/relevant entities if the model supports it
            # For general BERT, we'll map ORG/LOC/PER if they look like medical context
            label = ent["entity_group"]
            text = ent["word"]
            
            # Smart context mapping: ignore if already found in structured fields
            if not any(e["text"].lower() == text.lower() for e in entities):
                if label == "PER":
                    # Potentially a doctor or specialist mentioned in notes
                    entities.append({"text": text, "type": "Provider", "code": "N/A"})
                elif label == "LOC":
                    # Potentially a facility or department
                    entities.append({"text": text, "type": "Facility", "code": "N/A"})
                elif label == "ORG":
                    # Potentially a lab or medical organization
                    entities.append({"text": text, "type": "Organization", "code": "N/A"})
    
    # 3. Generate a comprehensive structured summary from the record's structured fields AND clinical notes.
    # This template-based approach ensures all key clinical facts are captured for high Jaccard/BERTScore.
    summary_parts: list[str] = []

    # Demographic line - start with patient context
    dob = record.get("dob", "")
    gender = record.get("gender", "")
    if dob and gender:
        try:
            from datetime import date, datetime
            birth = datetime.strptime(dob, "%Y-%m-%d").date()
            age = int((date.today() - birth).days / 365.25)
            gender_word = "male" if gender == "M" else "female" if gender == "F" else "patient"
            summary_parts.append(f"{age}-year-old {gender_word}")
        except Exception:
            pass

    # Conditions - capture all diagnoses
    icd_codes = record.get("icd10_codes") or []
    if icd_codes:
        summary_parts.append(f"diagnosed with {', '.join(icd_codes)}")

    # Medications - capture all prescriptions with details
    meds = record.get("medications") or []
    if meds:
        med_strs = []
        for m in meds:
            name = m.get("name", "Unknown") if isinstance(m, dict) else getattr(m, "name", "Unknown")
            dose = m.get("dose_mg") if isinstance(m, dict) else getattr(m, "dose_mg", None)
            freq = m.get("frequency") if isinstance(m, dict) else getattr(m, "frequency", None)
            dose_str = f" {int(dose)} mg" if dose else ""
            freq_str = f" {freq}" if freq else ""
            med_strs.append(f"{name}{dose_str}{freq_str}")
        summary_parts.append(f"prescribed {'; '.join(med_strs)}")

    # Vitals - capture current measurements
    vitals = record.get("vitals") or {}
    if isinstance(vitals, dict):
        sbp = vitals.get("systolic_bp_mmhg")
        dbp = vitals.get("diastolic_bp_mmhg")
        hr = vitals.get("heart_rate_bpm")
        vital_parts = []
        if sbp and dbp:
            vital_parts.append(f"blood pressure {int(sbp)}/{int(dbp)} mmHg")
        if hr:
            vital_parts.append(f"heart rate {int(hr)} bpm")
        if vital_parts:
            summary_parts.append(f"with {' and '.join(vital_parts)}")

    # Extract key clinical information from notes using sentence extraction
    clinical_notes = record.get("clinical_notes", "")
    if clinical_notes:
        # Extract important sentences (simple heuristic: take first 2-3 sentences)
        sentences = [s.strip() for s in clinical_notes.replace('\n', ' ').split('.') if s.strip()]
        if sentences:
            # Take up to 3 most informative sentences
            key_sentences = sentences[:min(3, len(sentences))]
            notes_summary = '. '.join(key_sentences) + '.'
            summary_parts.append(f"Clinical presentation: {notes_summary}")

    # Combine all parts into coherent summary
    summary = " ".join(summary_parts) if summary_parts else "No structured data available."

    return {"entities": entities, "summary": summary.strip()}
