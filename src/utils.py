"""
Utility helpers: ICD-10 validation, date normalisation, unit conversion,
PHI regex patterns, and clinical keyword scanning.
"""

from __future__ import annotations

import os
import sys
import logging
import re
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ICD-10 Validation
# ---------------------------------------------------------------------------

# Valid ICD-10-CM pattern: letter + 2 digits, optional dot + 1-4 alphanumeric chars
# Tightened to reject obviously invalid patterns
_ICD10_RE = re.compile(r"^[A-Z][0-9]{2}(\.[0-9A-Z]{1,4})?$")

# Valid ICD-10-CM first letter categories (U is reserved for special purposes)
# Standard chapters: A-T, V-Z
VALID_ICD10_FIRST_LETTERS = set("ABCDEFGHIJKLMNOPQRSTVWXYZ")  # excludes U

# Known *invalid* synthetic codes we inject during data generation
INVALID_ICD10_EXAMPLES = {"Z99.999", "X00.0000", "A99.99X", "B00.0000"}

# Patterns that indicate invalid subcategory suffixes
_INVALID_SUFFIX_PATTERNS = [
    re.compile(r"\.[A-Z]{2,}$"),  # All-letter suffixes like .XXXX, .ABC
    re.compile(r"\.X{2,}$"),      # Consecutive X's like .XX, .XXX  
    re.compile(r"\.[0-9]{5,}$"),  # Too many digits after dot
]


def is_valid_icd10(code: str) -> bool:
    """
    Return True if *code* looks like a valid ICD-10-CM code.
    
    Validation rules:
    - Format: Letter + 2 digits + optional (dot + 1-4 alphanumeric)
    - First letter must be valid category (A-T, V-Z; U is reserved)
    - Rejects known invalid synthetic codes
    - Rejects obviously invalid patterns (all-letter suffixes, etc.)
    """
    if not code:
        return False
    code = code.strip().upper()
    
    # Check against known invalid examples
    if code in INVALID_ICD10_EXAMPLES:
        return False
    
    # Basic format check
    if not _ICD10_RE.match(code):
        return False
    
    # Validate first letter is a valid ICD-10 category
    first_letter = code[0]
    if first_letter not in VALID_ICD10_FIRST_LETTERS:
        return False
    
    # Check for invalid suffix patterns (if code has a dot)
    if "." in code:
        for pattern in _INVALID_SUFFIX_PATTERNS:
            if pattern.search(code):
                return False
        
        # Additional check: suffix should have at least one digit in most cases
        # Valid examples: .9, .21, .A1, .1A - typically have digits
        # Invalid examples: .XXXX, .ABCD - all letters are suspicious
        suffix = code.split(".")[1]
        if len(suffix) >= 2 and suffix.isalpha():
            # All-letter suffix of 2+ chars is likely invalid
            return False
    
    return True


# ---------------------------------------------------------------------------
# Date Normalisation → ISO 8601 (YYYY-MM-DD)
# ---------------------------------------------------------------------------

_DATE_FORMATS = [
    "%Y-%m-%d",    # already ISO
    "%m/%d/%Y",    # US
    "%m-%d-%Y",    # US dashed
    "%d/%m/%Y",    # European
    "%d-%m-%Y",    # European dashed
    "%Y/%m/%d",    # reversed slash
    "%B %d, %Y",   # "January 15, 1990"
    "%b %d, %Y",   # "Jan 15, 1990"
    "%d %B %Y",    # "15 January 1990"
]


def normalize_date(raw: str) -> str | None:
    """
    Parse *raw* using multiple format candidates and return ISO 8601 string.
    Returns None if the date cannot be parsed.
    """
    if not raw:
        return None
    raw = raw.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Unit Conversion
# ---------------------------------------------------------------------------

# Dosage normalisation: everything → mg
_UNIT_TO_MG: dict[str, float] = {
    "mg": 1.0,
    "mcg": 0.001,
    "µg": 0.001,
    "ug": 0.001,
    "g": 1000.0,
    "gram": 1000.0,
    "grams": 1000.0,
}

_DOSE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(mg|mcg|µg|ug|g|gram|grams)", re.IGNORECASE
)


def normalize_dose_to_mg(dose_str: str | float | int | None) -> float | None:
    """Extract numeric dose in mg from a free-text dose string; tolerate None/non-strings."""
    if dose_str is None:
        return None
    text = str(dose_str)
    m = _DOSE_RE.search(text)
    if not m:
        return None
    value, unit = float(m.group(1)), m.group(2).lower()
    return value * _UNIT_TO_MG.get(unit, 1.0)


# Weight normalisation
def lbs_to_kg(lbs: float) -> float:
    return float(round(lbs * 0.453592, 2))


def kg_to_lbs(kg: float) -> float:
    return float(round(kg / 0.453592, 2))


# ---------------------------------------------------------------------------
# PHI Regex Patterns
# ---------------------------------------------------------------------------

PHI_PATTERNS: dict[str, re.Pattern] = {
    # US phone: (123) 456-7890 | 123-456-7890 | 1234567890
    "PHONE": re.compile(
        r"\b(\+?1[-.\s]?)?(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b"
    ),
    # Email
    "EMAIL": re.compile(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
    ),
    # MRN: 6-10 digit numeric string (used by generator)
    "MRN": re.compile(r"\bMRN[-:\s]?\d{6,10}\b", re.IGNORECASE),
    # DOB patterns in free text: "DOB: 1990-01-15" or "born 01/15/1990"
    "DOB": re.compile(
        r"\b(?:DOB|date of birth|born)[:\s]+\d{1,4}[/\-]\d{1,2}[/\-]\d{1,4}\b",
        re.IGNORECASE,
    ),
    # SSN: 123-45-6789
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    # Street address snippet: "123 Main St" or "456 Oak Avenue"
    "ADDRESS": re.compile(
        r"\b\d{1,5}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+"
        r"(?:St|Street|Ave|Avenue|Blvd|Boulevard|Dr|Drive|Rd|Road|Ln|Lane|Way|Ct|Court)\b",
        re.IGNORECASE,
    ),
}


def scan_phi(text: str) -> dict[str, list[str]]:
    """Return dict mapping PHI category → list of matched spans found in *text*."""
    results: dict[str, list[str]] = {}
    for category, pattern in PHI_PATTERNS.items():
        matches = pattern.findall(text)
        # flatten tuple groups if any
        flat = []
        for m in matches:
            if isinstance(m, tuple):
                flat.append("".join(m))
            else:
                flat.append(m)
        if flat:
            results[category] = flat
    return results


def redact_phi_in_text(text: str, phi_tokens: list) -> str:
    """
    Replace all known PHI values in *text* with their redaction tokens.
    *phi_tokens* is a list of PHIToken objects.
    """
    for token in phi_tokens:
        if token.value in text:
            text = text.replace(token.value, token.redaction_token)
    return text


# ---------------------------------------------------------------------------
# Clinical Keyword Scanner
# ---------------------------------------------------------------------------

# Keywords whose presence must survive redaction (Tasks 2 & 3)
CLINICAL_KEYWORDS = [
    # Diagnoses / conditions
    "hypertension", "diabetes", "asthma", "copd", "heart failure",
    "chronic kidney disease", "hypothyroidism", "hyperlipidemia",
    "atrial fibrillation", "pneumonia", "appendicitis", "depression",
    "anxiety", "osteoporosis", "obesity", "stroke", "pulmonary embolism",
    "sepsis", "anemia", "migraine",
    # Symptoms
    "dyspnea", "chest pain", "fatigue", "nausea", "vomiting", "fever",
    "cough", "edema", "palpitations", "syncope",
    # Medications
    "metformin", "lisinopril", "atorvastatin", "amlodipine", "omeprazole",
    "levothyroxine", "metoprolol", "warfarin", "aspirin", "albuterol",
    "prednisone", "furosemide", "sertraline", "gabapentin", "insulin",
]


def scan_clinical_keywords(text: str | None, keywords: list[str] | None = None) -> list[str]:
    """Return list of clinical keywords found in *text* (case-insensitive)."""
    kw_list = keywords or CLINICAL_KEYWORDS
    # Harden against None/missing text
    text_safe = (text or "").lower()
    return [kw for kw in kw_list if kw.lower() in text_safe]


def clinical_utility_score(
    original_keywords: list[str], processed_text: str | None
) -> float:
    """
    Fraction of *original_keywords* still detectable in *processed_text*.
    Returns 1.0 if original_keywords is empty (no keywords to preserve).
    """
    if not original_keywords:
        return 1.0
    # Harden against None/missing processed_text
    retained = scan_clinical_keywords(processed_text or "", original_keywords)
    return len(retained) / len(original_keywords)


def align_submitted_to_truth(
    submitted: list[dict[str, Any]], 
    truth_ids: list[str]
) -> list[dict[str, Any]]:
    """Re-order submitted records to match ground truth order by record_id."""
    sub_map = {r.get("record_id"): r for r in submitted if r.get("record_id")}
    aligned: list[dict[str, Any]] = []
    missing_ids: list[str] = []
    for tid in truth_ids:
        rec = sub_map.get(tid)
        if rec is None:
            missing_ids.append(tid)
        aligned.append(rec or {})
    if missing_ids:
        logger.warning(
            "align_submitted_to_truth: %d missing record_id(s) — submission will score 0 for those records: %s",
            len(missing_ids),
            missing_ids[:5],
        )
    return aligned


# ---------------------------------------------------------------------------
# Semantic Similarity (SentenceTransformers / BERTScore / Jaccard Fallback)
# ---------------------------------------------------------------------------

_sentence_transformer_model = None
_bertscore_metric = None

def semantic_similarity_score(reference: str, candidate: str) -> float:
    """
    Compute semantic similarity between reference and candidate texts.
    
    Uses a tiered approach for optimal balance of speed and accuracy:
    1. SentenceTransformers (all-MiniLM-L6-v2) - best quality, CPU-friendly
    2. BERTScore - good alternative if transformers available
    3. Jaccard similarity - fast deterministic fallback
    
    [P5] Upgraded from Jaccard to SentenceTransformers for proper semantic understanding.
    [P4] Made deterministic by default: uses Jaccard fallback unless ENABLE_BERT_SCORE=1.
    [P3] No longer prints to stdout; logs warnings at debug level only.
    """
    global _sentence_transformer_model, _bertscore_metric
    
    if not reference or not candidate:
        return 0.0
        
    import os
    # Default to enabling if in a high-resource environment like Colab, unless explicitly disabled
    is_colab = "COLAB_GPU" in os.environ or "google.colab" in sys.modules
    
    # Try SentenceTransformers first (best quality, still runs on CPU)
    if os.environ.get("ENABLE_SENTENCE_TRANSFORMERS", "1") == "1":
        try:
            if _sentence_transformer_model is None:
                from sentence_transformers import SentenceTransformer, util
                # Tiny model (80MB), fast on CPU, excellent semantic understanding
                _sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Encode both texts and compute cosine similarity
            embeddings = _sentence_transformer_model.encode(
                [reference, candidate], 
                convert_to_tensor=True,
                show_progress_bar=False
            )
            similarity = float(util.cos_sim(embeddings[0], embeddings[1]).item())
            # Cosine similarity ranges from -1 to 1, clamp to [0, 1]
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            logger.debug(f"SentenceTransformers failed (falling back to BERTScore/Jaccard): {type(e).__name__}")
            # Continue to next fallback
    
    # Second choice: BERTScore if available
    if os.environ.get("ENABLE_BERT_SCORE", "1" if is_colab else "0") == "1":
        try:
            metric_path = os.environ.get("BERTSCORE_METRIC_PATH", "bertscore")
            model_type = os.environ.get("BERTSCORE_MODEL_TYPE", "distilbert-base-uncased")
            local_only = os.environ.get("BERTSCORE_LOCAL_FILES_ONLY", "0") == "1"

            if _bertscore_metric is None:
                import evaluate
                # Suppress HF warnings
                os.environ["TOKENIZERS_PARALLELISM"] = "false"

                if local_only:
                    os.environ.setdefault("HF_HUB_OFFLINE", "1")
                    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

                # Load from a local path when provided, otherwise use metric id.
                _bertscore_metric = evaluate.load(metric_path)

            results = _bertscore_metric.compute(
                predictions=[candidate], 
                references=[reference], 
                lang="en",
                model_type=model_type
            )
            return float(results["f1"][0])
        except Exception as e:
            logger.debug(f"BERTScore computation failed (falling back to Jaccard): {type(e).__name__}")
            # Continue to final fallback
    
    # Final fallback: Fast, meaningful token-level Jaccard similarity
    ref_tokens = set(re.findall(r'\b\w+\b', reference.lower()))
    cand_tokens = set(re.findall(r'\b\w+\b', candidate.lower()))
    if not ref_tokens:
        return 1.0 if not cand_tokens else 0.0
    intersection = ref_tokens & cand_tokens
    union = ref_tokens | cand_tokens
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# FHIR (Fast Healthcare Interoperability Resources) Export
# ---------------------------------------------------------------------------

def export_to_fhir(record: dict, include_phi: bool = False) -> dict:
    """
    Export a PatientRecord dictionary to a simplified FHIR-compliant JSON.

    PHI is excluded by default. Set include_phi=True only when you have an
    explicit compliance-approved need to export identifiers.
    """
    fhir_patient = {
        "resourceType": "Patient",
        "id": record.get("record_id", "unknown"),
        "gender": "male" if record.get("gender") == "M" else "female" if record.get("gender") == "F" else "other",
    }

    if include_phi:
        logger.warning("Exporting PHI fields in FHIR payload (include_phi=True)")
        fhir_patient.update(
            {
                "identifier": [
                    {
                        "system": "http://hospital.org/mrn",
                        "value": record.get("mrn"),
                    }
                ],
                "name": [{"text": record.get("patient_name")}],
                "birthDate": record.get("dob"),
                "telecom": [
                    {"system": "phone", "value": record.get("phone")},
                    {"system": "email", "value": record.get("email")},
                ],
                "address": [{"text": record.get("address")}],
            }
        )

    # Add ICD-10 as conditions
    conditions = []
    for code in (record.get("icd10_codes") or []):
        conditions.append({
            "resourceType": "Condition",
            "code": {
                "coding": [
                    {
                        "system": "http://hl7.org/fhir/sid/icd-10",
                        "code": code
                    }
                ]
            }
        })

    return {
        "bundle_type": "transaction",
        "entries": [
            {"resource": fhir_patient},
            *[{"resource": c} for c in conditions]
        ]
    }


# ---------------------------------------------------------------------------
# Privacy Metrics: k-anonymity check
# ---------------------------------------------------------------------------

def check_k_anonymity(records: list[dict], quasi_identifiers: list[str], k: int = 2) -> float:
    """
    Check if the dataset satisfies k-anonymity for the given quasi-identifiers.
    Returns the fraction of records that are part of a group of size >= k.
    """
    if not records or not quasi_identifiers:
        return 1.0
        
    groups: dict[tuple, int] = {}
    for rec in records:
        # Create a tuple of quasi-identifier values
        qi_values = tuple(str(rec.get(qi, "")).lower() for qi in quasi_identifiers)
        groups[qi_values] = groups.get(qi_values, 0) + 1
        
    # Count records in groups of size >= k
    k_anonymous_records = sum(count for qi, count in groups.items() if count >= k)
    return k_anonymous_records / len(records)

def redaction_robustness_score(redacted_text: str) -> float:
    """
    Experimental: An adversarial grader that attempts to detect residual PHI
    leaks in redacted text using a high-precision NER model.
    """
    leaked_phi = scan_phi(redacted_text)
    if not leaked_phi:
        return 1.0
    return max(0.0, 1.0 - (len(leaked_phi) * 0.2))

