"""
Pydantic v2 models for the Medical Records OpenEnv.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# PHI categories
# ---------------------------------------------------------------------------

class PHICategory(str, Enum):
    NAME = "NAME"
    MRN = "MRN"
    DOB = "DOB"
    PHONE = "PHONE"
    EMAIL = "EMAIL"
    ADDRESS = "ADDRESS"
    SSN = "SSN"
    # Task 5 - Contextual PII
    PATIENT_IDENTIFIER = "patient_identifier"
    PROVIDER_IDENTIFIER = "provider_identifier"
    AMBIGUOUS = "ambiguous"


class PHIToken(BaseModel):
    """A single piece of Protected Health Information annotated in a record."""

    category: PHICategory
    value: str                       # exact text that must be redacted
    field: str                       # which record field contains this token
    redaction_token: str             # expected replacement, e.g. [REDACTED_NAME]


# ---------------------------------------------------------------------------
# Core patient record
# ---------------------------------------------------------------------------

class Vitals(BaseModel):
    heart_rate_bpm: float | None = None
    systolic_bp_mmhg: float | None = None
    diastolic_bp_mmhg: float | None = None
    temperature_c: float | None = None
    weight_kg: float | None = None
    height_cm: float | None = None


class Medication(BaseModel):
    name: str
    dose_mg: float | None = None
    frequency: str | None = None


class PatientRecord(BaseModel):
    """Clean, canonical patient record — ground truth."""

    record_id: str
    mrn: str                          # Medical Record Number
    patient_name: str
    dob: str                          # ISO 8601: YYYY-MM-DD
    gender: str                       # "M" | "F" | "Other"
    phone: str | None = None
    email: str | None = None
    address: str | None = None
    icd10_codes: list[str] = Field(default_factory=list)
    vitals: Vitals = Field(default_factory=Vitals)
    medications: list[Medication] = Field(default_factory=list)
    clinical_notes: str = ""          # free-text narrative


class DirtyRecord(PatientRecord):
    """PatientRecord with deliberately injected data-quality flaws (Task 1)."""

    injected_flaws: list[str] = Field(
        default_factory=list,
        description="Human-readable list of flaws injected for debugging/grading",
    )


class AnnotatedRecord(PatientRecord):
    """PatientRecord with PHI annotations for Tasks 2 & 3 ground truth."""

    phi_tokens: list[PHIToken] = Field(default_factory=list)
    clinical_keywords: list[str] = Field(
        default_factory=list,
        description="Keywords that must survive redaction (diagnoses, medications …)",
    )
    adversarial_identifiers: list[str] = Field(
        default_factory=list,
        description=(
            "Task 3 hidden indirect identifiers (e.g., rare disease + ZIP combos) "
            "that should be generalized or removed."
        ),
    )


# ---------------------------------------------------------------------------
# Environment I/O
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """What the agent receives at each step."""

    task_id: int
    task_description: str
    records: list[dict[str, Any]]    # serialised PatientRecord / DirtyRecord
    step: int
    max_steps: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeExtraction(BaseModel):
    # Harden against various LLM output formats
    entities: list[dict[str, Any]] = Field(default_factory=list)
    summary: str = ""


class Action(BaseModel):
    """What the agent submits at each step."""

    records: list[dict[str, Any]] | None = None    # agent's processed records
    knowledge: list[KnowledgeExtraction] | None = None
    is_final: bool = False           # set True to close episode early


class Reward(BaseModel):
    """Step-level reward signal."""

    score: float = Field(ge=0.0, le=1.0)
    breakdown: dict[str, Any] = Field(default_factory=dict)   # nested dicts/lists allowed
    done: bool = False
    info: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def clamp(cls, score: float, **kwargs) -> "Reward":
        """Helper that clamps score to [0, 1] before construction."""
        return cls(score=max(0.0, min(1.0, float(score or 0.0))), **kwargs)


class State(BaseModel):
    """Episode state snapshot — returned by state() method."""

    task_id: int
    seed: int
    step: int
    max_steps: int
    done: bool
    last_score: float | None
    last_breakdown: dict[str, Any]
    passed: bool
    audit_trail: list[dict[str, Any]]

