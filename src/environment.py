"""
MedicalOpenEnv — core environment class.
Implements reset() / step() / state() per the OpenEnv interface.
"""

from __future__ import annotations

from typing import Any, Callable
import logging

from .data_generator import EHRGenerator
from .models import (
    Action,
    AnnotatedRecord,
    DirtyRecord,
    Observation,
    PatientRecord,
    PHICategory,
    PHIToken,
    Reward,
    State,
)
from .tasks import task1_hygiene, task2_redaction, task3_anonymization, task4_knowledge, task5_reasoning

MAX_STEPS = 10

logger = logging.getLogger(__name__)

# Fields hidden from agent observations (grader-only metadata)
_HIDDEN_FIELDS = {"phi_tokens", "clinical_keywords", "adversarial_identifiers"}

TASK_DESCRIPTIONS = {
    1: (
        "Task 1 - Data Hygiene & Standardisation.\n"
        "You are given a batch of synthetic patient records that contain deliberate "
        "data-quality flaws: mixed date formats, wrong medication units, invalid ICD-10 "
        "codes, and missing required fields.\n"
        "Your job: return corrected records with all fields valid and standardised.\n"
        "Grader: correct_fields / total_fields -> 0.0-1.0. Pass bar: >= 0.85."
    ),
    2: (
        "Task 2 - PHI Detection & Redaction.\n"
        "Records contain Protected Health Information (PHI) embedded in both structured "
        "fields and free-text clinical notes.\n"
        "Replace every PHI occurrence with its category token: [REDACTED_NAME], "
        "[REDACTED_MRN], [REDACTED_DOB], [REDACTED_PHONE], [REDACTED_EMAIL], "
        "[REDACTED_ADDRESS].\n"
        "Do NOT remove clinical content (diagnoses, medications, symptoms).\n"
        "Grader: phi_score * 0.6 + utility_score * 0.4. "
        "Pass bar: phi_score == 1.0 AND utility_score >= 0.80."
    ),
    3: (
        "Task 3 - Full Anonymisation + Downstream Utility.\n"
        "Produce de-identified, pseudonymised records that (a) contain no PHI and "
        "(b) preserve enough clinical signal to keep a deterministic disease-risk "
        "model accurate. Use age-group buckets (18-40, 41-60, 61-75, 76+) instead "
        "of exact DOB; preserve gender, ICD-10 chapter prefix, and medication names.\n"
        "Grader: avg_phi * 0.4 + avg_ml * 0.3 + avg_fidelity * 0.2 + k_score * 0.1.\n"
        "Pass bar: phi_score == 1.0 AND ml_utility_score >= 0.60."
    ),
    4: (
        "Task 4 - Clinical Knowledge Extraction.\n"
        "Extract clinical entities (Conditions, Medications) and generate comprehensive "
        "patient summaries from clinical notes.\n"
        "Submit a list of knowledge objects (each {entities, summary}) in the same order as "
        "the input records. There is no record_id on a knowledge object — the grader pairs "
        "submitted_knowledge[i] with records[i] by index only.\n"
        "Grader: avg_entity_extraction * 0.4 + avg_code_precision * 0.3 + avg_summary_fidelity * 0.3. "
        "Pass bar: avg_entity_extraction >= 0.75 AND avg_summary_fidelity >= 0.50."
    ),
    5: (
        "Task 5 - Contextual PII Disambiguation (Expert).\n"
        "Advanced clinical reasoning challenge: Decide which mentions are patient PII based on context.\n"
        "Example: 'Dr. Smith saw Mr. Smith' — only 'Mr. Smith' is patient PII, 'Dr. Smith' is provider.\n"
        "You must distinguish patient identifiers from provider names, family members, and facility names.\n"
        "Grader: weighted combination of patient_phi_score (0.5), provider_phi_score (0.3), and contextual_accuracy (0.2).\n"
        "Pass bar: overall_score >= 0.70 AND patient_phi_score >= 0.80."
    ),
}

_TASK_GRADERS: dict[int, Callable[["MedicalOpenEnv", list[dict[str, Any]]], dict[str, Any]]] = {
    1: lambda env, submitted: task1_hygiene.grade(submitted, env._clean_truth),
    2: lambda env, submitted: task2_redaction.grade(submitted, env._annotated_records),
    3: lambda env, submitted: task3_anonymization.grade(
        submitted,
        env._annotated_records,
        env._baseline_ml_scores or None,
    ),
    4: lambda env, submitted: task4_knowledge.grade(submitted, env._clean_truth),
    5: lambda env, submitted: task5_reasoning.grade(submitted, env._annotated_records),
}


class MedicalOpenEnv:
    """
    OpenEnv-compliant environment for medical record cleaning and PHI redaction.
    """

    def __init__(self) -> None:
        self._task_id: int = 1
        self._seed: int = 42
        self._step: int = 0
        self._done: bool = False
        self._generator_id: str | None = None
        self._correlation_id: str | None = None

        # Task-specific ground truth (set on reset)
        self._dirty_records: list[DirtyRecord] = []
        self._clean_truth: list[PatientRecord] = []
        self._annotated_records: list[AnnotatedRecord] = []
        self._baseline_ml_scores: list[float] = []

        # Last grader result (for idempotent /grader endpoint)
        self._last_grade: dict[str, Any] = {}
        self._last_submitted: list[dict[str, Any]] = []
        self._history: list[dict[str, Any]] = []

    # ─────────────────────────────────────────────────────────────────────
    # OpenEnv interface
    # ─────────────────────────────────────────────────────────────────────

    def reset(self, task_id: int = 1, seed: int = 42) -> Observation:
        """Initialise a fresh episode and return the first observation."""
        if task_id not in (1, 2, 3, 4, 5):
            raise ValueError(f"task_id must be 1, 2, 3, 4, or 5 — got {task_id!r}")

        self._task_id = task_id
        self._seed = seed
        self._step = 0
        self._done = False
        self._last_grade = {}
        self._last_submitted = []
        self._history = []

        gen = EHRGenerator(seed=seed)
        self._generator_id = getattr(gen, "generator_id", None)
        n_records = 6

        if task_id == 1:
            # Use longitudinal generator for Task 1: 3 patients, 2 visits each = 6 records
            dirty, truths = gen.make_longitudinal_dirty_records(n_patients=3, visits_per_patient=2)
            self._dirty_records = dirty
            self._clean_truth = truths
            self._annotated_records = []
            observation_records = [r.model_dump() for r in dirty]

        elif task_id in (2, 3):
            annotated = gen.make_annotated_records(n=n_records)
            self._annotated_records = annotated
            self._dirty_records = []
            self._clean_truth = []
            if task_id == 3:
                self._baseline_ml_scores = (
                    task3_anonymization.compute_baseline_ml_scores(annotated)
                )
            # Expose records WITHOUT phi_tokens or clinical_keywords (grader targets only)
            observation_records = [
                {k: v for k, v in r.model_dump().items() if k not in _HIDDEN_FIELDS}
                for r in annotated
            ]

        elif task_id == 4:
            # Knowledge extraction from realistic, imperfect records.
            dirty, truths = gen.make_dirty_records(n=n_records)
            self._dirty_records = dirty
            self._clean_truth = truths
            observation_records = [
                {
                    k: v
                    for k, v in r.model_dump().items()
                    if k != "injected_flaws"
                }
                for r in dirty
            ]
        
        elif task_id == 5:
            # Contextual PII disambiguation challenge
            from .tasks.task5_reasoning import generate_ambiguous_records
            annotated = generate_ambiguous_records(seed=seed)

            self._annotated_records = annotated
            self._dirty_records = []
            self._clean_truth = []
            observation_records = [
                {k: v for k, v in r.model_dump().items() if k not in _HIDDEN_FIELDS}
                for r in annotated
            ]
        
        else:
            raise ValueError(f"Unknown task_id: {task_id}")


        obs = Observation(
            task_id=task_id,
            task_description=TASK_DESCRIPTIONS[task_id],
            records=observation_records,
            step=self._step,
            max_steps=MAX_STEPS,
            metadata={"seed": seed, "n_records": n_records, "generator_id": self._generator_id},
        )

        logger.info(
            "env.reset",
            extra={
                "task_id": task_id,
                "seed": seed,
                "generator_id": self._generator_id,
                "records": len(observation_records),
                "correlation_id": self._correlation_id,
            },
        )

        return obs

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict[str, Any]]:
        """
        Submit an action and receive (observation, reward, done, info).
        Dense reward with incremental shaping — every step returns a score plus improvement bonus.
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._step += 1
        
        # Determine records to grade based on task
        if self._task_id == 4:
            submitted_data = [k.model_dump() for k in (action.knowledge or [])]
        else:
            submitted_data = action.records or []
            
        self._last_submitted = submitted_data

        grade_result = self._grade(submitted_data)
        raw_score = float(grade_result.get("score", 0.0))
        # Primary score must stay strictly in (0, 1) for validators; rounding can yield 1.0 otherwise.
        clamped_score = Reward.clamp_score(raw_score)
        self._last_grade = {**grade_result, "score": clamped_score}

        # Calculate incremental reward shaping (use clamped scores for consistent audit trail)
        previous_score = self._history[-1]["score"] if self._history else 0.0
        improvement = clamped_score - previous_score
        
        # Incremental rewards for improvements/penalties for regressions
        improvement_bonus = 0.0
        if improvement > 0:
            improvement_bonus = min(0.1 * improvement, 0.05)  # Cap at +0.05 per step
        elif improvement < 0:
            improvement_bonus = max(0.1 * improvement, -0.025)  # Cap at -0.025 per step
        
        # Small step penalty to discourage random actions (-0.01)
        step_penalty = -0.01
        
        # Final shaped reward
        shaped_score = clamped_score + improvement_bonus + step_penalty

        # Update history for audit trail
        history_entry = {
            "step": self._step,
            "score": clamped_score,
            "shaped_score": shaped_score,
            "improvement": improvement,
            "breakdown": grade_result.get("breakdown", {}),
            "passed": grade_result.get("passed", False),
            "correlation_id": self._correlation_id,
        }
        self._history.append(history_entry)

        logger.info(
            "env.step",
            extra={
                "task_id": self._task_id,
                "seed": self._seed,
                "step": self._step,
                "score": clamped_score,
                "shaped_score": shaped_score,
                "improvement": improvement,
                "passed": grade_result.get("passed", False),
                "correlation_id": self._correlation_id,
            },
        )

        done = action.is_final or (self._step >= MAX_STEPS)
        self._done = done

        reward = Reward(
            score=clamped_score,
            breakdown={
                **grade_result.get("breakdown", {}),
                "base_score": clamped_score,
                "grader_raw_score": raw_score,
                "shaped_score": round(shaped_score, 4),   # Available for RL agents; NOT the primary signal
                "improvement_bonus": round(improvement_bonus, 4),
                "step_penalty": step_penalty,
                "improvement": round(improvement, 4),
            },
            done=done,
            info={
                **grade_result.get("info", {}),
                "passed": grade_result.get("passed", False),
                "step": self._step,
                "shaping_applied": True,
                "correlation_id": self._correlation_id,
            },
        )

        # Build next observation (same records — agent may re-submit)
        if self._task_id == 1:
            obs_records = [r.model_dump() for r in self._dirty_records]
        elif self._task_id == 4:
            obs_records = [r.model_dump() for r in self._clean_truth]
        else:
            obs_records = [
                {k: v for k, v in r.model_dump().items() if k not in _HIDDEN_FIELDS}
                for r in self._annotated_records
            ]  # clinical_keywords hidden: grader-internal only

        # [P3] Add actionable feedback hints to metadata
        metadata = {
            "seed": self._seed,
            "last_score": clamped_score,
            "passed": grade_result.get("passed", False),
        }

        if self._task_id == 1:
            per_record = grade_result.get("breakdown", {}).get("per_record", [])
            weak_fields = [
                f for rec in per_record
                for f, s in rec.get("field_scores", {}).items() if s < 0.5
            ]
            metadata["fields_needing_attention"] = list(set(weak_fields))

        elif self._task_id in (2, 3):
            leaked = [
                cat for rec in grade_result.get("breakdown", {}).get("per_record", [])
                for cat in rec.get("leaked_categories", [])
            ]
            metadata["leaked_phi_categories"] = list(set(leaked))

        elif self._task_id == 4:
            per_record = grade_result.get("per_record", [])
            metadata["low_entity_records"] = [
                r["record_id"] for r in per_record if r.get("entity_score", 1.0) < 0.5
            ]

        next_obs = Observation(
            task_id=self._task_id,
            task_description=TASK_DESCRIPTIONS[self._task_id],
            records=obs_records,
            step=self._step,
            max_steps=MAX_STEPS,
            metadata={**metadata, "generator_id": self._generator_id},
        )

        return next_obs, reward, done, reward.info

    def state(self) -> State:
        """Return a JSON-serialisable snapshot of the current episode state, including audit trail."""
        return State(
            task_id=self._task_id,
            seed=self._seed,
            step=self._step,
            max_steps=MAX_STEPS,
            done=self._done,
            last_score=self._last_grade.get("score"),
            last_breakdown=self._last_grade.get("breakdown", {}),
            passed=self._last_grade.get("passed", False),
            audit_trail=self._history,
        )

    def regrade(self) -> dict[str, Any]:
        """Re-grade the last submission (idempotent)."""
        if not self._last_submitted:
            message = "No submission yet. Call /step first."
            return {
                "detail": message,
                "error": message,
                "error_type": "state_error",
            }
        
        # Get the last graded result from history
        if not self._history:
            message = "No grading history available."
            return {
                "detail": message,
                "error": message,
                "error_type": "state_error",
            }
        
        # Regrade should mirror Reward.score from step() for idempotency.
        last_entry = self._history[-1]
        return {
            "score": last_entry.get("score", Reward.clamp_score(None)),
            "breakdown": last_entry.get("breakdown", {}),
            "passed": last_entry.get("passed", False),
            "info": {
                "step": last_entry.get("step", 0),
                "shaping_applied": True,
                "shaped_score": last_entry.get("shaped_score"),
            },
        }


    # ─────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────

    def _grade(self, submitted_records: list[dict[str, Any]]) -> dict[str, Any]:
        grader = _TASK_GRADERS.get(self._task_id)
        if grader is None:
            raise ValueError(f"Unknown task_id: {self._task_id}")
        return grader(self, submitted_records)

