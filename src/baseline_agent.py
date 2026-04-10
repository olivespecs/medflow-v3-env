"""
Hybrid (Rule + Local ML) baseline agent for the Medical Records OpenEnv.
Combines deterministic cleaning rules with PyTorch-based NER for high-fidelity PHI redaction.
Used by /baseline endpoint and --demo flag in baseline.py.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any
import logging
import os

from .environment import MedicalOpenEnv
from .models import Action
from .record_processors import (
    _fix_record_task1,
    _redact_record as _redact_record_core,
    _anonymise_record as _anonymise_record_core,
    _extract_knowledge_rule_based as _extract_knowledge_core,
    _redact_contextual_phi as _redact_contextual_core,
)
from .ner_agent import NER_CONFIDENCE_FINAL

logger = logging.getLogger(__name__)

SAFE_MODE = os.getenv("BASELINE_SAFE_MODE", "0").lower() in {"1", "true", "yes"}


# ---------------------------------------------------------------------------
# Global Hybrid Agent Instance (Lazy Loading)
# ---------------------------------------------------------------------------

_HYBRID_NER_AGENT = None

def get_ner_agent(force_safe_mode: bool | None = None):
    """Lazy initialize the NER agent with safe-mode awareness."""
    global _HYBRID_NER_AGENT
    safe_mode = force_safe_mode if force_safe_mode is not None else SAFE_MODE

    if _HYBRID_NER_AGENT is None or getattr(_HYBRID_NER_AGENT, "safe_mode", None) != safe_mode:
        from .ner_agent import LocalNERAgent
        _HYBRID_NER_AGENT = LocalNERAgent(safe_mode=safe_mode)
    return _HYBRID_NER_AGENT


def _ner_debug(agent: Any) -> dict[str, Any]:
    if agent is None:
        return {"enabled": False, "safe_mode": SAFE_MODE, "reason": "agent_uninitialized"}
    if hasattr(agent, "diagnostics"):
        return agent.diagnostics()
    return {
        "enabled": getattr(agent, "nlp", None) is not None,
        "safe_mode": getattr(agent, "safe_mode", SAFE_MODE),
        "device": getattr(agent, "device", None),
        "reason": getattr(agent, "disabled_reason", None),
    }


# ---------------------------------------------------------------------------
# PHI Validator & Safety Net
# ---------------------------------------------------------------------------

def _final_phi_safety_pass(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Final PHI validation pass using the dbmdz/bert-large-cased-finetuned-conll03-english model.
    Acts as a final safety net for all string fields in the PROCESSED records
    to catch any leaked PHI that deterministic rules or initial passes missed.
    Used for Tasks 2 and 3.
    """
    agent = get_ner_agent()
    if agent.nlp is None:
        return records

    processed = []
    for record in records:
        rec = deepcopy(record)
        for key, value in rec.items():
            # Scan all string fields (notes, addresses, names, etc.)
            if isinstance(value, str) and len(value) > 2:
                # Only run on fields that aren't already purely a redaction token
                if not (value.startswith("[REDACTED_") and value.endswith("]")):
                    # High confidence threshold for the final safety net to avoid over-redaction
                    # Uses dbmdz/bert-large-cased-finetuned-conll03-english as a final validator
                    rec[key] = agent.redact_text(value, confidence_threshold=NER_CONFIDENCE_FINAL)
        processed.append(rec)
    return processed


# ---------------------------------------------------------------------------
# Task 1 — Hygiene rule-based fixes (imported from record_processors)
# ---------------------------------------------------------------------------
# _fix_record_task1 is imported from record_processors at module level


# ---------------------------------------------------------------------------
# Task 2 — PHI redaction (wrapper using record_processors + NER)
# ---------------------------------------------------------------------------
# REDACTION_MAP and NOTES_PHI_FIELDS are imported from record_processors


def _redact_record(record: dict[str, Any]) -> dict[str, Any]:
    """Redact all known PHI fields and scrub notes with NER enhancement."""
    return _redact_record_core(record, ner_agent=get_ner_agent())


# ---------------------------------------------------------------------------
# Task 3 — Anonymisation (wrapper using record_processors + NER)
# ---------------------------------------------------------------------------


def _anonymise_record(record: dict[str, Any]) -> dict[str, Any]:
    """Pseudonymise record: redact PHI, bucket age, preserve clinical fields."""
    return _anonymise_record_core(record, ner_agent=get_ner_agent())


# ---------------------------------------------------------------------------
# Task 4 — Clinical Knowledge Extraction (wrapper using record_processors + NER)
# ---------------------------------------------------------------------------


def _extract_knowledge_rule_based(record: dict[str, Any]) -> dict[str, Any]:
    """Extract entities and generate summary using rule + ML hybrid approach."""
    return _extract_knowledge_core(record, ner_agent=get_ner_agent())


# ---------------------------------------------------------------------------
# Task 5 — Contextual PII Disambiguation (wrapper)
# ---------------------------------------------------------------------------


def _redact_contextual_phi(record: dict[str, Any]) -> dict[str, Any]:
    """Decide which mentions are patient PII based on context."""
    # We do NOT use the final safety net pass for Task 5,
    # as it would over-redact providers we need to keep.
    return _redact_contextual_core(record)


# ---------------------------------------------------------------------------
# Hybrid Baseline runner
# ---------------------------------------------------------------------------

def hybrid_baseline(env: MedicalOpenEnv) -> dict[str, Any]:
    """
    Run the hybrid (rule + Local ML) baseline agent on the current env episode.
    Resets the env to its last task_id + seed, runs one step, returns grade.
    """
    task_id = env._task_id
    seed = env._seed
    agent = get_ner_agent()

    obs = env.reset(task_id=task_id, seed=seed)

    if task_id == 1:
        # Task 1: Rule-based fixes with longitudinal consistency resolution
        # Collect ALL values per MRN to pick the most common one (Majority Vote)
        from collections import Counter
        mrn_dob_stats: dict[str, list[str]] = {}
        mrn_gender_stats: dict[str, list[str]] = {}
        mrn_name_stats: dict[str, list[str]] = {}
        
        processed_initial = []
        
        # First pass: standard cleaning and gather all variants per MRN
        for r in obs.records:
            cleaned = _fix_record_task1(r)
            mrn = cleaned.get("mrn")
            if mrn:
                if mrn not in mrn_dob_stats:
                    mrn_dob_stats[mrn] = []
                    mrn_gender_stats[mrn] = []
                    mrn_name_stats[mrn] = []
                
                if cleaned.get("dob"): mrn_dob_stats[mrn].append(cleaned["dob"])
                if cleaned.get("gender"): mrn_gender_stats[mrn].append(cleaned["gender"])
                if cleaned.get("patient_name"): mrn_name_stats[mrn].append(cleaned["patient_name"])
            
            processed_initial.append(cleaned)
            
        # Second pass: pick the most common value (majority) per MRN
        mrn_consensus: dict[str, dict[str, Any]] = {}
        for mrn in mrn_dob_stats:
            consensus = {}
            if mrn_dob_stats[mrn]:
                consensus["dob"] = Counter(mrn_dob_stats[mrn]).most_common(1)[0][0]
            if mrn_gender_stats[mrn]:
                consensus["gender"] = Counter(mrn_gender_stats[mrn]).most_common(1)[0][0]
            if mrn_name_stats[mrn]:
                consensus["patient_name"] = Counter(mrn_name_stats[mrn]).most_common(1)[0][0]
            mrn_consensus[mrn] = consensus

        # Third pass: enforce consensus across all records for the same MRN
        for r in processed_initial:
            mrn = r.get("mrn")
            if mrn and mrn in mrn_consensus:
                base = mrn_consensus[mrn]
                if "dob" in base: r["dob"] = base["dob"]
                if "gender" in base: r["gender"] = base["gender"]
                if "patient_name" in base: r["patient_name"] = base["patient_name"]
        
        action = Action(records=processed_initial, is_final=True)
                
    elif task_id == 2:
        # Task 2: Standard redaction + ML pass on notes
        processed = [_redact_record(r) for r in obs.records]
        # Final safety net pass on ALL fields
        processed = _final_phi_safety_pass(processed)
        action = Action(records=processed, is_final=True)
    elif task_id == 3:
        # Task 3: Anonymization + targeted adversarial scrubbing
        processed = [_anonymise_record(r) for r in obs.records]
        # Final safety net pass on ALL fields (names, locations in address, etc.)
        processed = _final_phi_safety_pass(processed)
        action = Action(records=processed, is_final=True)
    elif task_id == 4:
        processed = [_extract_knowledge_rule_based(r) for r in obs.records]
        # For Task 4, we submit knowledge objects
        action = Action(knowledge=processed, is_final=True)
    elif task_id == 5:
        # Task 5: Contextual PII disambiguation (no safety pass to avoid over-redaction)
        processed = [_redact_contextual_phi(r) for r in obs.records]
        action = Action(records=processed, is_final=True)
    else:
        # Fallback
        action = Action(records=obs.records, is_final=True)

    _, reward, _, info = env.step(action)

    reward_trace = deepcopy(env._history)
    debug = {
        "ner": _ner_debug(agent),
        "safe_mode": getattr(agent, "safe_mode", SAFE_MODE),
        "generator_id": obs.metadata.get("generator_id"),
        "records": len(obs.records),
        "seed": seed,
        "reward_trace": reward_trace,
        "correlation_id": env._correlation_id,
    }

    return {
        "task_id": task_id,
        "agent": "hybrid (rules + Local ML)",
        "score": reward.score,
        "breakdown": reward.breakdown,
        "passed": info.get("passed", False),
        "info": info,
        "debug": debug,
    }
