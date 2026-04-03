"""
MCP (Model Context Protocol) server for the Medical Records OpenEnv.
Standardized tool access for agents as per OpenEnv RFC 003.
"""

from __future__ import annotations

from typing import Any, Callable
import uuid
from fastapi import HTTPException
from fastmcp import FastMCP

from . import api as api_runtime
from .environment import MedicalOpenEnv
from .models import Action

mcp = FastMCP("Medical Records OpenEnv")

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "records": {"type": "array", "items": {"type": "object"}},
        "knowledge": {"type": "array", "items": {"type": "object"}},
        "is_final": {"type": "boolean"},
    },
}

ERROR_SCHEMA = {
    "type": "object",
    "properties": {
        "detail": {"type": ["string", "object"]},
        "error": {"type": ["string", "null"]},
        "error_type": {"type": "string"},
        "task_id": {"type": ["integer", "null"]},
        "episode_id": {"type": ["string", "null"]},
    },
}

TASKS = [
    {
        "id": 1,
        "name": "Data Hygiene & Standardisation",
        "pass_bar": "score >= 0.85",
        "description": "Fix messy EHR records: normalise dates, ICD-10, meds, and missing contact fields.",
    },
    {
        "id": 2,
        "name": "PHI Detection & Redaction",
        "pass_bar": "phi_score == 1.0 AND utility_score >= 0.8",
        "description": "Redact PHI across structured fields and notes using typed tokens.",
    },
    {
        "id": 3,
        "name": "Full Anonymisation + Utility",
        "pass_bar": "phi_score == 1.0 AND ml_utility_score >= 0.60",
        "description": "Anonymise with age buckets, adversarial scrub, and utility retention.",
    },
    {
        "id": 4,
        "name": "Clinical Knowledge Extraction",
        "pass_bar": "entity_extraction >= 0.75 AND summary_fidelity >= 0.50",
        "description": "Extract entities and summaries aligned to clinical notes.",
    },
    {
        "id": 5,
        "name": "Contextual PII Disambiguation",
        "pass_bar": "overall_score >= 0.70 AND patient_phi_score >= 0.80",
        "description": "Redact patient/family mentions while preserving providers/facilities.",
    },
]

# ---------------------------------------------------------------------------
# Episode store (shared with API)
# ---------------------------------------------------------------------------

MAX_MCP_EPISODES = api_runtime.MAX_EPISODES
MCP_EPISODE_TTL_SECONDS = api_runtime.EPISODE_TTL_SECONDS
_mcp_episodes = api_runtime._episodes
_mcp_episodes_lock = api_runtime._episodes_lock
MCPEpisodeEntry = api_runtime.EpisodeEntry


def _error_payload(
    detail: str,
    *,
    error_type: str = "mcp_error",
    task_id: int | None = None,
    episode_id: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "detail": detail,
        "error": detail,
        "error_type": error_type,
        "task_id": task_id,
        "episode_id": episode_id,
    }
    return payload


def _get_mcp_episode(episode_id: str) -> MCPEpisodeEntry:
    """Resolve an episode_id via the shared API store."""
    try:
        validated_episode_id = api_runtime._validate_episode_id(episode_id)
        return api_runtime._get_episode(validated_episode_id)
    except HTTPException as e:
        raise ValueError(str(e.detail)) from e


def _purge_mcp_expired() -> None:
    """Remove expired sessions from the shared store."""
    api_runtime._purge_expired()


def _validate_record_structure(record: dict, task_id: int) -> tuple[bool, str]:
    required = {
        1: ["record_id"],
        2: ["record_id"],
        3: ["record_id"],
        4: [],
        5: ["record_id", "clinical_notes"],
    }
    fields = required.get(task_id, [])
    missing = [f for f in fields if f not in record]
    if missing:
        return False, f"Missing required fields: {', '.join(missing)}"

    if task_id == 1:
        if "icd10_codes" in record and not isinstance(record.get("icd10_codes"), list):
            return False, "Field 'icd10_codes' must be a list"
        if "medications" in record and not isinstance(record.get("medications"), list):
            return False, "Field 'medications' must be a list"

    if task_id in (2, 3):
        notes = record.get("clinical_notes", "")
        if notes is not None and not isinstance(notes, str):
            return False, "Field 'clinical_notes' must be a string"
        if isinstance(notes, str) and len(notes) > api_runtime.MAX_NOTES_LENGTH:
            return False, (
                f"clinical_notes exceeds {api_runtime.MAX_NOTES_LENGTH} chars "
                f"({len(notes)} submitted). Truncate before submitting."
            )

    if task_id == 5:
        notes = record.get("clinical_notes")
        if not isinstance(notes, str):
            return False, "Field 'clinical_notes' must be a string"
        if len(notes) > api_runtime.MAX_NOTES_LENGTH:
            return False, (
                f"clinical_notes exceeds {api_runtime.MAX_NOTES_LENGTH} chars "
                f"({len(notes)} submitted). Truncate before submitting."
            )

    return True, ""


def _validate_knowledge_structure(knowledge: dict) -> tuple[bool, str]:
    if not isinstance(knowledge, dict):
        return False, "Knowledge object must be a dictionary"
    if "entities" not in knowledge or "summary" not in knowledge:
        return False, "Knowledge must include 'entities' and 'summary'"
    if not isinstance(knowledge.get("entities"), list):
        return False, "Field 'entities' must be a list"
    if not isinstance(knowledge.get("summary"), str):
        return False, "Field 'summary' must be a string"
    return True, ""


def _validate_mcp_records_payload(
    task_id: int,
    episode_id: str,
    records: list[dict[str, Any]] | None,
    knowledge: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    del knowledge
    submitted_records = records or []
    for i, record in enumerate(submitted_records):
        valid, msg = _validate_record_structure(record, task_id)
        if not valid:
            return _error_payload(
                f"Record[{i}]: {msg}",
                error_type="validation_error",
                task_id=task_id,
                episode_id=episode_id,
            )
    return None


def _validate_mcp_knowledge_payload(
    task_id: int,
    episode_id: str,
    records: list[dict[str, Any]] | None,
    knowledge: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    del records
    submitted_knowledge = knowledge or []
    for i, knowledge_obj in enumerate(submitted_knowledge):
        valid, msg = _validate_knowledge_structure(knowledge_obj)
        if not valid:
            return _error_payload(
                f"Knowledge[{i}]: {msg}",
                error_type="validation_error",
                task_id=task_id,
                episode_id=episode_id,
            )
    return None


MCPPayloadValidator = Callable[
    [int, str, list[dict[str, Any]] | None, list[dict[str, Any]] | None],
    dict[str, Any] | None,
]

_MCP_TASK_PAYLOAD_VALIDATORS: dict[int, MCPPayloadValidator] = {
    1: _validate_mcp_records_payload,
    2: _validate_mcp_records_payload,
    3: _validate_mcp_records_payload,
    4: _validate_mcp_knowledge_payload,
    5: _validate_mcp_records_payload,
}


@mcp.tool()
def reset(task_id: int = 1, seed: int = 42) -> dict[str, Any]:
    """
    Start a new episode in the environment.
    
    Args:
        task_id: 1=hygiene, 2=redaction, 3=anonymization, 4=knowledge
        seed: Random seed for synthetic data generation
    """
    with _mcp_episodes_lock:
        _purge_mcp_expired()
        if len(_mcp_episodes) >= MAX_MCP_EPISODES:
            return _error_payload(
                "Too many active MCP episodes. Please wait or try again later.",
                error_type="capacity_error",
                task_id=task_id,
            )

    env = MedicalOpenEnv()
    obs = env.reset(task_id=task_id, seed=seed)
    episode_id = str(uuid.uuid4())
    
    with _mcp_episodes_lock:
        _mcp_episodes[episode_id] = MCPEpisodeEntry(env=env)

    if api_runtime._persistence_store:
        try:
            api_runtime._persistence_store.save_episode(episode_id, env, task_id, seed)
        except Exception:
            # Non-fatal in MCP mode; in-memory episode still exists.
            pass
    
    return {
        "episode_id": episode_id,
        "observation": obs.model_dump(),
    }

@mcp.tool()
def step(
    episode_id: str, 
    records: list[dict[str, Any]] | None = None, 
    knowledge: list[dict[str, Any]] | None = None,
    is_final: bool = False
) -> dict[str, Any]:
    """
    Submit processed records or knowledge and advance the episode.

    Args:
        episode_id: The ID returned from reset()
        records: Processed patient records (Tasks 1–3); ignored for Task 4 if knowledge is set
        knowledge: Task 4 — list of {entities, summary} objects, same order as observation.records
        is_final: Set True if this is your final submission
    """
    try:
        entry = _get_mcp_episode(episode_id)
    except ValueError as e:
        return _error_payload(str(e), episode_id=episode_id)
    
    with entry.lock:
        task_id = entry.env._task_id
        payload_validator = _MCP_TASK_PAYLOAD_VALIDATORS.get(task_id)
        if payload_validator is None:
            return _error_payload(
                f"Unsupported task_id: {task_id}",
                error_type="validation_error",
                task_id=task_id,
                episode_id=episode_id,
            )

        validation_error = payload_validator(task_id, episode_id, records, knowledge)
        if validation_error is not None:
            return validation_error

        action = Action(records=records, knowledge=knowledge, is_final=is_final)
        next_obs, reward, done, info = entry.env.step(action)
    
    return {
        "episode_id": episode_id,
        "observation": next_obs.model_dump(),
        "reward": float(reward.score),
        "done": done,
        "info": {
            **info,
            "breakdown": reward.breakdown
        }
    }

@mcp.tool()
def state(episode_id: str) -> dict[str, Any]:
    """Get the current state and audit trail of an episode."""
    try:
        entry = _get_mcp_episode(episode_id)
    except ValueError as e:
        return _error_payload(str(e), episode_id=episode_id)
    
    with entry.lock:
        state_model = entry.env.state()
        return state_model.model_dump()


@mcp.tool()
def export(episode_id: str) -> dict[str, Any]:
    """Export full episode snapshot with audit trail and reward trend."""
    try:
        entry = _get_mcp_episode(episode_id)
    except ValueError as e:
        return _error_payload(str(e), episode_id=episode_id)

    with entry.lock:
        state_model = entry.env.state()
        state = state_model.model_dump()
        reward_trend = [
            {"step": h["step"], "score": h["score"], "passed": h.get("passed", False)}
            for h in state.get("audit_trail", [])
        ]
        return {
            "episode_id": episode_id,
            **state,
            "reward_trend": reward_trend,
        }


@mcp.tool()
def tasks() -> dict[str, Any]:
    """List available tasks, pass bars, and action schema."""
    return {"tasks": TASKS, "action_schema": ACTION_SCHEMA, "error_schema": ERROR_SCHEMA}


@mcp.tool()
def schema() -> dict[str, Any]:
    """Return schemas for action, observation (sample), reward breakdown, and errors."""
    env = MedicalOpenEnv()
    obs = env.reset(task_id=1, seed=42)
    _, reward, _, info = env.step(Action(records=obs.records, is_final=True))
    return {
        "action_schema": ACTION_SCHEMA,
        "observation_example": obs.model_dump(),
        "reward_example": {"score": reward.score, "breakdown": reward.breakdown, "info": info},
        "error_schema": ERROR_SCHEMA,
    }

