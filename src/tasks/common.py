"""Shared grader helpers for tasks 1-4.

Keep these lightweight and side-effect free to avoid grading regressions.
"""
from __future__ import annotations
from typing import Any, List


def normalize_record_list(payload: Any) -> list[dict[str, Any]]:
    """Return a safe list of dicts; fall back to empty list on bad shapes."""
    if not isinstance(payload, list):
        return []
    return [p if isinstance(p, dict) else {} for p in payload]


def validate_length_or_error(
    submitted: list[Any],
    expected_len: int,
    task_label: str,
) -> tuple[bool, dict[str, Any] | None]:
    """Ensure submission length matches ground truth; return error breakdown when not."""
    if len(submitted) != expected_len:
        return False, {
            "score": 0.0,
            "breakdown": {"error": f"{task_label}: expected {expected_len} items, got {len(submitted)}"},
            "passed": False,
            "info": {"expected_items": expected_len, "submitted_items": len(submitted)},
        }
    return True, None
