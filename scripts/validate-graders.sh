#!/usr/bin/env bash
#
# validate-graders.sh — Grader metadata validator
#
# Usage:
#   ./scripts/validate-graders.sh <ping_url>
#
# Optional env vars:
#   MIN_GRADER_TASKS (default: 3)
#   PYTHON_BIN       (default: python3 or python)

set -uo pipefail

PING_URL="${1:-}"
MIN_GRADER_TASKS="${MIN_GRADER_TASKS:-3}"

if [ -z "$PING_URL" ]; then
  printf "Usage: %s <ping_url>\n" "$0"
  exit 1
fi

PING_URL="${PING_URL%/}"

if ! command -v curl >/dev/null 2>&1; then
  printf "FAILED -- curl command not found\n"
  exit 1
fi

if [ -z "${PYTHON_BIN:-}" ]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    printf "FAILED -- python interpreter not found (tried python3 and python)\n"
    exit 1
  fi
fi

fetch_json() {
  local endpoint="$1"
  local out_file="$2"
  local code

  code=$(curl -sS -o "$out_file" -w "%{http_code}" "$PING_URL$endpoint" --max-time 30 || printf "000")
  if [ "$code" != "200" ]; then
    printf "FAILED -- %s returned HTTP %s (expected 200)\n" "$endpoint" "$code"
    return 1
  fi

  return 0
}

TMP_TASKS=$(mktemp)
TMP_CONTRACT=$(mktemp)
trap 'rm -f "$TMP_TASKS" "$TMP_CONTRACT"' EXIT

printf "Step 1/2: Fetching task metadata from %s\n" "$PING_URL"
fetch_json "/tasks" "$TMP_TASKS" || exit 1
fetch_json "/contract" "$TMP_CONTRACT" || exit 1

printf "Step 2/2: Evaluating grader metadata coverage\n"

"$PYTHON_BIN" - "$TMP_TASKS" "$TMP_CONTRACT" "$MIN_GRADER_TASKS" <<'PY'
import json
import sys
from typing import Any


def has_value(value: Any) -> bool:
    return value not in (None, "", {}, [])


def parse_tasks(path: str, label: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise SystemExit(f"FAILED -- {label} returned non-object JSON")

    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        raise SystemExit(f"FAILED -- {label} missing tasks[] array")

    clean: list[dict[str, Any]] = []
    for t in tasks:
        if isinstance(t, dict):
            clean.append(t)
    return clean


def summarize(label: str, tasks: list[dict[str, Any]]) -> tuple[int, int, int, int, list[Any]]:
    explicit_grader = [t for t in tasks if has_value(t.get("grader"))]
    grader_formula = [t for t in tasks if has_value(t.get("grader_formula"))]
    with_any = [t for t in tasks if has_value(t.get("grader")) or has_value(t.get("grader_formula"))]
    missing = [t.get("id", "?") for t in tasks if not (has_value(t.get("grader")) or has_value(t.get("grader_formula")))]

    total = len(tasks)
    print(
        f"{label}: "
        f"task_count={total}, "
        f"explicit_grader_count={len(explicit_grader)}, "
        f"grader_formula_count={len(grader_formula)}, "
        f"with_any_grader_metadata={len(with_any)}"
    )
    if missing:
        print(f"{label}: missing_grader_metadata_task_ids={missing}")

    return total, len(explicit_grader), len(grader_formula), len(with_any), missing


def main() -> int:
    tasks_tasks = parse_tasks(sys.argv[1], "/tasks")
    tasks_contract = parse_tasks(sys.argv[2], "/contract")
    min_required = int(sys.argv[3])

    _, explicit_tasks, _, _, _ = summarize("/tasks", tasks_tasks)
    _, explicit_contract, _, _, _ = summarize("/contract", tasks_contract)

    if explicit_tasks < min_required or explicit_contract < min_required:
        print(f"FAILED -- Not enough tasks with explicit graders (required_min={min_required})")
        return 1

    print(f"PASSED -- Explicit grader metadata threshold met (required_min={min_required})")
    return 0


raise SystemExit(main())
PY

status=$?
if [ "$status" -ne 0 ]; then
  exit "$status"
fi

printf "All grader checks passed.\n"
