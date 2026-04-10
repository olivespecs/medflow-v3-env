"""
Inference script for Medical Records OpenEnv (required filename: inference.py).

Required environment variables (as mandated by hackathon rules):
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

Optional aliases accepted for convenience:
    API_KEY        Alias for HF_TOKEN.
    OPENAI_API_KEY Alias for HF_TOKEN (used by OpenAI client).
    OPENENV_BASE_URL   Base URL of the running environment server.

The inference script uses the OpenAI Client for all LLM calls.
Stdout emits structured evaluator logs: [START], [STEP], [END].
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from typing import Any, List, Optional

import httpx

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.3-70B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or ""
ENV_BASE_URL = os.getenv("OPENENV_BASE_URL", "http://localhost:7860").rstrip("/")

SUPPORTED_TASK_IDS = [1, 2, 3, 4, 5]
MAX_STEPS = 1  # Our environment is single-step per episode (submit & done)
TEMPERATURE = 0.0
MAX_TOKENS = 4096
VERBOSE = os.getenv("INFERENCE_VERBOSE", "0") == "1"

# ---------------------------------------------------------------------------
# Logging helpers — strict [START] / [STEP] / [END] format for evaluators
# ---------------------------------------------------------------------------


def _stderr(level: str, msg: str) -> None:
    if VERBOSE:
        print(f"[{level}] {msg}", file=sys.stderr, flush=True)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_short = action if len(action) <= 200 else action[:200] + "..."
    print(
        f"[STEP] step={step} action={action_short} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# OpenAI client builder
# ---------------------------------------------------------------------------


def _build_openai_client():
    """Build OpenAI client from required environment variables."""
    try:
        from openai import OpenAI
    except ImportError:
        raise EnvironmentError(
            "openai package is required. Install with: pip install openai"
        )

    token = HF_TOKEN
    if not token:
        raise EnvironmentError(
            "Missing required environment variable: set HF_TOKEN, API_KEY, or OPENAI_API_KEY"
        )

    return OpenAI(base_url=API_BASE_URL, api_key=token)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    1: textwrap.dedent(
        """
        You are a medical data-cleaning AI. You receive synthetic patient records
        with deliberate data-quality flaws (mixed date formats, invalid ICD-10 codes,
        wrong medication units, OCR noise, missing fields).
        Fix every field to be valid and standardised.
        Return ONLY a valid JSON object with a "records" key containing an array of
        processed patient records — one per input record, in the same order.
        No conversational filler.
        """
    ).strip(),
    2: textwrap.dedent(
        """
        You are a PHI (Protected Health Information) redaction AI.
        Replace every PHI value (name, MRN, DOB, phone, email, address) with its
        category token: [REDACTED_NAME], [REDACTED_MRN], [REDACTED_DOB],
        [REDACTED_PHONE], [REDACTED_EMAIL], [REDACTED_ADDRESS].
        Do NOT remove clinical content (diagnoses, medications, symptoms).
        Return ONLY a valid JSON object with a "records" key.
        No conversational filler.
        """
    ).strip(),
    3: textwrap.dedent(
        """
        You are a medical data anonymisation AI. De-identify records while preserving
        clinical signal. Replace DOB with age_group (18-40, 41-60, 61-75, 76+).
        Redact all PHI. Remove adversarial indirect identifiers.
        Return ONLY a valid JSON object with a "records" key.
        No conversational filler.
        """
    ).strip(),
    4: textwrap.dedent(
        """
        You are a clinical knowledge extraction AI. For each patient record, extract
        clinical entities (conditions with ICD-10 codes, medications with names) and
        write a concise clinical summary.
        Return ONLY a valid JSON object with a "knowledge" key containing an array of
        objects: [{"entities": [...], "summary": "..."}, ...]
        One knowledge object per input record, in the same order.
        No conversational filler.
        """
    ).strip(),
    5: textwrap.dedent(
        """
        You are a contextual PII disambiguation AI. Redact patient/family identifiers
        only (Mr., Mrs., Ms., Miss + name, family terms). Do NOT redact provider
        names (Dr., Nurse, RN, Prof.) or facility names (Clinic, Hospital, Center).
        In ambiguous surname cases, use context to preserve providers and redact patients.
        Return ONLY a valid JSON object with a "records" key.
        No conversational filler.
        """
    ).strip(),
}


def _build_user_prompt(task_id: int, task_description: str, records: list[dict]) -> str:
    return textwrap.dedent(
        f"""
        Task ID: {task_id}
        Description: {task_description}

        Input Records ({len(records)} records):
        {json.dumps(records, ensure_ascii=False, indent=2)}

        Return exactly {len(records)} items in your output, in the same order as the input.
        Do not drop, merge, duplicate, or add items.
        """
    ).strip()


# ---------------------------------------------------------------------------
# LLM call with retry
# ---------------------------------------------------------------------------


def _call_llm(client, task_id: int, task_description: str, records: list[dict]) -> str:
    """Call the LLM and return raw text content. Returns empty string on failure."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS.get(task_id, "Solve the task.")},
        {"role": "user", "content": _build_user_prompt(task_id, task_description, records)},
    ]

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else ""
    except Exception as exc:
        _stderr("ERROR", f"LLM request failed: {exc}")
        # Retry without json_object in case the provider doesn't support it
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            text = (completion.choices[0].message.content or "").strip()
            return text if text else ""
        except Exception as exc2:
            _stderr("ERROR", f"LLM retry failed: {exc2}")
            return ""


# ---------------------------------------------------------------------------
# Payload extraction from LLM JSON output
# ---------------------------------------------------------------------------


def _extract_records(content: str, task_id: int) -> list[dict]:
    """Extract records or knowledge list from LLM JSON output."""
    if not content:
        return []

    # Try direct JSON parse
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Try to find JSON block in markdown
        import re
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(1))
            except json.JSONDecodeError:
                return []
        else:
            return []

    if not isinstance(parsed, dict):
        return []

    key = "knowledge" if task_id == 4 else "records"
    result = parsed.get(key)

    # Handle wrapper keys
    if result is None:
        for k in ("output", "data", "result", "response"):
            if k in parsed and isinstance(parsed[k], list):
                result = parsed[k]
                break

    if not isinstance(result, list):
        return []

    return result


# ---------------------------------------------------------------------------
# Run a single task through the environment
# ---------------------------------------------------------------------------


def _run_task(
    env_base_url: str,
    task_id: int,
    seed: int,
    client=None,
    model_name: str = "",
) -> dict[str, Any]:
    """Run one episode through the OpenEnv HTTP API. Returns result dict."""

    def _error(score: float = 0.0001, msg: str = "") -> dict:
        return {
            "score": score,
            "passed": False,
            "done": True,
            "error": msg,
        }

    try:
        with httpx.Client(timeout=120.0) as http:
            # 1. Reset
            try:
                resp = http.post(
                    f"{env_base_url}/reset",
                    json={"task_id": task_id, "seed": seed},
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                return _error(msg=f"Reset failed: {e}")

            obs = data.get("observation", {})
            episode_id = data.get("episode_id", "")
            task_description = obs.get("task_description", "")
            records = obs.get("records", [])
            n_records = len(records)

            # 2. Get action from LLM
            action_payload: list[dict] = []
            error_msg: Optional[str] = None

            if client is not None:
                raw = _call_llm(client, task_id, task_description, records)
                action_payload = _extract_records(raw, task_id)

                if not action_payload:
                    error_msg = "LLM output had no valid records/knowledge"
                    _stderr("WARN", f"Task {task_id}: {error_msg}")

                if action_payload and len(action_payload) != n_records:
                    error_msg = f"Wrong count: expected {n_records}, got {len(action_payload)}"
                    _stderr("WARN", f"Task {task_id}: {error_msg}")
                    action_payload = []
            else:
                return _error(msg="No LLM client provided")

            # 3. Step
            if not action_payload:
                # Submit empty action to get a grade (will score low but won't crash)
                action_json = {"is_final": True}
                if task_id == 4:
                    action_json["knowledge"] = []
                else:
                    action_json["records"] = []
            else:
                action_json = {"is_final": True}
                if task_id == 4:
                    action_json["knowledge"] = action_payload
                else:
                    action_json["records"] = action_payload

            try:
                step_resp = http.post(
                    f"{env_base_url}/step",
                    params={"episode_id": episode_id},
                    json=action_json,
                )
                step_resp.raise_for_status()
                step_data = step_resp.json()
            except Exception as e:
                return _error(msg=f"Step failed: {e}")

            reward = float(step_data.get("reward", 0.0))
            done = bool(step_data.get("done", True))
            passed = bool(step_data.get("info", {}).get("passed", False))

            if error_msg:
                return {
                    "score": reward,
                    "passed": passed,
                    "done": done,
                    "error": error_msg,
                }

            return {
                "score": reward,
                "passed": passed,
                "done": done,
                "error": None,
            }

    except Exception as e:
        _stderr("ERROR", f"Unexpected error in task {task_id}: {e}")
        return _error(msg=f"Unexpected error: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Medical Records OpenEnv inference")
    parser.add_argument(
        "--task", type=int, action="append", choices=SUPPORTED_TASK_IDS, dest="tasks",
    )
    parser.add_argument("--all", action="store_true", help="Run all 5 tasks")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env-base-url", default=ENV_BASE_URL, help="Environment server URL")
    parser.add_argument("--demo", action="store_true", help="Run deterministic local baseline (no LLM key needed)")
    args = parser.parse_args()

    task_ids = sorted(set(args.tasks or []))
    if args.all or not task_ids:
        task_ids = SUPPORTED_TASK_IDS.copy()

    env_url = args.env_base_url.rstrip("/")

    # Demo mode: run local hybrid baseline without LLM API key
    if args.demo:
        _run_demo_baseline(task_ids, args.seed)
        return

    # LLM mode: require API key
    try:
        client = _build_openai_client()
    except EnvironmentError as e:
        print(f"[ERROR] {e}", flush=True)
        sys.exit(1)

    # Log start
    task_names = {
        1: "Data Hygiene",
        2: "PHI Redaction",
        3: "Anonymisation",
        4: "Knowledge Extraction",
        5: "Contextual PII",
    }
    task_list = ", ".join(task_names.get(t, str(t)) for t in task_ids)
    log_start(task_list, "medical-records-cleaner", MODEL_NAME)

    _stderr("INFO", f"Running LLM inference ({MODEL_NAME}) for tasks: {task_ids}")

    rewards: List[float] = []
    steps_total = 0

    for task_id in task_ids:
        result = _run_task(
            env_base_url=env_url,
            task_id=task_id,
            seed=args.seed,
            client=client,
            model_name=MODEL_NAME,
        )

        score = result.get("score", 0.0)
        done = result.get("done", True)
        error = result.get("error")

        rewards.append(score)
        steps_total += 1

        action_summary = f"task{task_id}_submit"
        log_step(
            step=steps_total,
            action=action_summary,
            reward=score,
            done=done,
            error=error,
        )

    avg_score = sum(rewards) / max(len(rewards), 1)
    all_passed = all(r >= 0.5 for r in rewards)  # heuristic: >0.5 means meaningful progress

    log_end(
        success=all_passed,
        steps=steps_total,
        score=avg_score,
        rewards=rewards,
    )

    _stderr("INFO", f"Average score: {avg_score:.4f}")
    _stderr("INFO", f"Results: {json.dumps(rewards, indent=2)}")


def _run_demo_baseline(task_ids: list[int], seed: int) -> None:
    """Run the deterministic local hybrid baseline (no LLM API key needed)."""
    from src.baseline_agent import hybrid_baseline
    from src.environment import MedicalOpenEnv

    task_names = {
        1: "Data Hygiene",
        2: "PHI Redaction",
        3: "Anonymisation",
        4: "Knowledge Extraction",
        5: "Contextual PII",
    }
    task_list = ", ".join(task_names.get(t, str(t)) for t in task_ids)
    log_start(task_list, "medical-records-cleaner", "hybrid-baseline")

    _stderr("INFO", f"Running local hybrid baseline for tasks: {task_ids}")

    rewards: List[float] = []
    steps_total = 0

    for task_id in task_ids:
        try:
            env = MedicalOpenEnv()
            env.reset(task_id=task_id, seed=seed)
            result = hybrid_baseline(env)
            score = float(result.get("score", 0.0))
            passed = bool(result.get("passed", False))
        except Exception as e:
            _stderr("ERROR", f"Baseline failed for task {task_id}: {e}")
            score = 0.0001
            passed = False

        rewards.append(score)
        steps_total += 1
        log_step(
            step=steps_total,
            action=f"task{task_id}_baseline",
            reward=score,
            done=True,
            error=None if passed else f"score={score:.4f}",
        )

    avg_score = sum(rewards) / max(len(rewards), 1)
    all_passed = all(r >= 0.5 for r in rewards)

    log_end(
        success=all_passed,
        steps=steps_total,
        score=avg_score,
        rewards=rewards,
    )

    _stderr("INFO", f"Average score: {avg_score:.4f}")


if __name__ == "__main__":
    main()
