"""
Hackathon inference runner (required filename: inference.py).

MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
     
- The inference script must be named `inference.py` and placed in the root directory of the project 
- Participants must use OpenAI Client for all LLM calls using above variables 
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from openai import OpenAI

# Standard defaults from hackathon instructions
DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_ENV_BASE_URL = "http://localhost:7860"
SUPPORTED_TASK_IDS = [1, 2, 3, 4, 5]


def _emit(tag: str, payload: dict[str, Any]) -> None:
    """Emit evaluator-friendly structured logs."""
    print(f"[{tag}] {json.dumps(payload, ensure_ascii=True)}", flush=True)


def log_start(tasks: list[int], mode: str, model: str, env_base_url: str, seed: int) -> None:
    _emit(
        "START",
        {
            "tasks": tasks,
            "mode": mode,
            "model": model,
            "env_base_url": env_base_url,
            "seed": seed,
        },
    )


def log_step(task_id: int, score: float, passed: bool, done: bool, error: str | None = None) -> None:
    _emit(
        "STEP",
        {
            "task_id": task_id,
            "score": round(score, 4),
            "passed": passed,
            "done": done,
            "error": error,
        },
    )


def log_end(success: bool, tasks: list[int], average_score: float, results: list[dict[str, Any]]) -> None:
    _emit(
        "END",
        {
            "success": success,
            "tasks": tasks,
            "average_score": round(average_score, 4),
            "results": results,
        },
    )


def _import_openai() -> tuple[Any, Any]:
    """Import OpenAI client classes only when LLM mode is used."""
    try:
        from openai import BadRequestError, OpenAI
    except ImportError as e:
        raise EnvironmentError(
            "OpenAI SDK is not installed. Install dependencies via "
            "`pip install -r requirements.txt` or `pip install -e .[llm]`."
        ) from e
    return OpenAI, BadRequestError


def _coerce_list(value: Any) -> list[dict[str, Any]] | None:
    """Return value if it's a non-empty list of dicts, else None."""
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return value
    return None


def _coerce_dict(value: Any, task_id: int) -> list[dict[str, Any]] | None:
    """Pull a records/knowledge list out of a dict, checking standard and wrapper keys."""
    if not isinstance(value, dict):
        return None

    # Direct hit: the model followed instructions exactly
    key = "knowledge" if task_id == 4 else "records"
    result = _coerce_list(value.get(key))
    if result is not None:
        return result

    # Known provider wrapper keys (output, data, result, response, content, text)
    for k in ("output", "data", "result", "response", "content", "text"):
        if k in value:
            result = _coerce_list(value[k]) or _coerce_dict(value[k], task_id)
            if result is not None:
                return result

    # Bare valid object? (task 4: knowledge shape; tasks 1-3: EHR key heuristic)
    if task_id == 4:
        if "entities" in value and "summary" in value:
            return [value]
    else:
        if any(k in value for k in ("patient_name", "clinical_notes", "mrn", "dob")):
            return [value]

    # Last resort: recurse into child lists/dicts
    for child in value.values():
        if isinstance(child, (list, dict)):
            result = _coerce_list(child) or _coerce_dict(child, task_id)
            if result is not None:
                return result

    return None


def _coerce_str(value: Any, task_id: int) -> list[dict[str, Any]] | None:
    """Re-parse a string as JSON and coerce the result."""
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not (s.startswith("{") or s.startswith("[")):
        return None
    try:
        parsed = json.loads(s)
        return _coerce_list(parsed) or _coerce_dict(parsed, task_id)
    except json.JSONDecodeError:
        return None


def _extract_payload_from_response(content: str, task_id: int) -> list[dict[str, Any]]:
    """Parse model output into either a list of records or a list of knowledge objects."""
    try:
        parsed = json.loads(content)
        result = _coerce_dict(parsed, task_id) or _coerce_list(parsed) or _coerce_str(parsed, task_id)
        if result is not None:
            return result
    except Exception:
        pass

    raise ValueError(
        f"Model output did not contain a valid {'knowledge' if task_id == 4 else 'records'} list"
    )


def _build_client() -> tuple[Any, str]:
    """Build OpenAI client using required environment variables."""
    OpenAI, _ = _import_openai()

    api_base_url = os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL)
    model_name = os.getenv("MODEL_NAME")
    # Supports both HF_TOKEN and API_KEY/OPENAI_API_KEY as per instructions
    hf_token = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")

    missing = []
    if not model_name: missing.append("MODEL_NAME")
    if not hf_token: missing.append("HF_TOKEN or API_KEY/OPENAI_API_KEY")

    if missing:
        raise EnvironmentError(
            "Missing required environment variables for LLM inference: "
            + ", ".join(missing)
        )

    # Narrow optional type for static type checkers.
    assert model_name is not None

    return OpenAI(base_url=api_base_url, api_key=hf_token), model_name


def _is_response_format_unsupported(err: Exception) -> bool:
    """Detect providers that reject response_format=json_object."""
    body = getattr(err, "body", None)
    if not isinstance(body, dict):
        return False

    payload = body.get("error", body)
    if not isinstance(payload, dict):
        return False

    fields = [
        str(payload.get("param", "")).lower(),
        str(payload.get("code", "")).lower(),
        str(payload.get("type", "")).lower(),
        str(payload.get("message", "")).lower(),
    ]
    return any(
        "response_format" in field
        or "json_object" in field
        or "json_validate_failed" in field
        or "failed to validate json" in field
        or "failed_generation" in field
        for field in fields
    )


def _create_chat_completion(
    client: Any,
    model_name: str,
    messages: list[dict[str, str]],
):
    """Create a completion, retrying without JSON mode only when explicitly unsupported."""
    _, BadRequestError = _import_openai()

    try:
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
    except BadRequestError as api_err:
        if not _is_response_format_unsupported(api_err):
            raise

        print(f"[WARN] Model {model_name} rejected JSON mode, retrying without response_format")
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
        )


def _llm_process_task(
    client: Any,
    model_name: str,
    task_id: int,
    task_description: str,
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    """Ask the model to solve the task and return JSON plus raw content.

    Retries once with a stricter correction prompt when output is empty,
    unparseable, or has the wrong number of items.
    """
    system_prompt = (
        "You are a medical AI assistant. Solve the clinical data task exactly as described. "
        "Return ONLY a valid JSON object. No conversational filler."
    )
    
    task_specific_guidance = ""

    if task_id == 4:
        output_format = (
            "Return a JSON object with a 'knowledge' key containing an array of objects. "
            "Example: {\"knowledge\": [{\"entities\": [{\"text\": \"...\", \"type\": \"...\", \"code\": \"...\"}], \"summary\": \"...\"}]}"
        )
        task_specific_guidance = (
            "For each record, extract condition and medication entities and provide a concise clinical summary. "
            "Each entity must include text, type, and code."
        )
    elif task_id == 5:
        output_format = (
            "Return a JSON object with a 'records' key containing an array of processed patient records. "
            "Example: {\"records\": [{\"record_id\": \"...\", \"clinical_notes\": \"...\", \"patient_name\": \"...\", ...}]}"
        )
        task_specific_guidance = (
            "Contextual PII disambiguation: redact patient/family identifiers only. "
            "Do NOT redact provider or facility names. In ambiguous surname cases, use context "
            "(titles like Dr./RN and role cues) to preserve providers while redacting patients. "
            "Example: in 'Dr. Smith met Mr. Smith', preserve 'Dr. Smith' and redact only 'Mr. Smith'."
        )
    else:
        output_format = (
            "Return a JSON object with a 'records' key containing an array of processed patient records. "
            "Example: {\"records\": [{\"record_id\": \"...\", \"clinical_notes\": \"...\", \"patient_name\": \"...\", ...}]}"
        )

    output_key = "knowledge" if task_id == 4 else "records"
    expected_items = len(records)

    count_requirement = (
        f"Return exactly {expected_items} items in '{output_key}'. "
        "Do not drop, merge, duplicate, or add items. Preserve input order."
    )

    def _build_prompt(extra_instruction: str = "") -> str:
        suffix = f"\nAdditional Instruction: {extra_instruction}" if extra_instruction else ""
        guidance_line = (
            f"Task-Specific Guidance: {task_specific_guidance}\n" if task_specific_guidance else ""
        )
        return (
            f"Task ID: {task_id}\n"
            f"Description: {task_description}\n\n"
            f"Input Records (JSON):\n{json.dumps(records, ensure_ascii=True)}\n\n"
            f"Output Format: {output_format}\n"
            f"{guidance_line}"
            f"Count Requirement: {count_requirement}"
            f"{suffix}"
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _build_prompt()},
    ]

    try:
        # Attempt once, then do one strict corrective retry on malformed output.
        for attempt in (1, 2):
            response = _create_chat_completion(
                client=client,
                model_name=model_name,
                messages=messages,
            )

            content = (response.choices[0].message.content or "").strip()
            if not content:
                if attempt == 2:
                    return [], ""

                print(f"[WARN] Task {task_id} empty model output (attempt {attempt}); retrying once")
                messages = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": _build_prompt(
                            "Your previous response was empty. Return ONLY valid JSON that satisfies all format and count requirements."
                        ),
                    },
                ]
                continue

            try:
                payload = _extract_payload_from_response(content, task_id)
            except Exception:
                payload = []

            if payload and len(payload) == expected_items:
                return payload, content

            if attempt == 2:
                return [], content

            print(
                f"[WARN] Task {task_id} wrong JSON shape/count (attempt {attempt}): "
                f"expected {expected_items}, got {len(payload)}; retrying once"
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": _build_prompt(
                        "Your previous response had the wrong JSON schema or item count. "
                        "Return ONLY valid JSON with the required top-level key and exact item count."
                    ),
                },
            ]

        return [], ""
    except Exception as e:
        print(f"[WARN] LLM inference failed for task {task_id}: {e}")
        return [], ""


def _run_task_via_api(
    env_base_url: str,
    task_id: int,
    seed: int,
    client: Any | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Run one task episode through the OpenEnv HTTP API."""
    with httpx.Client(timeout=120.0) as http:
        # 1. Reset
        reset_resp = http.post(
            f"{env_base_url}/reset",
            json={"task_id": task_id, "seed": seed},
        )
        reset_resp.raise_for_status()
        obs = reset_resp.json()["observation"]

        task_description = obs["task_description"]
        records = obs["records"]

        # 2. Process (LLM)
        if not client or not model_name:
            raise ValueError("LLM mode requires client and model_name")
            
        processed_payload, raw_content = _llm_process_task(
            client=client,
            model_name=model_name,
            task_id=task_id,
            task_description=task_description,
            records=records,
        )

        expected_items = len(records)

        # 3a. Harden: reject wrong-length payloads before sending to grader.
        if processed_payload and len(processed_payload) != expected_items:
            redacted_len = len(raw_content or "")
            print(f"\n[ERROR] Task {task_id} LLM output had wrong item count.")
            print(
                f"[INFO] Expected {expected_items}, got {len(processed_payload)}. "
                f"Raw model output suppressed (len={redacted_len} chars) to avoid leaking content."
            )
            return {
                "task_id": task_id,
                "score": 0.0,
                "breakdown": {},
                "passed": False,
                "done": True,
                "error": f"Wrong item count: expected {expected_items}, got {len(processed_payload)}",
                "model_output_len": redacted_len,
            }

        # 3b. Harden: Handle empty or malformed payloads before posting
        if not processed_payload:
            redacted_len = len(raw_content or "")
            print(f"\n[ERROR] Task {task_id} LLM output was empty or malformed.")
            print(f"[INFO] Raw model output suppressed (len={redacted_len} chars) to avoid leaking content.")
            return {
                "task_id": task_id,
                "score": 0.0,
                "breakdown": {},
                "passed": False,
                "done": True,
                "error": "Empty/Malformed model payload",
                "model_output_len": redacted_len,
            }

        # 4. Step
        action_json: dict[str, Any] = {"is_final": True}
        if task_id == 4:
            action_json["knowledge"] = processed_payload
        else:
            action_json["records"] = processed_payload

        step_resp = http.post(
            f"{env_base_url}/step",
            params={"episode_id": reset_resp.json()["episode_id"]},
            json=action_json,
        )
        step_resp.raise_for_status()
        step_payload = step_resp.json()

    return {
        "task_id": task_id,
        "score": float(step_payload.get("reward", 0.0)),
        "breakdown": step_payload.get("info", {}).get("breakdown", {}),
        "passed": bool(step_payload.get("info", {}).get("passed", False)),
        "done": bool(step_payload.get("done", False)),
    }


def _run_demo_local(task_ids: list[int], seed: int) -> list[dict[str, Any]]:
    """Deterministic local fallback for quick validation."""
    from src.baseline_agent import hybrid_baseline
    from src.environment import MedicalOpenEnv

    results: list[dict[str, Any]] = []
    for task_id in task_ids:
        env = MedicalOpenEnv()
        env.reset(task_id=task_id, seed=seed)
        result = hybrid_baseline(env)
        results.append(
            {
                "task_id": task_id,
                "score": float(result.get("score", 0.0)),
                "breakdown": result.get("breakdown", {}),
                "passed": bool(result.get("passed", False)),
                "done": True,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenEnv inference runner")
    parser.add_argument("--task", type=int, action="append", choices=SUPPORTED_TASK_IDS, dest="tasks")
    parser.add_argument("--all", action="store_true", help="Run all tasks")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env-base-url", default=os.getenv("OPENENV_BASE_URL", DEFAULT_ENV_BASE_URL))
    parser.add_argument("--demo", action="store_true", help="Run deterministic local baseline")
    args = parser.parse_args()

    task_ids = sorted(set(args.tasks or []))
    if args.all or not task_ids:
        task_ids = SUPPORTED_TASK_IDS.copy()

    # Check if LLM environment variables are set
    has_llm = bool(os.getenv("MODEL_NAME") and (os.getenv("HF_TOKEN") or os.getenv("API_KEY")))
    
    # Auto-select mode: demo if no LLM vars, LLM inference if vars are present
    mode = "demo" if (args.demo or not has_llm) else "llm"
    model_for_logs = os.getenv("MODEL_NAME", "hybrid-baseline") if mode == "llm" else "hybrid-baseline"
    log_start(task_ids, mode, model_for_logs, args.env_base_url.rstrip("/"), int(args.seed))

    if args.demo or not has_llm:
        if not args.demo:
            print("[INFO] No LLM env vars found — running local deterministic baseline")
        else:
            print(f"[INFO] Running local demo for tasks: {task_ids}")
        results = _run_demo_local(task_ids=task_ids, seed=args.seed)
        for r in results:
            log_step(
                task_id=int(r.get("task_id", -1)),
                score=float(r.get("score", 0.0)),
                passed=bool(r.get("passed", False)),
                done=bool(r.get("done", True)),
                error=r.get("error"),
            )
    else:
        client, model_name = _build_client()
        print(f"[INFO] Running LLM inference ({model_name}) for tasks: {task_ids}")
        results: list[dict[str, Any]] = []
        for task_id in task_ids:
            result = _run_task_via_api(
                env_base_url=args.env_base_url.rstrip("/"),
                task_id=task_id,
                seed=args.seed,
                client=client,
                model_name=model_name,
            )
            results.append(result)
            log_step(
                task_id=int(result.get("task_id", task_id)),
                score=float(result.get("score", 0.0)),
                passed=bool(result.get("passed", False)),
                done=bool(result.get("done", True)),
                error=result.get("error"),
            )

    print("\n" + "="*50)
    print(json.dumps({"results": results}, indent=2))
    avg = sum(r["score"] for r in results) / max(len(results), 1)
    log_end(success=all(bool(r.get("passed", False)) for r in results), tasks=task_ids, average_score=avg, results=results)
    print(f"\nAverage score: {avg:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()
