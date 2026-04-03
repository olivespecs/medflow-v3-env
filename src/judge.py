"""
Deterministic rule-based Judge for the Medical Records OpenEnv.

Evaluates a submission's reward breakdown and produces:
  - A letter grade (A / B / C / D / F)
  - A safety verdict (Safe / UNSAFE)
  - Per-dimension feedback strings
  - Regret percentage (how far from optimal)
  - A printable verdict block for the UI
"""

from __future__ import annotations
from typing import Any, Callable

# ── Grade thresholds ────────────────────────────────────────────────────────
_GRADE_THRESHOLDS = [
    (0.95, "A 🟢"),
    (0.85, "B 🟡"),
    (0.70, "C 🟠"),
    (0.50, "D 🔴"),
    (0.00, "F ⛔"),
]

TaskJudgeFn = Callable[[dict[str, Any], float, bool], dict[str, Any]]


def _letter_grade(score: float) -> str:
    for threshold, grade in _GRADE_THRESHOLDS:
        if score >= threshold:
            return grade
    return "F ⛔"


def _regret(score: float) -> str:
    regret = max(0.0, 1.0 - score) * 100
    return f"{regret:.1f}%"


# ── Task-specific feedback ──────────────────────────────────────────────────

def _judge_task1(breakdown: dict[str, Any], score: float, passed: bool) -> dict[str, Any]:
    # The Task 1 grader emits avg_dob, avg_gender, avg_icd10_codes, avg_vitals,
    # avg_medications, avg_phone, avg_email, avg_address — never "field_avg".
    # Average whichever keys are present; fall back to overall score only if none exist.
    _FIELD_AVG_KEYS = [
        "avg_dob", "avg_gender", "avg_icd10_codes", "avg_vitals",
        "avg_medications", "avg_phone", "avg_email", "avg_address",
    ]
    available = [breakdown[k] for k in _FIELD_AVG_KEYS if k in breakdown]
    field_avg = sum(available) / len(available) if available else score

    long_score = breakdown.get("longitudinal_consistency", breakdown.get("consistency_score", 0.0))

    feedback = []
    if field_avg >= 0.90:
        feedback.append("✅ Per-field corrections are excellent.")
    elif field_avg >= 0.70:
        feedback.append("🟡 Per-field corrections are good but some fields are still malformed.")
    else:
        feedback.append("🔴 Many fields still contain errors — check date formats, ICD-10 codes, and units.")

    if long_score >= 0.90:
        feedback.append("✅ Longitudinal consistency across visits is strong.")
    elif long_score >= 0.60:
        feedback.append("🟡 Some MRN records still have inconsistent DOB or gender across visits.")
    else:
        feedback.append("🔴 Longitudinal consistency is poor — apply majority-vote consensus per MRN.")

    safety = "✅ Safe — no PHI concerns in Task 1."
    return {"feedback": feedback, "safety": safety, "passed": passed}


def _judge_task2(breakdown: dict[str, Any], score: float, passed: bool) -> dict[str, Any]:
    phi_score = breakdown.get("phi_score", 0.0)
    utility_score = breakdown.get("utility_score", 0.0)
    leaked = breakdown.get("total_leaked", breakdown.get("phi_leak_penalty", 0))

    feedback = []

    # PHI coverage
    if phi_score == 1.0:
        feedback.append("✅ PHI coverage perfect — all identifiers redacted.")
    elif phi_score >= 0.80:
        feedback.append("🟡 PHI coverage near-perfect but some identifiers leaked.")
    else:
        feedback.append("🔴 CRITICAL: Significant PHI leakage detected. Check clinical notes for embedded names, emails, phones.")

    # Utility
    if utility_score >= 0.85:
        feedback.append("✅ Clinical utility well preserved — keywords and diagnoses retained.")
    elif utility_score >= 0.70:
        feedback.append("🟡 Clinical utility is acceptable but over-redaction detected.")
    else:
        feedback.append("🔴 Clinical utility too low — ICD-10 codes, medications, or notes are being over-redacted.")

    safety = (
        "✅ Safe — all PHI successfully redacted." if phi_score == 1.0
        else f"⚠️ UNSAFE — {leaked} PHI token(s) still present in output."
    )
    return {"feedback": feedback, "safety": safety, "passed": passed}


def _judge_task3(breakdown: dict[str, Any], score: float, passed: bool) -> dict[str, Any]:
    phi_score = breakdown.get("phi_score", 0.0)
    ml_score = breakdown.get("ml_utility_score", breakdown.get("ml_score", 0.0))
    fidelity = breakdown.get("fidelity_score", breakdown.get("avg_fidelity", 0.0))
    k_score = breakdown.get("k_score", breakdown.get("k_anonymity_score", 0.0))
    leaked = breakdown.get("total_leaked", breakdown.get("phi_leak_penalty", 0))

    feedback = []

    if phi_score == 1.0:
        feedback.append("✅ PHI + adversarial identifiers fully redacted.")
    elif phi_score >= 0.80:
        feedback.append("🟡 PHI almost fully redacted — check adversarial rare-disease + ZIP combos in notes.")
    else:
        feedback.append("🔴 CRITICAL: PHI or adversarial identifiers are leaking. Linkage attacks are possible.")

    if ml_score >= 0.80:
        feedback.append("✅ Downstream ML fidelity excellent — clinical signal preserved.")
    elif ml_score >= 0.60:
        feedback.append("🟡 ML fidelity meets pass bar but could be improved — keep ICD-10 chapter prefixes and medication names.")
    else:
        feedback.append("🔴 Downstream ML fidelity too low — over-anonymisation is destroying clinical signal.")

    if fidelity >= 0.80:
        feedback.append("✅ Clinical fidelity strong.")
    else:
        feedback.append("🟡 Clinical fidelity could be improved — avoid removing ICD prefixes, gender, and age-group.")

    if k_score >= 0.80:
        feedback.append("✅ k-anonymity (k=2) satisfied for most records.")
    else:
        feedback.append("🟡 Not all records satisfy k=2 on quasi-identifiers (age-group, gender, address).")

    safety = (
        "✅ Safe — full de-identification achieved." if phi_score == 1.0
        else f"⚠️ UNSAFE — {leaked} identifiers still present. Dataset is re-identifiable."
    )
    return {"feedback": feedback, "safety": safety, "passed": passed}


def _judge_task4(breakdown: dict[str, Any], score: float, passed: bool) -> dict[str, Any]:
    entity_score = breakdown.get("avg_entity_extraction", breakdown.get("entity_score", 0.0))
    code_score = breakdown.get("avg_code_precision", breakdown.get("code_precision", 0.0))
    summary_score = breakdown.get("avg_summary_fidelity", breakdown.get("summary_fidelity", 0.0))

    feedback = []

    if entity_score >= 0.85:
        feedback.append("✅ Entity extraction is excellent — conditions and medications well identified.")
    elif entity_score >= 0.70:
        feedback.append("🟡 Entity extraction acceptable — some conditions or medications missed.")
    else:
        feedback.append("🔴 Entity extraction poor — extract all ICD-10-mapped conditions and all medications from the record.")

    if code_score >= 0.85:
        feedback.append("✅ Code precision is high — ICD-10 codes correctly assigned.")
    elif code_score >= 0.60:
        feedback.append("🟡 Some codes are imprecise — ensure exact ICD-10 codes from the record are used.")
    else:
        feedback.append("🔴 Code precision low — map entities to their exact codes from the record's icd10_codes field.")

    if summary_score >= 0.50:
        feedback.append("✅ Summary fidelity meets pass bar — comprehensive summary captures key clinical facts.")
    elif summary_score >= 0.30:
        feedback.append("🟡 Summary fidelity below pass bar — include age, all diagnoses, medications, and vitals.")
    else:
        feedback.append("🔴 Summary too generic or missing key clinical facts. Include all clinical facts from the patient notes — diagnoses, medications, vitals.")

    safety = "✅ Task 4 is a knowledge extraction task — no PHI safety concerns."
    return {"feedback": feedback, "safety": safety, "passed": passed}


def _judge_task5(breakdown: dict[str, Any], score: float, passed: bool) -> dict[str, Any]:
    patient = breakdown.get("patient_phi_score", 0.0)
    provider = breakdown.get("provider_phi_score", 0.0)
    contextual = breakdown.get("contextual_accuracy", 0.0)
    overall = breakdown.get("score", score)

    feedback = []

    if patient >= 0.95:
        feedback.append("✅ Patient and family identifiers fully redacted.")
    elif patient >= 0.80:
        feedback.append("🟡 Patient/family redaction is good but a few identifiers still leak.")
    else:
        feedback.append("🔴 Patient identifiers are leaking — tighten redaction on patient/family mentions.")

    if provider >= 0.90:
        feedback.append("✅ Provider and facility identifiers correctly preserved (no over-redaction).")
    elif provider >= 0.70:
        feedback.append("🟡 Some provider/facility identifiers were redacted; preserve clinician and site names.")
    else:
        feedback.append("🔴 Over-redaction of providers/facilities — restore clinician and site context where appropriate.")

    if contextual >= 0.85:
        feedback.append("✅ Ambiguous surnames handled with strong contextual disambiguation.")
    elif contextual >= 0.65:
        feedback.append("🟡 Contextual disambiguation is partial — ambiguous surnames occasionally mishandled.")
    else:
        feedback.append("🔴 Contextual disambiguation failing — ambiguous surnames often misclassified.")

    if overall >= 0.70 and patient >= 0.80:
        feedback.append("✅ Pass threshold met (overall ≥ 0.70 and patient ≥ 0.80).")
    else:
        feedback.append("❌ Pass threshold missed — overall ≥ 0.70 and patient ≥ 0.80 required.")

    # Align safety with pass bar: overall ≥ 0.70 and patient ≥ 0.80 → Safe; otherwise Unsafe.
    if overall >= 0.70 and patient >= 0.80:
        safety = "✅ Safe — patient/family PII redacted and contextual handling meets bar."
    else:
        safety = "⚠️ UNSAFE — contextual PII risk; meet overall ≥ 0.70 and patient ≥ 0.80."

    return {"feedback": feedback, "safety": safety, "passed": passed}


# ── Helpers ─────────────────────────────────────────────────────────────────

def _apply_pass_fail_safety(safety: str, passed: bool) -> str:
    """Combine task safety verdict with pass/fail outcome.

    If the task passed, keep the task-specific safety string (PHI-aware for Tasks 2/3/5).
    If the task failed, override to Unsafe to reflect the failure status.
    """
    if passed:
        # Preserve task-level nuance; only override to unsafe when the task fails.
        return safety
    return "⚠️ UNSAFE — task failed; fix issues before trusting safety output."


def _feedback_lines(feedback: list[str]) -> str:
    return "\n".join(f"- {f}" for f in feedback)


# ── Public API ──────────────────────────────────────────────────────────────

def judge(task_id: int, score: float, breakdown: dict[str, Any], passed: bool) -> dict[str, Any]:
    """
    Evaluate a submission and return a structured verdict.

    Returns
    -------
    dict with keys:
        grade        : str   e.g. "A 🟢"
        regret       : str   e.g. "12.3%"
        safety       : str
        feedback     : list[str]
        passed       : bool
        verdict_text : str   formatted markdown block
    """
    grade = _letter_grade(score)
    regret = _regret(score)

    dispatch: dict[int, TaskJudgeFn] = {
        1: _judge_task1,
        2: _judge_task2,
        3: _judge_task3,
        4: _judge_task4,
        5: _judge_task5,
    }
    task_judge = dispatch.get(task_id, _judge_task1)
    result = task_judge(breakdown, score, passed)

    feedback_lines = _feedback_lines(result["feedback"])
    status_label = "✅ PASS" if passed else "❌ FAIL"
    safety = _apply_pass_fail_safety(result["safety"], passed)

    verdict_text = f"""## ⚖️ Judge Verdict

| Metric | Value |
|---|---|
| **Score** | `{score:.4f}` |
| **Grade** | **{grade}** |
| **Status** | {status_label} |
| **Regret from Optimal** | {regret} |
| **Safety** | {safety} |

### 📋 Dimension Feedback
{feedback_lines}
"""

    return {
        "grade": grade,
        "regret": regret,
        "safety": safety,
        "feedback": result["feedback"],
        "passed": passed,
        "verdict_text": verdict_text,
    }
