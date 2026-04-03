"""
Medical Records OpenEnv — Gradio Dashboard
==========================================
5-tab interactive dashboard inspired by the best-in-class OpenEnv UI patterns.

Tabs
----
1. 🎮  Pipeline           — run any task with any agent, see before/after + verdict
2. 🤖  Multi-Task Bench   — hybrid baseline across all 5 tasks at once
3. 🔬  Robustness Sweep   — run the same task over multiple seeds, show variance
4. 📖  Scoring Guide      — task formulas, pass bars, PHI categories, adversarial rules
5. ℹ️  About              — architecture, key design decisions, API endpoints
"""

import gradio as gr
import json
import os
import pandas as pd
from typing import Any

from .baseline_agent import (
    _anonymise_record,
    _extract_knowledge_rule_based,
    _fix_record_task1,
    _redact_record,
    _redact_contextual_phi,
)
from .environment import MedicalOpenEnv
from .judge import judge
from .models import Action
from .utils import export_to_fhir

# ── Constants ────────────────────────────────────────────────────────────────

_OPENAI_AGENT = "OpenAI / LLM (MODEL_NAME env)"
_HYBRID_AGENT = "Hybrid (Rules + BERT NER)"

_TASK_NAMES = {
    1: "Task 1 — Data Hygiene & Standardisation",
    2: "Task 2 — PHI Detection & Redaction",
    3: "Task 3 — Full Anonymisation + Utility",
    4: "Task 4 — Clinical Knowledge Extraction",
    5: "Task 5 — Contextual PII Disambiguation",
}

_PASS_BARS = {
    1: "score ≥ 0.85",
    2: "phi_score = 1.0  AND  utility_score ≥ 0.80",
    3: "phi_score = 1.0  AND  ml_utility_score ≥ 0.60",
    4: "entity_extraction ≥ 0.75  AND  summary_fidelity ≥ 0.50",
    5: "overall_score ≥ 0.70  AND  patient_phi_score ≥ 0.80",
}

_FORMULAS = {
    1: "per_field_avg × 0.8 + longitudinal_consistency × 0.2",
    2: "phi_score × 0.6 + utility_score × 0.4",
    3: "avg_phi × 0.4 + avg_ml_utility × 0.3 + avg_fidelity × 0.2 + k_score × 0.1",
    4: "entity_extraction × 0.4 + code_precision × 0.3 + summary_fidelity × 0.3",
    5: "patient_phi × 0.5 + provider_phi × 0.3 + contextual_accuracy × 0.2",
}

# ── NER agent (lazy) ─────────────────────────────────────────────────────────

_ner_agent = None

def _get_ner():
    global _ner_agent
    if _ner_agent is None:
        try:
            from .ner_agent import LocalNERAgent
            _ner_agent = LocalNERAgent()
        except Exception as e:
            print(f"[UI] NER agent unavailable: {e}")
    return _ner_agent


# ── Agent dispatch ────────────────────────────────────────────────────────────

def _run_hybrid(records, task_id: int):
    if task_id == 1:
        return [_fix_record_task1(r) for r in records]
    elif task_id == 2:
        return [_redact_record(r) for r in records]
    elif task_id == 3:
        return [_anonymise_record(r) for r in records]
    elif task_id == 4:
        return [_extract_knowledge_rule_based(r) for r in records]
    elif task_id == 5:
        return [_redact_contextual_phi(r) for r in records]
    return records


def _run_llm(records, task_id: int):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None  # caller falls back to hybrid
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        model = os.environ.get("MODEL_NAME", "gpt-4o")
        system = (
            "You are a medical data engineer. "
            "Return ONLY a valid JSON object with a 'records' key containing a list of processed records "
            "matching the input schema. For Task 4 use a 'knowledge' key with a list of "
            "{entities:[{text,type,code}], summary:str} objects, one per input record."
        )
        user = f"TASK ID: {task_id}\nINPUT RECORDS:\n{json.dumps(records, default=str)}"
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        parsed = json.loads(resp.choices[0].message.content)
        key = "knowledge" if task_id == 4 else "records"
        result = parsed.get(key, list(parsed.values())[0] if parsed else [])
        return result if isinstance(result, list) else [result]
    except Exception as e:
        print(f"[UI] LLM call failed: {e}")
        return None


def _run_agent(agent_type: str, records, task_id: int):
    if agent_type == _OPENAI_AGENT:
        result = _run_llm(records, task_id)
        if result is not None:
            return result
    return _run_hybrid(records, task_id)


def _step_env(env: MedicalOpenEnv, task_id: int, processed):
    if task_id == 4:
        action = Action(knowledge=processed, is_final=True)
    else:
        action = Action(records=processed, is_final=True)
    return env.step(action)


# ── Grade colour ─────────────────────────────────────────────────────────────

def _grade_color(score: float) -> str:
    if score >= 0.90: return "#22c55e"   # green
    if score >= 0.75: return "#eab308"   # yellow
    if score >= 0.60: return "#f97316"   # orange
    return "#ef4444"                     # red


# ── Tab 1: Pipeline ────────────────────────────────────────────────────────────

def _json_to_html_table(records: list[dict], max_col_width: int = 300) -> str:
    """
    Renders a list of JSON-like dicts into a clean HTML table.
    Crucial for fixing the Gradio DataFrame scroll-lock bug when rendering
    nested structures with wrap=True.
    """
    if not records:
        return "<p style='color: #64748b; font-style: italic;'>No records to display.</p>"

    keys = list(records[0].keys())
    html = [
        "<div style='max-height: 600px; overflow-y: auto; overflow-x: auto; border: 1px solid #1e293b; border-radius: 8px; font-size: 0.85rem;'>",
        "<table style='width: 100%; border-collapse: collapse; text-align: left; font-family: sans-serif;'>"
    ]

    # Header
    html.append("<thead style='background-color: #0f172a; position: sticky; top: 0; z-index: 10;'><tr>")
    for key in keys:
        html.append(f"<th style='padding: 10px 12px; border-bottom: 2px solid #334155; font-weight: 600; color: #cbd5e1;'>{key}</th>")
    html.append("</tr></thead>")

    # Body
    html.append("<tbody>")
    for idx, row in enumerate(records):
        bg_color = "#1e293b" if idx % 2 == 0 else "#0f172a"
        html.append(f"<tr style='background-color: {bg_color}; border-bottom: 1px solid #334155; transition: background-color 0.2s ease;' onmouseover=\"this.style.backgroundColor='#334155'\" onmouseout=\"this.style.backgroundColor='{bg_color}'\">")
        for key in keys:
            val = row.get(key, "")
            # Format nested objects nicely
            if isinstance(val, (dict, list)):
                import json
                val_str = json.dumps(val, indent=2)
                display_val = f"<pre style='margin: 0; font-family: monospace; font-size: 0.75rem; color: #94a3b8; white-space: pre-wrap; word-break: break-word;'>{val_str}</pre>"
            else:
                display_val = f"<div style='max-width: {max_col_width}px; word-wrap: break-word; white-space: normal;'>{str(val)}</div>"
            html.append(f"<td style='padding: 10px 12px; vertical-align: top; color: #cbd5e1;'>{display_val}</td>")
        html.append("</tr>")
    
    html.append("</tbody></table></div>")
    return "".join(html)


def _diff_records(original: list[dict], processed: list[dict]) -> str:
    """Render a field-level diff between original and processed records."""
    if not original or not processed:
        return "<p style='color:#64748b;font-style:italic;'>No diff available.</p>"

    rows = []
    for idx, (orig, proc) in enumerate(zip(original, processed)):
        changed = []
        for key, orig_val in orig.items():
            proc_val = proc.get(key)
            if proc_val != orig_val:
                changed.append(
                    f"<tr><td>{key}</td><td><pre>{json.dumps(orig_val, ensure_ascii=False)}</pre></td>"
                    f"<td><pre>{json.dumps(proc_val, ensure_ascii=False)}</pre></td></tr>"
                )
        if changed:
            rows.append(
                f"<h4 style='margin:8px 0;'>Record {idx+1} — ID: {orig.get('record_id','?')}</h4>"
                "<table style='width:100%;border-collapse:collapse;font-size:0.85rem;'>"
                "<thead><tr><th style='text-align:left;'>Field</th><th>Original</th><th>Processed</th></tr></thead>"
                f"<tbody>{''.join(changed)}</tbody></table>"
            )
    if not rows:
        return "<p style='color:#22c55e;'>No field-level differences detected.</p>"
    return "".join(rows)


def _feedback_summary(task_id: int, breakdown: dict[str, Any]) -> str:
    lines = []
    if task_id in (2, 3):
        leaks = []
        for rec in breakdown.get("per_record", []):
            leaks.extend(rec.get("leaked_categories", []))
        leaks = sorted(set(leaks))
        if leaks:
            lines.append(f"- ❌ PHI leaked: {', '.join(leaks)}")
        else:
            lines.append("- ✅ No PHI leaks detected")
        util = breakdown.get("utility_score") or breakdown.get("ml_utility_score") or breakdown.get("ml_score")
        if util is not None:
            lines.append(f"- 📈 Clinical utility: {util:.3f}")

    if task_id == 1 and "fields_needing_attention" in breakdown:
        weak = breakdown.get("fields_needing_attention") or []
        if weak:
            lines.append(f"- 🔧 Fields needing attention: {', '.join(sorted(set(weak)))}")

    if task_id == 4:
        ent = breakdown.get("avg_entity_extraction")
        summ = breakdown.get("avg_summary_fidelity")
        if ent is not None:
            lines.append(f"- 📚 Entity extraction: {ent:.3f}")
        if summ is not None:
            lines.append(f"- 📝 Summary fidelity: {summ:.3f}")

    if task_id == 5:
        for k in ("patient_phi_score", "provider_phi_score", "contextual_accuracy"):
            if k in breakdown:
                lines.append(f"- {k.replace('_',' ').title()}: {breakdown[k]:.3f}")

    if not lines:
        lines.append("- No additional feedback available.")
    return "\n".join(lines)

def run_pipeline(task_id: int, seed: int, agent_type: str):
    log_lines = []

    try:
        env = MedicalOpenEnv()
        obs = env.reset(task_id=int(task_id), seed=int(seed))
        records = obs.records
        log_lines.append(f"✅ Episode started — task={task_id}, seed={seed}, records={len(records)}")

        processed = _run_agent(agent_type, records, int(task_id))
        log_lines.append(f"✅ Agent `{agent_type}` processed {len(processed)} records")

        _, reward, done, info = _step_env(env, int(task_id), processed)
        log_lines.append(f"✅ Graded — score={reward.score:.4f}, passed={info.get('passed')}, done={done}")

        verdict = judge(int(task_id), reward.score, reward.breakdown, info.get("passed", False))
        log_lines.append(f"✅ Judge: grade={verdict['grade']}, regret={verdict['regret']}")

        feedback_md = _feedback_summary(int(task_id), reward.breakdown)
        diff_html = _diff_records(records, processed)

        # Render as HTML instead of DataFrame to avoid Gradio scroll-lock bug
        html_orig = _json_to_html_table(records)
        if task_id == 4:
            processed_view = [
                {"entities_count": len(p.get("entities", [])), "summary": p.get("summary", "")[:120]}
                for p in processed
            ]
        else:
            processed_view = processed
        html_proc = _json_to_html_table(processed_view)

        # Score breakdown table
        bd = reward.breakdown
        rows = [{"Metric": k, "Value": f"{v:.4f}" if isinstance(v, float) else str(v)}
                for k, v in bd.items() if k != "per_record"]
        df_breakdown = pd.DataFrame(rows)

        # FHIR
        fhir = {}
        if records and task_id != 4:
            try:
                fhir = export_to_fhir(records[0])
            except Exception:
                fhir = {"error": "FHIR export unavailable"}

        export_payload = {
            "task_id": int(task_id),
            "seed": int(seed),
            "observation": obs.model_dump(),
            "processed": processed,
            "reward": reward.breakdown,
            "passed": info.get("passed", False),
            "history": env._history,
        }
        export_json = json.dumps(export_payload, default=str, indent=2)

        action_log = "\n".join(log_lines)
        return (
            html_orig,
            html_proc,
            df_breakdown,
            verdict["verdict_text"],
            feedback_md,
            diff_html,
            action_log,
            json.dumps(fhir, indent=2),
            export_json,
        )

    except Exception as e:
        error_msg = f"❌ Pipeline error: {e}"
        log_lines.append(error_msg)
        empty_html = _json_to_html_table([])
        return (
            empty_html,
            empty_html,
            pd.DataFrame(),
            f"## Error\n```\n{e}\n```",
            "- No feedback (error)",
            empty_html,
            "\n".join(log_lines),
            "{}",
            "{}",
        )


# ── Tab 3: Multi-Task Benchmark ────────────────────────────────────────────────

def run_multitask_benchmark(seed: int):
    rows = []
    log_lines = []
    all_processed = {}

    for tid in [1, 2, 3, 4, 5]:
        try:
            env = MedicalOpenEnv()
            obs = env.reset(task_id=tid, seed=int(seed))
            processed = _run_hybrid(obs.records, tid)
            all_processed[f"Task_{tid}"] = processed
            _, reward, _, info = _step_env(env, tid, processed)
            v = judge(tid, reward.score, reward.breakdown, info.get("passed", False))

            # Extract key metrics
            bd = reward.breakdown
            phi = bd.get("phi_score", bd.get("avg_phi_score", "—"))
            util = bd.get("utility_score", bd.get("ml_utility_score", bd.get("ml_score", bd.get("avg_entity_extraction", "—"))))

            rows.append({
                "Task": _TASK_NAMES[tid],
                "Score": f"{reward.score:.4f}",
                "Grade": v["grade"],
                "Pass": "✅" if info.get("passed") else "❌",
                "PHI Score": f"{phi:.4f}" if isinstance(phi, float) else phi,
                "Utility/ML": f"{util:.4f}" if isinstance(util, float) else util,
            })
            log_lines.append(f"Task {tid}: score={reward.score:.4f} | grade={v['grade']} | passed={info.get('passed')}")
        except Exception as e:
            rows.append({"Task": _TASK_NAMES[tid], "Score": "ERROR", "Grade": "—", "Pass": "❌", "PHI Score": "—", "Utility/ML": "—"})
            log_lines.append(f"Task {tid}: ERROR — {e}")

    avg = sum(float(r["Score"]) for r in rows if r["Score"] != "ERROR") / max(1, len([r for r in rows if r["Score"] != "ERROR"]))
    summary = f"**Average Score (Hybrid Baseline, seed={seed}): `{avg:.4f}`**"

    return pd.DataFrame(rows), summary, "\n".join(log_lines), all_processed


# ── Tab 4: Robustness Sweep ────────────────────────────────────────────────────

def run_robustness_sweep(task_id: int, n_seeds: int):
    seeds = list(range(42, 42 + int(n_seeds)))
    rows = []
    log_lines = []

    for s in seeds:
        try:
            env = MedicalOpenEnv()
            obs = env.reset(task_id=int(task_id), seed=s)
            processed = _run_hybrid(obs.records, int(task_id))
            _, reward, _, info = _step_env(env, int(task_id), processed)
            rows.append({"Seed": s, "Score": reward.score, "Passed": info.get("passed", False)})
            log_lines.append(f"Seed {s}: score={reward.score:.4f}, passed={info.get('passed')}")
        except Exception as e:
            rows.append({"Seed": s, "Score": 0.0, "Passed": False})
            log_lines.append(f"Seed {s}: ERROR — {e}")

    df = pd.DataFrame(rows)
    scores = df["Score"].tolist()
    mn = min(scores)
    mx = max(scores)
    avg = sum(scores) / len(scores)
    variance = mx - mn

    summary = f"""### 🔬 Robustness Results — {_TASK_NAMES[int(task_id)]}

| Metric | Value |
|---|---|
| Seeds tested | {len(seeds)} |
| Min score | `{mn:.4f}` |
| Max score | `{mx:.4f}` |
| Average score | `{avg:.4f}` |
| Score variance (max−min) | `{variance:.4f}` |
| Passes | {sum(1 for r in rows if r['Passed'])} / {len(rows)} |
"""

    return df, summary, "\n".join(log_lines)


# ── Scoring Guide content ──────────────────────────────────────────────────────

_SCORING_GUIDE = """
## 📖 Scoring Guide & Reference

### How Episodes Work
```
POST /reset {task_id, seed}
  → 6 synthetic EHRs delivered as observation
  → Agent processes records
POST /step {records or knowledge, is_final: true}
  → Task grader scores submission
  → Dense reward returned with per-record breakdown
  → Up to 10 steps per episode
```

---

### Task Formulas & Pass Bars

| Task | Formula | Pass Bar |
|---|---|---|
| **Task 1** — Hygiene | `per_field_avg × 0.8 + longitudinal_consistency × 0.2` | score ≥ 0.85 |
| **Task 2** — Redaction | `phi_score × 0.6 + utility_score × 0.4` | phi = 1.0 AND utility ≥ 0.80 |
| **Task 3** — Anonymisation | `avg_phi × 0.4 + avg_ml_utility × 0.3 + avg_fidelity × 0.2 + k_score × 0.1` | phi = 1.0 AND ml_utility ≥ 0.60 |
| **Task 4** — Knowledge | `entity × 0.4 + code_precision × 0.3 + summary_fidelity × 0.3` | entity ≥ 0.75 AND summary ≥ 0.50 |
| **Task 5** — Contextual PII | `patient_phi × 0.5 + provider_phi × 0.3 + contextual × 0.2` | overall ≥ 0.70 AND patient_phi ≥ 0.80 |

---

### PHI Categories (Tasks 2 & 3)

| Category | Required Redaction Token | Where It Appears |
|---|---|---|
| Patient name | `[REDACTED_NAME]` | Structured field + notes (`"Pt. Smith..."`) |
| MRN | `[REDACTED_MRN]` | Structured field + inline note mentions |
| Date of Birth | `[REDACTED_DOB]` | Structured field |
| Phone | `[REDACTED_PHONE]` | Structured field + notes (`"Reachable at 555-..."`) |
| Email | `[REDACTED_EMAIL]` | Structured field + notes (`"email: user@..."`) |
| Address | `[REDACTED_ADDRESS]` | Structured field |

---

### Adversarial Identifiers (Task 3 only)

Task 3 embeds **hidden re-identification traps** in clinical notes — rare disease + ZIP code
combinations that would allow a linkage attack on a published dataset:

> *"the Gaucher disease patient from 10012"*
> *"transferred from ZIP 90210 with Fabry disease diagnosis"*

These are **never shown as PHI tokens** — the agent must detect them contextually.
The grader blends a direct-PHI score (75%) with an adversarial-privacy score (25%).

---

### k-Anonymity (Task 3)

Records must satisfy **k=2** on quasi-identifiers:
- Age-group bucket (`18-40`, `41-60`, `61-75`, `76+`)  — replace exact DOB
- Gender
- Address (first word / prefix)

The k-anonymity component contributes **10%** of the Task 3 score.

---

### Task 4 — Reference Summary Format

The grader compares your summary against a **structured clinical abstract** built
from ICD-10 codes, medications, and vitals (not raw notes).

- High score: concise summary covering core structured facts.
- Low score: generic text or hallucinated details not grounded in record fields.
- Copying clinical notes verbatim does not provide a guaranteed advantage.

---

### Task 5 — Contextual PII Disambiguation

Purpose: decide what to redact vs. keep based on **who** the identifier refers to.

- Redact patient and family identifiers (e.g., "Mr. Smith" when the patient is Smith)
- Preserve provider identifiers (e.g., "Dr. Smith" stays)
- Handle ambiguous mentions carefully; the grader scores `patient_phi_score`, `provider_phi_score`, and `contextual_accuracy`, combined into `overall_score`
- Pass bar: overall ≥ 0.70 **and** patient_phi_score ≥ 0.80

---

### Grade Scale

| Grade | Score Range | Meaning |
|---|---|---|
| 🟢 A | ≥ 0.95 | Excellent — ready for production |
| 🟡 B | ≥ 0.85 | Good — passes environment minimums |
| 🟠 C | ≥ 0.70 | Acceptable — but fails strict pass bars |
| 🔴 D | ≥ 0.50 | Poor — fundamental issues |
| ⛔ F | < 0.50 | Failing — re-examine approach |
"""

# ── About content ─────────────────────────────────────────────────────────────

_ABOUT = """
## ℹ️ About — Medical Records OpenEnv

### What This Is
An **OpenEnv-compliant reinforcement learning environment** for healthcare AI.
Agents process batches of 6 synthetic Electronic Health Records across 5 progressively harder tasks.
All patient data is 100% synthetic — generated with Faker under a fixed seed.

### Architecture
```
Agent / Client
    │
    ▼
FastAPI (src/api.py)          ← write + read endpoint rate limiting
    │                           ← Episode TTL: 1 hour, max 100 active
    ▼
MedicalOpenEnv (src/environment.py)
    ├── EHRGenerator (src/data_generator.py)   ← Faker + seeded RNG
    ├── Task 1 Grader (task1_hygiene.py)
    ├── Task 2 Grader (task2_redaction.py)
    ├── Task 3 Grader (task3_anonymization.py) ← mock ML + k-anon
    ├── Task 4 Grader (task4_knowledge.py)     ← ref summary builder
    └── Task 5 Grader (task5_reasoning.py)     ← contextual disambiguation

Gradio UI (src/ui.py)         ← this dashboard, mounted at /ui
MCP Tool Server (src/mcp_server.py) ← OpenEnv RFC 003, at /mcp
Hybrid Baseline (src/baseline_agent.py)
    └── LocalNERAgent (src/ner_agent.py)       ← BERT NER (optional)
```

### Key Design Decisions

| Decision | Rationale |
|---|---|
| Synthetic data only | No HIPAA exposure risk, fully reproducible |
| Dense per-record rewards | Agent gets granular feedback, not just episode-end signal |
| Adversarial identifiers hidden | Agent cannot exploit knowledge of trap locations |
| Task 4 reference matching | Uses structured reference abstracts, not raw notes |
| Rate limiting on write + key read endpoints | Protects shared judging infrastructure |

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/tasks` | GET | List tasks with formulas and pass bars |
| `/reset` | POST | Start episode → `{episode_id, observation}` |
| `/step` | POST | Submit action → `{reward, done, info}` |
| `/state` | GET | Episode snapshot + audit trail |
| `/export` | GET | Full episode export + `reward_trend` (for replay/debugging) |
| `/grader` | GET | Re-grade last submission (idempotent) |
| `/metrics` | GET | In-process counters (episodes, steps, errors, rate-limit hits) |
| `/baseline` | GET | Run hybrid baseline across all 5 tasks |
| `/health` | GET | `{"status": "healthy"}` |
| `/health/detailed` | GET | Health + NER + BERTScore + metrics + capacity |
| `/mode` | GET | `{"mode": "agentic"}` |
| `/schema` | GET | JSON schemas for action/observation/reward |
| `/openapi.json` | GET | OpenAPI spec for validator tooling |
| `/mcp` | — | FastMCP tools (OpenEnv RFC 003) |

### Gymnasium Integration

`gym_env.py` at the repo root provides a drop-in gymnasium wrapper:

```python
from gym_env import MedicalRecordsGymEnv       # HTTP API wrapper
from gym_env import LocalMedicalRecordsGymEnv  # direct Python wrapper (no server)

env = LocalMedicalRecordsGymEnv(task_id=2)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

Compatible with RLlib · Stable Baselines 3 · CleanRL · TRL and any gymnasium-compatible framework.
Install the optional dep: `pip install gymnasium`

### Environment Limits
- Records per episode: **6**
- Max steps: **10**
- Episode TTL: **1 hour**
- Max concurrent episodes: **100**
- Write rate limit: **10 req / 60s per IP** on `/reset` and `/step`
- Read rate limit: **60 req / 60s per IP** on key read endpoints (`/baseline`, `/state`, `/grader`, `/export`, `/metrics`, `/health/detailed`)

### Stack
`FastAPI` · `Gradio` · `Pydantic` · `Faker` · `PyTorch` · `Transformers` ·
`BERTScore` · `fastmcp` · `httpx` · `pandas` · `numpy` · `python-dotenv` ·
`gymnasium` *(optional, for gym_env.py)*
"""

# ── Persistent accordions ─────────────────────────────────────────────────────

def _scoring_accordion():
    with gr.Accordion("📖 Scoring Formula Reference", open=False):
        gr.Markdown("""
| Task | Formula | Pass Bar |
|---|---|---|
| 1 — Hygiene | `per_field × 0.8 + consistency × 0.2` | ≥ 0.85 |
| 2 — Redaction | `phi × 0.6 + utility × 0.4` | phi=1.0 & utility≥0.80 |
| 3 — Anonymisation | `phi×0.4 + ml_utility×0.3 + fid×0.2 + k×0.1` | phi=1.0 & ml_utility≥0.60 |
| 4 — Knowledge | `entity×0.4 + code×0.3 + summary×0.3` | entity≥0.75 & summary≥0.50 |
| 5 — Contextual | `patient×0.5 + provider×0.3 + contextual×0.2` | overall≥0.70 & patient≥0.80 |
""")

def _phi_accordion():
    with gr.Accordion("🔐 PHI Token Reference", open=False):
        gr.Markdown("""
| PHI | Token |
|---|---|
| Name | `[REDACTED_NAME]` |
| MRN | `[REDACTED_MRN]` |
| DOB | `[REDACTED_DOB]` |
| Phone | `[REDACTED_PHONE]` |
| Email | `[REDACTED_EMAIL]` |
| Address | `[REDACTED_ADDRESS]` |

PHI appears in **both structured fields and clinical notes** — embedded informally as `"Pt. Smith..."`, `"email: user@..."`, etc.
""")


# ── Main UI builder ────────────────────────────────────────────────────────────

def create_ui():
    css = """
    .gr-button-primary { background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important; border: none !important; }
    .gr-button-secondary { background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important; border: none !important; color: white !important; }
    .score-display { font-size: 2.5rem; font-weight: 700; text-align: center; }
    .log-box { font-family: 'Courier New', monospace; font-size: 0.82rem; background: #0f172a; color: #94a3b8; border-radius: 8px; padding: 12px; }
    """

    with gr.Blocks(
        title="Medical Records OpenEnv",
    ) as demo:

        # Queue can hang on some hosted deployments; keep direct mode by default.
        if os.getenv("GRADIO_ENABLE_QUEUE", "0") == "1":
            demo.queue(default_concurrency_limit=1)

        # ── Header ────────────────────────────────────────────────────────────
        gr.Markdown("""
# 🏥 Medical Records Data Cleaner & PII Redactor
**OpenEnv-compliant RL environment for healthcare AI** — synthetic EHR cleaning, PHI redaction, anonymisation, and clinical knowledge extraction.

`Flow: Reset → Process EHRs → Submit → Get dense reward → Iterate up to 10 steps`
        """)

        # ── Persistent accordions ─────────────────────────────────────────────
        _scoring_accordion()
        _phi_accordion()



        # ══════════════════════════════════════════════════════════════════════
        with gr.Tabs():

            # ─── Tab 1: Pipeline ───────────────────────────────────────────
            with gr.TabItem("🎮 Pipeline"):
                gr.Markdown("**Process records with the selected agent and see the before/after comparison.**")

                # ── Controls for Pipeline ──────────────────────────────────────────────
                with gr.Row():
                    task_dd = gr.Dropdown(
                        choices=[1, 2, 3, 4, 5],
                        value=1,
                        label="🎯 Task ID",
                        info="1=Hygiene · 2=Redaction · 3=Anonymisation · 4=Knowledge · 5=Contextual"
                    )
                    seed_num = gr.Number(value=42, label="🎲 Random Seed", precision=0)
                    agent_dd = gr.Dropdown(
                        choices=[_HYBRID_AGENT, _OPENAI_AGENT],
                        value=_HYBRID_AGENT,
                        label="🤖 Agent",
                        info="Hybrid requires no API key"
                    )

                gr.Markdown("---")

                run_btn = gr.Button("🚀 Run Pipeline", variant="primary", size="lg")

                with gr.Row():
                    # Left pane: data
                    with gr.Column(scale=3):
                        with gr.Tabs():
                            with gr.TabItem("📥 Input Records"):
                                out_orig = gr.HTML(label="Original (dirty/annotated) records")
                            with gr.TabItem("📤 Processed Output"):
                                out_proc = gr.HTML(label="Agent output")
                            with gr.TabItem("📊 Score Breakdown"):
                                out_breakdown = gr.DataFrame(label="Per-metric scores", wrap=True)
                            with gr.TabItem("🧭 Feedback"):
                                out_feedback = gr.Markdown(label="Actionable feedback")
                            with gr.TabItem("🧾 Diff"):
                                out_diff = gr.HTML(label="Field-level diff")
                            with gr.TabItem("🏥 FHIR Export"):
                                out_fhir = gr.JSON(label="First record → FHIR R4 bundle")
                            with gr.TabItem("📦 Export JSON"):
                                out_export = gr.JSON(label="Episode export")

                    # Right pane: verdict + log
                    with gr.Column(scale=2):
                        out_verdict = gr.Markdown(label="Judge Verdict", value="*Run the pipeline to see the verdict.*")
                        out_log = gr.Textbox(
                            label="📋 Action Log",
                            lines=10,
                            interactive=False,
                            elem_classes=["log-box"],
                        )

                run_btn.click(
                    fn=run_pipeline,
                    inputs=[task_dd, seed_num, agent_dd],
                    outputs=[out_orig, out_proc, out_breakdown, out_verdict, out_feedback, out_diff, out_log, out_fhir, out_export],
                    queue=False,
                )

            # ─── Tab 2: Multi-Task Benchmark ───────────────────────────────
            with gr.TabItem("🤖 Multi-Task Benchmark"):
                gr.Markdown("""
**Run the Hybrid baseline across all 5 tasks simultaneously.**
This gives you a complete picture of baseline performance — the reference every frontier model must beat.

Note: Task 4 is intentionally strict and often the hardest baseline to clear because summary fidelity is
scored against structured clinical facts, not free-form note overlap.
                """)

                bench_btn = gr.Button("▶ Run Full Benchmark (Hybrid Baseline, all 5 tasks)", variant="primary", size="lg")
                bench_seed = gr.Number(value=42, label="🎲 Seed", precision=0)

                bench_summary = gr.Markdown(value="*Click Run to start the benchmark.*")

                with gr.Row():
                    with gr.Column(scale=2):
                        bench_df = gr.DataFrame(label="Results by Task", wrap=True)
                        bench_json = gr.JSON(label="Processed Outputs (All Tasks)")
                    with gr.Column(scale=1):
                        bench_log = gr.Textbox(
                            label="📋 Benchmark Log",
                            lines=12,
                            interactive=False,
                            elem_classes=["log-box"],
                        )

                bench_btn.click(
                    fn=run_multitask_benchmark,
                    inputs=[bench_seed],
                    outputs=[bench_df, bench_summary, bench_log, bench_json],
                    queue=False,
                )

            # ─── Tab 3: Robustness Sweep ──────────────────────────────────
            with gr.TabItem("🔬 Robustness Sweep"):
                gr.Markdown("""
**Test how consistently the Hybrid baseline performs across multiple random seeds.**
Score variance shows task difficulty — a frontier LLM must beat the baseline AND maintain low variance.
                """)

                with gr.Row():
                    sweep_task = gr.Dropdown(choices=[1, 2, 3, 4, 5], value=2, label="🎯 Task")
                    sweep_n = gr.Slider(minimum=3, maximum=10, value=5, step=1, label="🔢 Number of Seeds")

                sweep_btn = gr.Button("🔬 Run Robustness Sweep", variant="secondary", size="lg")

                sweep_summary = gr.Markdown(value="*Click Run to start the sweep.*")

                with gr.Row():
                    with gr.Column(scale=2):
                        sweep_df = gr.DataFrame(label="Score per seed", wrap=True)
                    with gr.Column(scale=1):
                        sweep_log = gr.Textbox(
                            label="📋 Sweep Log",
                            lines=12,
                            interactive=False,
                            elem_classes=["log-box"],
                        )

                sweep_btn.click(
                    fn=run_robustness_sweep,
                    inputs=[sweep_task, sweep_n],
                    outputs=[sweep_df, sweep_summary, sweep_log],
                    queue=False,
                )

            # ─── Tab 4: Scoring Guide ─────────────────────────────────────
            with gr.TabItem("📖 Scoring Guide"):
                gr.Markdown(_SCORING_GUIDE)

            # ─── Tab 5: About ─────────────────────────────────────────────
            with gr.TabItem("ℹ️ About"):
                gr.Markdown(_ABOUT)

    return demo


if __name__ == "__main__":
    ui = create_ui()
    ui.launch(server_name="0.0.0.0", server_port=7861)

