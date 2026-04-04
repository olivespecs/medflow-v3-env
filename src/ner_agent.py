"""
ML-powered NER support for the Hybrid Baseline.
Uses Hugging Face Transformers for deep PHI detection in unstructured notes.
"""

from __future__ import annotations

from typing import Any
from copy import deepcopy
import logging
import re
import os
import sys

# A medical-specific NER model would be better, but we'll start with a robust general one
# that handles names, dates, locations, and organizations.
MODEL_NAME = "dbmdz/bert-large-cased-finetuned-conll03-english"

# ---------------------------------------------------------------------------
# Confidence thresholds (tunable via env vars)
# ---------------------------------------------------------------------------
# 0.4 — base threshold for standard PHI redaction. Low enough to catch most
#        informal name mentions; high enough to avoid obvious false positives.
NER_CONFIDENCE_DEFAULT = float(os.getenv("NER_CONFIDENCE_THRESHOLD", "0.4"))

# 0.6 — raised threshold used for the final safety-net pass in baseline_agent.
#        The safety net runs AFTER rule-based redaction, so remaining capitalised
#        words are more likely real names → higher confidence bar is appropriate.
NER_CONFIDENCE_FINAL = float(os.getenv("NER_CONFIDENCE_THRESHOLD_FINAL", "0.6"))

# 0.3 — lower threshold for Task 3 adversarial-identifier scrubbing.
#        Rare-disease / ZIP combos may look innocuous to the model, so we cast
#        a wider net and accept more false positives at this stage.
NER_CONFIDENCE_ADVERSARIAL = float(os.getenv("NER_CONFIDENCE_THRESHOLD_ADVERSARIAL", "0.3"))

# Provider/facility allowlist to avoid over-redacting utility-bearing terms
PROVIDER_ALLOWLIST = {
    "hospital",
    "clinic",
    "centre",
    "center",
    "department",
    "unit",
    "pharmacy",
    "pharmaceutical",
    "lab",
    "laboratory",
    "md",
    "rn",
    "dr",
    "doctor",
    "nurse",
    "surgeon",
    "attending",
    "prof",
}


logger = logging.getLogger(__name__)


class LocalNERAgent:
    def __init__(self, model_name: str = MODEL_NAME, safe_mode: bool | None = None):
        # Auto-enable NER when ML dependencies are available.
        # Override with USE_TRANSFORMERS_NER=0 to force disable, or =1 to force enable.
        # Auto-enable on GPU environments (Colab, etc.) for best accuracy.
        is_colab = "COLAB_GPU" in os.environ or "google.colab" in sys.modules
        env_safe = os.getenv("BASELINE_SAFE_MODE", "0")
        self.safe_mode = safe_mode if safe_mode is not None else self._is_safe_mode(env_safe)

        # Check if ML dependencies are available by attempting lazy import
        _ml_available = False
        if not self.safe_mode:
            try:
                import importlib.util
                _has_transformers = importlib.util.find_spec("transformers") is not None
                _has_torch = importlib.util.find_spec("torch") is not None
                _ml_available = _has_transformers and _has_torch
            except Exception:
                _ml_available = False

        enable_default = "1" if (is_colab or _ml_available) else "0"

        self.nlp = None
        self.device = -1
        self.disabled_reason: str | None = None
        self.enable_requested = os.getenv("USE_TRANSFORMERS_NER", enable_default) == "1"

        if self.safe_mode:
            self.disabled_reason = "transformers disabled in baseline safe mode"
            logger.info("LocalNERAgent running in safe mode; using rule-based redaction only")
            return

        if self.enable_requested:
            try:
                # Lazy imports — avoid blocking startup unless explicitly enabled
                import torch
                from transformers import pipeline

                self.device = 0 if torch.cuda.is_available() else -1
                self.nlp = pipeline(
                    "ner",
                    model=model_name,
                    aggregation_strategy="simple",
                    device=self.device,
                )
            except Exception as e:  # pragma: no cover - defensive fallback
                self.disabled_reason = f"transformers load failed: {type(e).__name__}"
                self.nlp = None
                logger.warning("LocalNERAgent fallback to rule-only: %s", e)

    def _is_safe_mode(self, env_safe: str) -> bool:
        """Determine if safe mode should be enabled."""
        return env_safe.lower() in {"1", "true", "yes"}

    def diagnostics(self) -> dict[str, Any]:
        return {
            "enabled": self.nlp is not None,
            "device": self.device if self.nlp is not None else None,
            "safe_mode": self.safe_mode,
            "requested": self.enable_requested,
            "reason": self.disabled_reason,
        }

    def redact_text(self, text: str, confidence_threshold: float = NER_CONFIDENCE_DEFAULT) -> str:
        if not text:
            return ""

        redacted_text = text
        if self.nlp is not None:
            # Selective triggering: run if any likely identifiers remain
            has_title_case = bool(re.search(r"\b[A-Z][a-z]+\b", redacted_text))
            has_upper = bool(re.search(r"\b[A-Z]{2,}\b", redacted_text))
            has_initials = bool(re.search(r"\b[A-Z]\.\s*[A-Z]\.?", redacted_text))
            if not (has_title_case or has_upper or has_initials):
                # Heuristic: if nothing looks like a name/identifier, skip the model
                return redacted_text

            entities = self.nlp(text)

            # Per-label confidence thresholds to reduce over-redaction of facilities
            label_thresholds = {
                "PER": confidence_threshold,
                "LOC": max(confidence_threshold, 0.55),
                "ORG": max(confidence_threshold, 0.55),
            }

            # Collect filtered spans
            spans = []
            for ent in entities:
                label = ent.get("entity_group") or ent.get("entity") or ""
                score = ent.get("score", 1.0)
                word = ent.get("word", "")
                start = ent.get("start")
                end = ent.get("end")

                if start is None or end is None:
                    continue

                # Confidence gate
                if score < label_thresholds.get(label, confidence_threshold):
                    continue

                # Provider/facility allowlist: keep common care-site terms
                word_lower = word.lower()
                if label in {"ORG", "LOC"}:
                    if any(token in word_lower for token in PROVIDER_ALLOWLIST):
                        continue

                spans.append({
                    "start": start,
                    "end": end,
                    "label": label,
                })

            # Merge overlapping/adjacent spans and apply redactions from end to start
            def _priority(lbl: str) -> int:
                return {"PER": 3, "LOC": 2, "ORG": 1}.get(lbl, 0)

            spans = sorted(spans, key=lambda s: (s["start"], -_priority(s["label"])))
            merged: list[dict[str, Any]] = []
            for span in spans:
                if not merged:
                    merged.append(span)
                    continue
                last = merged[-1]
                if span["start"] <= last["end"]:  # overlap/adjacent
                    # Expand bounds and keep higher-priority label
                    last["end"] = max(last["end"], span["end"])
                    if _priority(span["label"]) > _priority(last["label"]):
                        last["label"] = span["label"]
                else:
                    merged.append(span)

            token_map = {
                "PER": "[REDACTED_NAME]",
                "LOC": "[REDACTED_ADDRESS]",
                "ORG": "[REDACTED_ADDRESS]",
            }

            for span in sorted(merged, key=lambda s: s["start"], reverse=True):
                token = token_map.get(span["label"], "[REDACTED]")
                redacted_text = redacted_text[: span["start"]] + token + redacted_text[span["end"] :]
            # Collapse duplicate adjacent tokens
            redacted_text = re.sub(r"(\[REDACTED_[A-Z]+\])\1+", r"\1", redacted_text)
            
        # Fallback for patterns that NER might miss (regex)
        redacted_text = re.sub(r"\bMRN\d{6,10}\b", "[REDACTED_MRN]", redacted_text)
        redacted_text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "[REDACTED_DOB]", redacted_text)
        redacted_text = re.sub(
            r"\b(\+?1[-.\s]?)?(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b",
            "[REDACTED_PHONE]",
            redacted_text,
        )
        redacted_text = re.sub(
            r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
            "[REDACTED_EMAIL]",
            redacted_text,
        )
        
        return redacted_text

    def process_record(self, record: dict[str, Any], task_id: int = 2) -> dict[str, Any]:
        """Process one record with task-specific logic.

        task_id=1: data hygiene fixes
        task_id=2: PHI redaction
        task_id=3: anonymization + utility preservation
        task_id=4: knowledge extraction
        """
        from .record_processors import (
            _anonymise_record,
            _extract_knowledge_rule_based,
            _fix_record_task1,
            _redact_record,
        )

        if task_id == 1:
            return _fix_record_task1(record)
        if task_id == 3:
            return _anonymise_record(record, ner_agent=self)
        if task_id == 4:
            return _extract_knowledge_rule_based(record, ner_agent=self)

        # Task 2 default path: deterministic redaction first, then optional NER pass.
        rec = _redact_record(record, ner_agent=self)
        if self.nlp is not None and rec.get("clinical_notes"):
            rec = deepcopy(rec)
            rec["clinical_notes"] = self.redact_text(rec["clinical_notes"])
        return rec

def run_ner_baseline(env: Any) -> dict[str, Any]:
    from .models import Action
    
    agent = LocalNERAgent()
    obs = env.reset(task_id=env._task_id, seed=env._seed)
    
    processed = [agent.process_record(r) for r in obs.records]
    
    action = Action(records=processed, is_final=True)
    _, reward, _, info = env.step(action)
    
    return {
        "task_id": env._task_id,
        "agent": f"local-ner ({MODEL_NAME})",
        "score": reward.score,
        "breakdown": reward.breakdown,
        "passed": info.get("passed", False),
        "info": info,
    }
