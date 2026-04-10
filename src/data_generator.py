"""
Faker-based synthetic EHR generator.
All data is 100% synthetic — no real patient data is ever used or stored.
"""

from __future__ import annotations

import random
import uuid
import re
from copy import deepcopy

from faker import Faker

from .models import (
    AnnotatedRecord,
    DirtyRecord,
    Medication,
    PHICategory,
    PHIToken,
    PatientRecord,
    Vitals,
)
from .utils import INVALID_ICD10_EXAMPLES

# ---------------------------------------------------------------------------
# Clinical content templates (fully synthetic)
# ---------------------------------------------------------------------------

VALID_ICD10_POOL = [
    "I10",      # Essential hypertension
    "E11.9",    # Type 2 diabetes, uncomplicated
    "J45.20",   # Mild intermittent asthma
    "J44.1",    # COPD with acute exacerbation
    "I50.9",    # Heart failure, unspecified
    "N18.3",    # Chronic kidney disease, stage 3
    "E03.9",    # Hypothyroidism, unspecified
    "E78.5",    # Hyperlipidemia, unspecified
    "I48.91",   # Unspecified AF
    "J18.9",    # Pneumonia, unspecified organism
    "K37",      # Appendicitis, unspecified
    "F32.1",    # Major depressive disorder, single episode
    "F41.1",    # Generalised anxiety disorder
    "M81.0",    # Osteoporosis
    "E66.9",    # Obesity, unspecified
    "I63.9",    # Stroke, unspecified
    "I26.99",   # Pulmonary embolism
    "A41.9",    # Sepsis, unspecified
    "D64.9",    # Anaemia, unspecified
    "G43.909",  # Migraine, unspecified
]

MEDICATION_POOL = [
    Medication(name="Metformin", dose_mg=500.0, frequency="twice daily"),
    Medication(name="Lisinopril", dose_mg=10.0, frequency="once daily"),
    Medication(name="Atorvastatin", dose_mg=20.0, frequency="once daily at night"),
    Medication(name="Amlodipine", dose_mg=5.0, frequency="once daily"),
    Medication(name="Omeprazole", dose_mg=20.0, frequency="once daily"),
    Medication(name="Levothyroxine", dose_mg=0.05, frequency="once daily"),
    Medication(name="Metoprolol", dose_mg=25.0, frequency="twice daily"),
    Medication(name="Warfarin", dose_mg=5.0, frequency="once daily"),
    Medication(name="Aspirin", dose_mg=81.0, frequency="once daily"),
    Medication(name="Albuterol", dose_mg=2.5, frequency="as needed"),
    Medication(name="Furosemide", dose_mg=40.0, frequency="once daily"),
    Medication(name="Sertraline", dose_mg=50.0, frequency="once daily"),
    Medication(name="Gabapentin", dose_mg=300.0, frequency="three times daily"),
    Medication(name="Insulin glargine", dose_mg=20.0, frequency="once daily at bedtime"),
]

CLINICAL_NOTE_TEMPLATES = [
    (
        "Patient presents with {symptom1} and {symptom2}. "
        "History significant for {diagnosis1}. "
        "Currently on {med1} for {diagnosis2}. "
        "Vitals stable. Plan: continue current medications, follow-up in 4 weeks."
    ),
    (
        "Chief complaint: {symptom1}. "
        "Assessment: {diagnosis1} and {diagnosis2}. "
        "Medication {med1} increased to {dose}. "
        "Patient educated on diet and lifestyle modifications."
    ),
    (
        "{diagnosis1} exacerbation noted. "
        "Patient reports {symptom1} and {symptom2} over the past 3 days. "
        "{med1} administered. Labs pending. Referral to cardiology considered."
    ),
    (
        "Follow-up visit for {diagnosis1}. "
        "Patient denies {symptom1}. "
        "Medications: {med1} {dose}, {med2} as prescribed. "
        "A1C within target range. Next visit in 3 months."
    ),
    (
        "Chief complaint: {symptom1} for 3 days. Patient is a {age}-year-old presenting "
        "with worsening {symptom2}. Past medical history includes {diagnosis1} managed "
        "with {med1} {dose} and {diagnosis2}. Labs: CBC pending, BMP within normal limits "
        "except creatinine 1.4 (stable). Vitals: BP 138/88, HR 82, SpO2 97% on room air. "
        "Assessment: Acute exacerbation of {diagnosis1}. Plan: Increase {med1} dosage, "
        "add {med2} PRN, return to clinic in 2 weeks or sooner if symptoms worsen."
    ),
    (
        "Patient seen for follow-up of {diagnosis1} and {diagnosis2}. "
        "Current symptoms include intermittent {symptom1} and mild {symptom2}. "
        "Recent stress tests showed no acute changes. Current regimen of {med1} "
        "and {med2} appears partially effective. Vitals today: Temp 98.6F, BP 124/76, "
        "Wt 185 lbs. Lab review shows stable kidney function but elevated LDL. "
        "Impression: Chronic {diagnosis1} with secondary {diagnosis2}. "
        "Plan: Titrate {med1} to {dose} daily, maintain {med2}, and repeat lipid panel "
        "at the next visit in 6 weeks."
    ),
]

SYMPTOM_POOL = [
    "dyspnea", "chest pain", "fatigue", "nausea", "vomiting",
    "fever", "cough", "edema", "palpitations", "syncope",
]

DIAGNOSIS_POOL = [
    "hypertension", "diabetes", "asthma", "COPD", "heart failure",
    "hypothyroidism", "hyperlipidemia", "atrial fibrillation",
    "chronic kidney disease", "depression", "anxiety",
]

RARE_DISEASE_POOL = [
    "Gaucher disease",
    "Creutzfeldt-Jakob disease",
    "Paroxysmal nocturnal hemoglobinuria",
    "Fibrodysplasia ossificans progressiva",
]

# ---------------------------------------------------------------------------
# Dirty-data injection helpers
# ---------------------------------------------------------------------------

BAD_DATE_FORMATS = [
    lambda d: d.strftime("%m/%d/%Y"),      # US slash
    lambda d: d.strftime("%m-%d-%Y"),      # US dash
    lambda d: d.strftime("%Y/%m/%d"),      # reversed slash
    lambda d: d.strftime("%d/%m/%Y"),      # European slash
    lambda d: d.strftime("%B %d, %Y"),     # "January 15, 1990"
]

BAD_UNIT_SUBSTITUTIONS: dict[str, str] = {
    # swap mg ↔ mcg for some medications
    "Levothyroxine": "mcg",   # should be stored as mg
    "Albuterol": "g",         # wildly wrong
    "Metformin": "mcg",       # wrong unit
}

INVALID_ICD10_POOL = list(INVALID_ICD10_EXAMPLES) + ["X99.000", "Y88.888"]


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------

class EHRGenerator:
    """Deterministic synthetic EHR generator backed by Faker + random."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.fake = Faker("en_US")
        Faker.seed(seed)
        self._rng = random.Random(seed)
        # Provide a stable identifier for observability/debugging
        self.generator_id = f"ehrgen-{seed}"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_mrn(self) -> str:
        return f"MRN{self._rng.randint(100000, 999999)}"

    def _make_vitals(self) -> Vitals:
        return Vitals(
            heart_rate_bpm=self._rng.uniform(55, 105),
            systolic_bp_mmhg=self._rng.uniform(100, 160),
            diastolic_bp_mmhg=self._rng.uniform(60, 100),
            temperature_c=self._rng.uniform(36.0, 38.5),
            weight_kg=self._rng.uniform(50, 120),
            height_cm=self._rng.uniform(150, 195),
        )

    def _make_medications(self, n: int = 2) -> list[Medication]:
        return [deepcopy(m) for m in self._rng.sample(MEDICATION_POOL, k=min(n, len(MEDICATION_POOL)))]

    def _make_icd_codes(self, n: int = 2) -> list[str]:
        return self._rng.sample(VALID_ICD10_POOL, k=min(n, len(VALID_ICD10_POOL)))

    def _make_clinical_note(self, dob_str: str, meds: list[Medication]) -> str:
        template = self._rng.choice(CLINICAL_NOTE_TEMPLATES)
        diag_words = [self._rng.choice(DIAGNOSIS_POOL)] * 2
        syms = self._rng.sample(SYMPTOM_POOL, k=2)
        med_objs = meds[:2] if len(meds) >= 2 else meds * 2
        
        # Calculate age for templates that need it
        try:
            from datetime import date, datetime
            birth = datetime.strptime(dob_str, "%Y-%m-%d").date()
            age = int((date.today() - birth).days / 365.25)
        except Exception:
            age = self._rng.randint(18, 90)

        return template.format(
            age=age,
            symptom1=syms[0],
            symptom2=syms[1],
            diagnosis1=diag_words[0],
            diagnosis2=diag_words[1],
            med1=med_objs[0].name if med_objs else "aspirin",
            med2=med_objs[1].name if len(med_objs) > 1 else "metoprolol",
            dose=f"{int(med_objs[0].dose_mg or 10)} mg" if med_objs else "10 mg",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def make_clean_record(self) -> PatientRecord:
        """Generate a single clean patient record."""
        dob_dt = self.fake.date_of_birth(minimum_age=18, maximum_age=90)
        dob_str = dob_dt.strftime("%Y-%m-%d")
        meds = self._make_medications(n=self._rng.randint(1, 4))
        icd_codes = self._make_icd_codes(n=self._rng.randint(1, 3))
        return PatientRecord(
            record_id=str(uuid.uuid4()),
            mrn=self._make_mrn(),
            patient_name=self.fake.name(),
            dob=dob_str,
            gender=self._rng.choice(["M", "F"]),
            phone=self.fake.phone_number(),
            email=self.fake.email(),
            address=self.fake.address().replace("\n", ", "),
            icd10_codes=icd_codes,
            vitals=self._make_vitals(),
            medications=meds,
            clinical_notes=self._make_clinical_note(dob_str, meds),
        )

    # ------------------------------------------------------------------
    # Task 1: Dirty records
    # ------------------------------------------------------------------

    def _inject_ocr_noise(self, text: str, intensity: float = 0.3) -> str:
        """
        Inject realistic OCR-style errors into text.
        
        Args:
            text: Input text to corrupt
            intensity: Probability of applying noise (0.0-1.0), default 0.3
            
        Error types applied:
        - Character substitutions (common OCR confusions)
        - Character transpositions (swap adjacent chars)
        - Missing characters (random deletion)
        - Doubled characters (random duplication)
        - Spacing errors (merge words or insert spaces)
        """
        if not text or self._rng.random() > intensity:
            return text
        
        # Expanded OCR substitution map with common confusions
        char_subs = {
            # Original substitutions
            "o": "0", "O": "0", "I": "1", "l": "1", "s": "5", "S": "5",
            "z": "2", "Z": "2", "B": "8", "E": "3", "a": "@", "n": "m",
            # Additional realistic OCR confusions
            "1": "l", "0": "O", "5": "S", "8": "B", "6": "b", "9": "g",
            "c": "e", "e": "c", "h": "b", "b": "h", "q": "g", "g": "q",
            "u": "v", "v": "u", "w": "vv", "m": "nn", "d": "cl",
        }
        
        # Multi-character OCR confusions (ligature-like errors)
        multi_char_subs = {
            "rn": "m", "cl": "d", "vv": "w", "nn": "m", "ri": "n",
            "li": "h", "ll": "U",
        }
        
        result = text
        
        # Apply multi-character substitutions first (lower probability)
        if self._rng.random() < 0.2:
            for pattern, replacement in multi_char_subs.items():
                if pattern in result and self._rng.random() < 0.3:
                    # Replace only first occurrence to avoid over-corruption
                    result = result.replace(pattern, replacement, 1)
                    break
        
        chars = list(result)
        
        # Determine number of errors based on text length (1-3 errors)
        num_errors = min(self._rng.randint(1, 3), max(1, len(chars) // 5))
        
        for _ in range(num_errors):
            if len(chars) < 2:
                break
                
            error_type = self._rng.choices(
                ["substitution", "transposition", "missing", "doubled", "spacing"],
                weights=[0.35, 0.20, 0.15, 0.15, 0.15],
                k=1
            )[0]
            
            if error_type == "substitution":
                # Character substitution (original behavior, enhanced)
                idx = self._rng.randint(0, len(chars) - 1)
                if chars[idx] in char_subs:
                    replacement = char_subs[chars[idx]]
                    if len(replacement) == 1:
                        chars[idx] = replacement
                    else:
                        # Multi-char replacement (e.g., "w" -> "vv")
                        chars = chars[:idx] + list(replacement) + chars[idx+1:]
                        
            elif error_type == "transposition" and len(chars) >= 2:
                # Swap adjacent characters (e.g., "the" -> "teh")
                idx = self._rng.randint(0, len(chars) - 2)
                # Avoid transposing across word boundaries
                if chars[idx] != " " and chars[idx + 1] != " ":
                    chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
                    
            elif error_type == "missing" and len(chars) > 3:
                # Drop a character (e.g., "patient" -> "patint")
                idx = self._rng.randint(1, len(chars) - 2)  # Avoid first/last
                if chars[idx] != " ":  # Don't delete spaces
                    chars.pop(idx)
                    
            elif error_type == "doubled" and len(chars) > 2:
                # Duplicate a character (e.g., "dose" -> "dosse")
                idx = self._rng.randint(1, len(chars) - 2)
                if chars[idx] != " ":  # Don't double spaces
                    chars.insert(idx, chars[idx])
                    
            elif error_type == "spacing":
                # Spacing errors
                if self._rng.random() < 0.5:
                    # Merge words: remove a space
                    space_indices = [i for i, c in enumerate(chars) if c == " "]
                    if space_indices:
                        idx = self._rng.choice(space_indices)
                        chars.pop(idx)
                else:
                    # Insert extra space within a word
                    non_space_indices = [i for i, c in enumerate(chars) 
                                        if c != " " and i > 0 and i < len(chars) - 1]
                    if non_space_indices:
                        idx = self._rng.choice(non_space_indices)
                        chars.insert(idx, " ")
        
        return "".join(chars)

    def make_longitudinal_dirty_records(self, n_patients: int = 3, visits_per_patient: int = 2) -> tuple[list[DirtyRecord], list[PatientRecord]]:
        """
        Generate longitudinal records where some patients have multiple visits.
        Ensures consistency in ground truth, but injects inconsistency in dirty records.
        """
        dirty, truths = [], []
        
        for _ in range(n_patients):
            # Create a base identity for the patient
            dob = self.fake.date_of_birth(minimum_age=18, maximum_age=90).strftime("%Y-%m-%d")
            mrn = self._make_mrn()
            patient_name = self.fake.name()
            gender = self._rng.choice(["M", "F"])
            address = self.fake.address().replace("\n", ", ")
            phone = self.fake.phone_number()
            email = self.fake.email()
            
            for v in range(visits_per_patient):
                # Create a clean record for this visit
                meds = self._make_medications(n=self._rng.randint(1, 4))
                icd_codes = self._make_icd_codes(n=self._rng.randint(1, 3))
                
                clean = PatientRecord(
                    record_id=str(uuid.uuid4()),
                    mrn=mrn,
                    patient_name=patient_name,
                    dob=dob,
                    gender=gender,
                    phone=phone,
                    email=email,
                    address=address,
                    icd10_codes=icd_codes,
                    vitals=self._make_vitals(),
                    medications=meds,
                    clinical_notes=self._make_clinical_note(dob, meds),
                )
                
                dirty_rec = DirtyRecord(**clean.model_dump(), injected_flaws=[])
                truths.append(clean)
                
                # Inject flaws, including potential longitudinal inconsistency and OCR noise
                flaw_count = self._rng.randint(1, 3)
                available_flaws = ["bad_date", "bad_icd", "bad_unit", "missing_field", "inconsistent_gender", "inconsistent_dob", "ocr_noise"]
                chosen = self._rng.sample(available_flaws, k=min(flaw_count, len(available_flaws)))

                for flaw in chosen:
                    if flaw == "bad_date":
                        from datetime import datetime
                        dob_dt = datetime.strptime(clean.dob, "%Y-%m-%d")
                        bad_fmt = self._rng.choice(BAD_DATE_FORMATS)
                        dirty_rec.dob = bad_fmt(dob_dt)
                        dirty_rec.injected_flaws.append(f"bad_date: {dirty_rec.dob!r}")

                    elif flaw == "bad_icd":
                        bad_code = self._rng.choice(INVALID_ICD10_POOL)
                        dirty_rec.icd10_codes = dirty_rec.icd10_codes[:-1] + [bad_code]
                        dirty_rec.injected_flaws.append(f"bad_icd: {bad_code!r}")

                    elif flaw == "bad_unit" and dirty_rec.medications:
                        med_idx = self._rng.randint(0, len(dirty_rec.medications) - 1)
                        med = dirty_rec.medications[med_idx]
                        bad_unit = BAD_UNIT_SUBSTITUTIONS.get(med.name, "mcg")
                        dirty_rec.medications[med_idx] = Medication(
                            name=med.name,
                            dose_mg=med.dose_mg,
                            frequency=f"{int(med.dose_mg or 1)} {bad_unit} {med.frequency}",
                        )
                        dirty_rec.injected_flaws.append(f"bad_unit: {med.name} in {bad_unit}")

                    elif flaw == "missing_field":
                        field = self._rng.choice(["phone", "email", "address"])
                        setattr(dirty_rec, field, None)
                        dirty_rec.injected_flaws.append(f"missing_field: {field}")
                        
                    elif flaw == "inconsistent_gender" and v > 0:
                        # Only inject inconsistency if it's not the first visit
                        dirty_rec.gender = "F" if gender == "M" else "M"
                        dirty_rec.injected_flaws.append(f"inconsistent_gender: {dirty_rec.gender}")
                        
                    elif flaw == "inconsistent_dob" and v > 0:
                        # Small shift in DOB
                        from datetime import datetime, timedelta
                        dob_dt = datetime.strptime(dob, "%Y-%m-%d")
                        new_dob = (dob_dt + timedelta(days=self._rng.randint(1, 365))).strftime("%Y-%m-%d")
                        dirty_rec.dob = new_dob
                        dirty_rec.injected_flaws.append(f"inconsistent_dob: {new_dob}")
                    
                    elif flaw == "ocr_noise":
                        dirty_rec.patient_name = self._inject_ocr_noise(dirty_rec.patient_name)
                        if dirty_rec.icd10_codes:
                            dirty_rec.icd10_codes = [self._inject_ocr_noise(c) for c in dirty_rec.icd10_codes]
                        dirty_rec.injected_flaws.append("ocr_noise_injected")

                dirty.append(dirty_rec)
                
        return dirty, truths

    def make_dirty_records(self, n: int = 6) -> tuple[list[DirtyRecord], list[PatientRecord]]:
        """
        Return (dirty_records, ground_truth_records).
        Each dirty record has 1–3 injected flaws.
        """
        dirty, truths = [], []
        for _ in range(n):
            clean = self.make_clean_record()
            dirty_rec = DirtyRecord(**clean.model_dump(), injected_flaws=[])
            truths.append(clean)
            flaw_count = self._rng.randint(1, 3)
            available_flaws = ["bad_date", "bad_icd", "bad_unit", "missing_field"]
            chosen = self._rng.sample(available_flaws, k=min(flaw_count, len(available_flaws)))

            for flaw in chosen:
                if flaw == "bad_date":
                    from datetime import datetime
                    dob_dt = datetime.strptime(clean.dob, "%Y-%m-%d")
                    bad_fmt = self._rng.choice(BAD_DATE_FORMATS)
                    dirty_rec.dob = bad_fmt(dob_dt)
                    dirty_rec.injected_flaws.append(f"bad_date: {dirty_rec.dob!r}")

                elif flaw == "bad_icd":
                    bad_code = self._rng.choice(INVALID_ICD10_POOL)
                    dirty_rec.icd10_codes = dirty_rec.icd10_codes[:-1] + [bad_code]
                    dirty_rec.injected_flaws.append(f"bad_icd: {bad_code!r}")

                elif flaw == "bad_unit" and dirty_rec.medications:
                    med_idx = self._rng.randint(0, len(dirty_rec.medications) - 1)
                    med = dirty_rec.medications[med_idx]
                    # Store wrong units indicator in frequency field
                    bad_unit = BAD_UNIT_SUBSTITUTIONS.get(med.name, "mcg")
                    dirty_rec.medications[med_idx] = Medication(
                        name=med.name,
                        dose_mg=med.dose_mg,
                        frequency=f"{int(med.dose_mg or 1)} {bad_unit} {med.frequency}",
                    )
                    dirty_rec.injected_flaws.append(
                        f"bad_unit: {med.name} expressed in {bad_unit}"
                    )

                elif flaw == "missing_field":
                    field = self._rng.choice(["phone", "email"])
                    setattr(dirty_rec, field, None)
                    dirty_rec.injected_flaws.append(f"missing_field: {field}")

            dirty.append(dirty_rec)
        return dirty, truths

    # ------------------------------------------------------------------
    # Tasks 2 & 3: Annotated records with PHI
    # ------------------------------------------------------------------

    def make_annotated_records(self, n: int = 6) -> list[AnnotatedRecord]:
        """
        Generate records with PHI annotations and clinical keyword lists.
        The PHI tokens carry the exact text + expected redaction token.
        Includes both obvious structured PHI and harder embedded PHI
        in clinical notes that require genuine NER to detect.
        """
        records: list[AnnotatedRecord] = []
        for _ in range(n):
            clean = self.make_clean_record()
            tokens: list[PHIToken] = []
            adversarial_identifiers: list[str] = []

            # Name in structured field
            tokens.append(PHIToken(
                category=PHICategory.NAME,
                value=clean.patient_name,
                field="patient_name",
                redaction_token="[REDACTED_NAME]",
            ))
            # MRN
            tokens.append(PHIToken(
                category=PHICategory.MRN,
                value=clean.mrn,
                field="mrn",
                redaction_token="[REDACTED_MRN]",
            ))
            # DOB
            tokens.append(PHIToken(
                category=PHICategory.DOB,
                value=clean.dob,
                field="dob",
                redaction_token="[REDACTED_DOB]",
            ))
            # Phone
            if clean.phone:
                tokens.append(PHIToken(
                    category=PHICategory.PHONE,
                    value=clean.phone,
                    field="phone",
                    redaction_token="[REDACTED_PHONE]",
                ))
            # Email
            if clean.email:
                tokens.append(PHIToken(
                    category=PHICategory.EMAIL,
                    value=clean.email,
                    field="email",
                    redaction_token="[REDACTED_EMAIL]",
                ))
            # Address
            if clean.address:
                tokens.append(PHIToken(
                    category=PHICategory.ADDRESS,
                    value=clean.address,
                    field="address",
                    redaction_token="[REDACTED_ADDRESS]",
                ))

            # ----------------------------------------------------------------
            # Build clinical notes with EMBEDDED harder PHI patterns.
            # Standard structured-field redaction misses these; NER/LLM needed.
            # ----------------------------------------------------------------
            # 1. Last name used informally ("Pt. <Surname>") — regex won't match
            last_name = clean.patient_name.split()[-1]
            pt_alias = f"Pt. {last_name}"
            tokens.append(PHIToken(
                category=PHICategory.NAME,
                value=pt_alias,
                field="clinical_notes",
                redaction_token="[REDACTED_NAME]",
            ))

            # 2. Email username hint in prose ("reach via <username>@...")
            if clean.email and "@" in clean.email:
                email_hint_full = clean.email  # full email in freetext
                tokens.append(PHIToken(
                    category=PHICategory.EMAIL,
                    value=email_hint_full,
                    field="clinical_notes",
                    redaction_token="[REDACTED_EMAIL]",
                ))

            # Plant name + MRN + harder patterns into clinical notes
            name_mention = f"Patient {clean.patient_name} (MRN{clean.mrn.replace('MRN', '')}) "
            enriched_notes = (
                name_mention
                + clean.clinical_notes
                + f" {pt_alias} reported DOB: {clean.dob}."
                + (f" Contact email: {clean.email}." if clean.email else "")
                + (f" Reachable at {clean.phone}." if clean.phone else "")
            )

            # ----------------------------------------------------------------
            # Task-3 adversarial identifiers: indirect re-identification clues.
            # These are intentionally NOT in phi_tokens and are graded only in task 3.
            # ----------------------------------------------------------------
            zip_match = re.search(r"\b\d{5}\b", clean.address or "")
            zip_code = zip_match.group(0) if zip_match else str(self._rng.randint(10001, 99950))
            rare_dx = self._rng.choice(RARE_DISEASE_POOL)
            metaphor_tag = f"the {rare_dx} patient from {zip_code}"
            linkage_tag = f"sibling of {last_name.capitalize()}"

            enriched_notes += (
                f" Case structurally resembles {metaphor_tag}."
                f" Genetic linkage suspected as {linkage_tag} presents similarly."
            )

            adversarial_identifiers.extend([metaphor_tag, linkage_tag])

            # Clinical keywords present in notes
            from .utils import scan_clinical_keywords
            clinical_kws = scan_clinical_keywords(enriched_notes)

            records.append(AnnotatedRecord(
                **clean.model_dump(exclude={"clinical_notes"}),
                clinical_notes=enriched_notes,
                phi_tokens=tokens,
                clinical_keywords=clinical_kws,
                adversarial_identifiers=adversarial_identifiers,
            ))
        return records
