import copy

from src.record_processors import _redact_record, _fix_record_task1
from src.utils import normalize_dose_to_mg


def test_redact_record_is_case_insensitive_for_structured_phi():
    record = {
        "record_id": "r1",
        "patient_name": "John Smith",
        "mrn": "MRN123456",
        "dob": "1990-01-01",
        "phone": "555-123-4567",
        "email": "john@example.com",
        "address": "123 Main St",
        "clinical_notes": "patient smith reported chest pain. mrn123456 noted.",
    }

    redacted = _redact_record(copy.deepcopy(record))
    notes_lower = redacted.get("clinical_notes", "").lower()
    assert "john" not in notes_lower
    assert "smith" not in notes_lower
    assert "mrn123456" not in notes_lower


def test_fix_record_handles_non_dict_vitals_and_med_list():
    record = {
        "record_id": "r2",
        "mrn": "MRN999999",
        "patient_name": "Jane Doe",
        "dob": "1990/01/15",
        "gender": "female",
        "vitals": "not-a-dict",
        "medications": "not-a-list",
        "clinical_notes": "Vitals unknown",
    }

    fixed = _fix_record_task1(record)
    assert isinstance(fixed.get("vitals"), dict)
    assert isinstance(fixed.get("medications"), list)


def test_normalize_dose_to_mg_handles_none_and_numeric():
    assert normalize_dose_to_mg(None) is None
    assert normalize_dose_to_mg(5) is None
    assert normalize_dose_to_mg("500 mcg") == 0.5
