"""Tests for simplified FHIR export utility."""

from __future__ import annotations

from src.utils import export_to_fhir


def test_export_to_fhir_basic_shape():
    record = {
        "record_id": "rec-123",
        "mrn": "MRN123456",
        "patient_name": "Jane Doe",
        "dob": "1988-01-15",
        "gender": "F",
        "phone": "555-111-2222",
        "email": "jane@example.com",
        "address": "123 Main St",
        "icd10_codes": ["I10", "E11.9"],
    }

    out = export_to_fhir(record, include_phi=True)

    assert out["bundle_type"] == "transaction"
    assert "entries" in out
    assert len(out["entries"]) == 3  # 1 patient + 2 conditions

    patient = out["entries"][0]["resource"]
    assert patient["resourceType"] == "Patient"
    assert patient["id"] == "rec-123"
    assert patient["identifier"][0]["value"] == "MRN123456"
    assert patient["name"][0]["text"] == "Jane Doe"
    assert patient["birthDate"] == "1988-01-15"
    assert patient["gender"] == "female"


def test_export_to_fhir_condition_codes_preserved():
    record = {
        "record_id": "rec-abc",
        "mrn": "MRN789",
        "patient_name": "John Smith",
        "dob": "1970-06-30",
        "gender": "M",
        "icd10_codes": ["J44.9", "N18.3"],
    }

    out = export_to_fhir(record, include_phi=True)
    condition_resources = [entry["resource"] for entry in out["entries"][1:]]

    codes = [
        cond["code"]["coding"][0]["code"]
        for cond in condition_resources
        if cond.get("resourceType") == "Condition"
    ]
    assert codes == ["J44.9", "N18.3"]


def test_export_to_fhir_unknown_gender_maps_to_other():
    record = {
        "record_id": "rec-x",
        "mrn": "MRN000",
        "patient_name": "Alex",
        "dob": "1995-04-20",
        "gender": "Unknown",
        "icd10_codes": [],
    }

    out = export_to_fhir(record, include_phi=True)
    patient = out["entries"][0]["resource"]
    assert patient["gender"] == "other"


def test_export_to_fhir_default_excludes_phi():
    record = {
        "record_id": "rec-123",
        "mrn": "MRN123456",
        "patient_name": "Jane Doe",
        "dob": "1988-01-15",
        "gender": "F",
        "phone": "555-111-2222",
        "email": "jane@example.com",
        "address": "123 Main St",
        "icd10_codes": ["I10"],
    }

    out = export_to_fhir(record)
    patient = out["entries"][0]["resource"]

    assert "identifier" not in patient
    assert "name" not in patient
    assert "telecom" not in patient
    assert "address" not in patient
    assert "birthDate" not in patient
