"""Tests for API input validation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_step_validates_missing_required_fields_task1():
    """Test that /step validates required fields for Task 1."""
    # Create an episode
    response = client.post("/reset", json={"task_id": 1, "seed": 42})
    assert response.status_code == 201
    episode_id = response.json()["episode_id"]
    
    # Submit records missing the critical record_id field
    invalid_records = [
        {"dob": "1990-01-01"}  # Missing record_id (required for alignment)
    ]
    
    response = client.post(
        f"/step?episode_id={episode_id}",
        json={"records": invalid_records, "is_final": False}
    )
    
    assert response.status_code == 422
    assert "Missing required fields" in response.json()["detail"]


def test_step_validates_icd10_codes_type_task1():
    """Test that /step validates icd10_codes is a list for Task 1."""
    response = client.post("/reset", json={"task_id": 1, "seed": 42})
    episode_id = response.json()["episode_id"]
    
    # Submit record with icd10_codes as string instead of list
    invalid_records = [
        {
            "record_id": "test-1",
            "dob": "1990-01-01",
            "gender": "M",
            "icd10_codes": "I10"  # Should be a list
        }
    ]
    
    response = client.post(
        f"/step?episode_id={episode_id}",
        json={"records": invalid_records, "is_final": False}
    )
    
    assert response.status_code == 422
    assert "must be a list" in response.json()["detail"]


def test_step_validates_clinical_notes_type_task2():
    """Test that /step validates clinical_notes is a string for Task 2."""
    response = client.post("/reset", json={"task_id": 2, "seed": 42})
    episode_id = response.json()["episode_id"]
    
    # Submit record with clinical_notes as list instead of string
    invalid_records = [
        {
            "record_id": "test-1",
            "clinical_notes": ["note1", "note2"]  # Should be a string
        }
    ]
    
    response = client.post(
        f"/step?episode_id={episode_id}",
        json={"records": invalid_records, "is_final": False}
    )
    
    assert response.status_code == 422
    assert "must be a string" in response.json()["detail"]


def test_step_validates_knowledge_structure_task4():
    """Test that /step validates knowledge structure for Task 4."""
    response = client.post("/reset", json={"task_id": 4, "seed": 42})
    episode_id = response.json()["episode_id"]
    
    # Submit knowledge missing required fields
    invalid_knowledge = [
        {"entities": []}  # Missing 'summary'
    ]
    
    response = client.post(
        f"/step?episode_id={episode_id}",
        json={"knowledge": invalid_knowledge, "is_final": False}
    )
    
    assert response.status_code == 422
    assert "Missing required field: 'summary'" in response.json()["detail"]


def test_step_validates_entities_type_task4():
    """Test that /step validates entities is a list for Task 4."""
    response = client.post("/reset", json={"task_id": 4, "seed": 42})
    episode_id = response.json()["episode_id"]
    
    # Submit knowledge with entities as string instead of list
    invalid_knowledge = [
        {
            "entities": "not a list",  # Should be a list
            "summary": "test summary"
        }
    ]
    
    response = client.post(
        f"/step?episode_id={episode_id}",
        json={"knowledge": invalid_knowledge, "is_final": False}
    )
    
    assert response.status_code == 422
    assert "must be a list" in response.json()["detail"]


def test_step_accepts_valid_records_task1():
    """Test that /step accepts valid records for Task 1."""
    response = client.post("/reset", json={"task_id": 1, "seed": 42})
    episode_id = response.json()["episode_id"]
    obs = response.json()["observation"]
    
    # Submit valid records (use the original records from observation)
    valid_records = obs["records"]
    
    response = client.post(
        f"/step?episode_id={episode_id}",
        json={"records": valid_records, "is_final": True}
    )
    
    assert response.status_code == 200
    assert "reward" in response.json()


def test_step_accepts_valid_knowledge_task4():
    """Test that /step accepts valid knowledge for Task 4."""
    response = client.post("/reset", json={"task_id": 4, "seed": 42})
    episode_id = response.json()["episode_id"]
    obs = response.json()["observation"]
    
    # Submit valid knowledge
    valid_knowledge = [
        {
            "entities": [
                {"text": "I10", "type": "Condition", "code": "I10"}
            ],
            "summary": "Patient has hypertension."
        }
        for _ in obs["records"]
    ]
    
    response = client.post(
        f"/step?episode_id={episode_id}",
        json={"knowledge": valid_knowledge, "is_final": True}
    )
    
    assert response.status_code == 200
    assert "reward" in response.json()
