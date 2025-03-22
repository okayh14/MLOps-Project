import os
import sys
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Patch the database module before importing the actual API
# This ensures that the API uses the test database instead of the production one
from test.mocks.database import Base, engine, SessionLocal

sys.modules["backend.data_service.database"] = sys.modules["test.mocks.database"]
# Import the FastAPI app and dependencies AFTER mocking the DB module
from backend.data_service.api import app, get_db
from backend.data_service.models import PatientData

# Local session for testing
TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


@pytest.fixture(scope="function")
def test_db():
    """
    Creates a fresh test database before each test function and tears it down afterwards.
    Ensures isolation between test cases by creating/dropping all tables.
    """
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(test_db):
    """
    Provides a FastAPI TestClient instance with the test database dependency injected.
    Ensures that all API routes use the mocked test database during testing.
    """

    def override_get_db():
        try:
            yield test_db
        finally:
            pass  # DB is handled in the test_db fixture

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def sample_patient():
    """
    Provides a consistent sample patient dictionary used for testing POST requests.
    Matches the expected schema of the API.
    """
    return {
        "patient_id": "X001",
        "age": 67,
        "sex": "Male",
        "cholesterol": 208,
        "blood_pressure": "158/88",
        "heart_rate": 72,
        "diabetes": False,
        "family_history": True,
        "smoking": False,
        "obesity": False,
        "alcohol_consumption": False,
        "exercise_hours_per_week": 3.5,
        "diet": "Average",
        "previous_heart_problems": False,
        "medication_use": False,
        "stress_level": 5,
        "sedentary_hours_per_day": 6.2,
        "income": 50000,
        "bmi": 28.5,
        "triglycerides": 150,
        "physical_activity_days_per_week": 3,
        "sleep_hours_per_day": 7,
        "country": "Germany",
        "continent": "Europe",
        "hemisphere": "Northern",
        "heart_attack_risk": False,
    }


def test_get_prepared_data_empty_db(client):
    """
    Test GET /data when the database is empty.
    Expects a 404 response indicating no data is available.
    """
    response = client.get("/data")
    assert response.status_code == 404
    assert response.json()["detail"] == "No data available"


def test_get_prepared_data_with_data(client, test_db, sample_patient):
    """
    Test GET /data after inserting a valid patient into the test database.
    Expects the preprocessed data to be returned and contain expected fields.
    """
    test_db.add(PatientData(**sample_patient))
    test_db.commit()

    response = client.get("/data")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["age"] == 67
    assert "systolic_blood_pressure" in data[0]


def test_clean_data(client, sample_patient):
    """
    Test POST /clean with raw patient data.
    Expects the API to return the cleaned version of the data including transformed fields.
    """
    response = client.post("/clean", json=[sample_patient])
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert "systolic_blood_pressure" in data[0]


def test_create_patient(client, sample_patient):
    """
    Test POST /patients to create a new patient record in the database.
    Then use GET /data to confirm the data was stored and preprocessed correctly.
    
    Note: patient_id is removed during preprocessing and should not appear in the response.
    The presence of the correct patient is verified by comparing other fields.
    """
    response = client.post("/patients", json=sample_patient)
    assert response.status_code == 201

    response = client.get("/data")
    assert response.status_code == 200
    data = response.json()
    assert any(
        p["age"] == sample_patient["age"]
        and p["bmi"] == sample_patient["bmi"]
        and p["continent"] == sample_patient["continent"]
        for p in data
    )
