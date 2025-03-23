from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session
from backend.data_service.database import SessionLocal  # Database session factory
from backend.data_service.models import PatientData  # SQLAlchemy model for patient data
from pydantic import BaseModel  # For request validation
from backend.data_service.data_preparation import (
    data_preparation,
)  # Data preprocessing logic
import pandas as pd
from backend.data_service.database import (
    Base,
    engine,
)  # SQLAlchemy base and engine for table creation
from typing import List, Dict, Any

# Create database tables if they do not exist
Base.metadata.create_all(bind=engine)

# Initialize FastAPI application instance
app = FastAPI()


# Dependency for getting a new database session
# Ensures the session is properly closed after use
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Pydantic model for validating incoming patient data
class PatientRequest(BaseModel):
    patient_id: str
    age: int
    sex: str
    cholesterol: int
    blood_pressure: str
    heart_rate: int
    diabetes: bool
    family_history: bool
    smoking: bool
    obesity: bool
    alcohol_consumption: bool
    exercise_hours_per_week: float
    diet: str
    previous_heart_problems: bool
    medication_use: bool
    stress_level: int
    sedentary_hours_per_day: float
    income: int
    bmi: float
    triglycerides: int
    physical_activity_days_per_week: int
    sleep_hours_per_day: int
    country: str
    continent: str
    hemisphere: str
    heart_attack_risk: bool


# GET endpoint for fetching and preprocessing all stored patient data
@app.get("/data")
async def get_prepared_data(db: Session = Depends(get_db)):
    """
    Retrieves all patient records from the database,
    applies data preprocessing, and returns the processed data.
    """
    # Query all patients from the database
    patients = db.query(PatientData).all()

    if not patients:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No data available"
        )

    # Convert patient objects to dictionaries, excluding SQLAlchemy metadata
    raw_data = [
        {k: v for k, v in vars(patient).items() if k != "_sa_instance_state"}
        for patient in patients
    ]

    # Convert to pandas DataFrame for preprocessing
    df = pd.DataFrame(raw_data)

    # Apply preprocessing logic
    prepared_data = data_preparation(df)

    # Return the cleaned/prepared data as JSON
    return prepared_data.to_dict(orient="records")


# POST endpoint for cleaning uploaded patient data without saving to DB
@app.post("/clean")
async def clean_data_api(patients: List[Dict[str, Any]], drop_target: bool = False):
    """
    Accepts patient data, preprocesses it and returns the cleaned dataset.
    Useful for preprocessing before training or testing.
    """
    test_patient = pd.DataFrame(patients)
    prepared_df = data_preparation(test_patient, drop_target=drop_target)
    return prepared_df.to_dict(orient="records")


# POST endpoint to add a new patient to the database
@app.post("/patients", status_code=status.HTTP_201_CREATED)
async def create_patient(patient: PatientRequest, db: Session = Depends(get_db)):
    """
    Adds a new patient entry to the database.
    Checks for duplicate patient_id before inserting.
    """
    try:
        # Check if the patient ID already exists
        if (
            db.query(PatientData)
            .filter(PatientData.patient_id == patient.patient_id)
            .first()
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Patient with this ID already exists.",
            )
        try:
            # Create a new PatientData instance using unpacked validated data
            new_patient = PatientData(**patient.model_dump())
        except Exception as e:
            print(e)

        # Add and persist the patient in the database
        db.add(new_patient)
        db.flush()  # Executes SQL without committing yet
        db.commit()  # Commits the transaction
        db.refresh(new_patient)  # Refresh instance with DB values

        return {"status": "success", "new_patient": new_patient}

    except Exception as e:
        db.rollback()  # Roll back transaction in case of failure
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred: {str(e)}",
        )
