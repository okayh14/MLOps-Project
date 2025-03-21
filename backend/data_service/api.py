from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session
from database import SessionLocal  # Datenbankverbindung importieren
from models import PatientData
from pydantic import BaseModel
from data_preparation import data_preparation
import pandas as pd
from database import Base, engine
from typing import List, Dict, Any


Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI()


# Dependency, die eine Session zurückgibt und sicherstellt,
# dass sie geschlossen wird.
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


# GET endpoint to fetch and process all patient data
@app.get("/data")
async def get_prepared_data(db: Session = Depends(get_db)):
    """API-Endpunkt, der alle Patienten aus der Datenbank abruft und verarbeitet."""

    # Alle Patienten aus der DB holen
    patients = db.query(PatientData).all()

    if not patients:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No data available"
        )

    # Alle Patienten-Datensätze in Dictionaries umwandeln (SQLAlchemy-Metadaten rausfiltern)
    raw_data = [
        {k: v for k, v in vars(patient).items() if k != "_sa_instance_state"}
        for patient in patients
    ]

    # In DataFrame umwandeln
    df = pd.DataFrame(raw_data)

    # Datenvorbereitung
    prepared_data = data_preparation(df)

    # Rückgabe als JSON
    return prepared_data.to_dict(orient="records")


@app.post("/clean")
async def clean_data_api(patients: List[Dict[str, Any]], drop_target: bool = False):
    test_patient = pd.DataFrame(patients)
    prepared_df = data_preparation(test_patient, drop_target=drop_target)
    return prepared_df.to_dict(orient="records")


# POST-Endpoint zum Hinzufügen eines Patienten
@app.post("/patients", status_code=status.HTTP_201_CREATED)
async def create_patient(patient: PatientRequest, db: Session = Depends(get_db)):
    """Erstellt einen neuen Patienten in der Datenbank."""
    try:
        # Prüfen, ob die Patient-ID bereits existiert
        if (
            db.query(PatientData)
            .filter(PatientData.patient_id == patient.patient_id)
            .first()
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Patient mit dieser ID existiert bereits.",
            )
        try:

            # Neuen Patienten anlegen
            new_patient = PatientData(**patient.model_dump())
        except Exception as e:
            print(e)

        db.add(new_patient)
        db.flush()  # Führt SQL aus, aber committet noch nicht
        db.commit()  # Speichert die Änderungen
        db.refresh(new_patient)  # Holt die aktualisierten Daten aus der DB

        return {'status':'success',
                'new_patient': new_patient}  # Gibt den neu erstellten Patienten zurück

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ein Fehler ist aufgetreten: {str(e)}",
        )
