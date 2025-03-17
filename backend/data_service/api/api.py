from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session
from database.database import SessionLocal  # Datenbankverbindung importieren
from database.models import PatientData
from pydantic import BaseModel
from services.data_preparation import data_preparation
import pandas as pd
from database.database import Base, engine


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


# GET endpoint to fetch and process the first row of patient data
@app.get("/data_preparation/")
def get_prepared_data(db: Session = Depends(get_db)):
    """API-Endpunkt, der eine Zeile der Datenbank abruft und verarbeitet."""

    # Step 1: Holen der ersten Zeile aus der DB
    patient = db.query(PatientData).first()

    # Wenn keine Daten vorhanden sind, gib einen Fehler zurück
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No data available"
        )

    df = pd.DataFrame([patient.__dict__])  # Umwandeln in DataFrame

    # Step 3: Datenvorbereitung - Rufe die bestehende Funktion auf
    prepared_data = data_preparation(df)
    # Step 4: Umwandeln des DataFrames zurück in ein Dictionary (für FastAPI)
    prepared_data_dict = prepared_data.to_dict(orient="records")[
        0
    ]  # Nur die erste Zeile als Dictionary

    return prepared_data_dict  # Return als JSON (automatisch durch FastAPI)


# POST-Endpoint zum Hinzufügen eines Patienten
@app.post("/patients/", status_code=status.HTTP_201_CREATED)
def create_patient(patient: PatientRequest, db: Session = Depends(get_db)):
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

        # Neuen Patienten anlegen
        new_patient = PatientData(**patient.model_dump())

        db.add(new_patient)
        db.flush()  # Führt SQL aus, aber committet noch nicht
        db.commit()  # Speichert die Änderungen
        db.refresh(new_patient)  # Holt die aktualisierten Daten aus der DB

        return new_patient  # Gibt den neu erstellten Patienten zurück

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ein Fehler ist aufgetreten: {str(e)}",
        )
