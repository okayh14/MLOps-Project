import os
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import select
from database.database import SessionLocal, engine, Base
from database.models import PatientData  # Importiere das PatientData-Modell


def initialize_database():
    """Prüft, ob die DB leer ist und füllt sie mit lokalen CSV-Daten."""
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    result = session.execute(select(PatientData).limit(1)).fetchone()

    if result is None:
        print("Datenbank ist leer. Lade Daten aus der lokalen Datei...")
        load_initial_data(session)
    else:
        print("Datenbank ist bereits befüllt.")

    session.close()


def load_initial_data(session: Session):
    """Lädt die lokale CSV-Datei und speichert sie in der Datenbank."""
    # Relativer Pfad zur CSV-Datei im 'data' Ordner
    csv_file = os.path.join(os.path.dirname(__file__), "..", "data", "data.csv")

    # Überprüfen, ob die Datei existiert
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Die CSV-Datei wurde nicht gefunden: {csv_file}")

    print(f"CSV-Datei gefunden: {csv_file}")
    df = pd.read_csv(csv_file)

    # Standardisiere die Spaltennamen
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    patients = [
        PatientData(
            patient_id=row["patient_id"],
            age=row["age"],
            sex=row["sex"],
            cholesterol=row["cholesterol"],
            blood_pressure=row["blood_pressure"],
            heart_rate=row["heart_rate"],
            diabetes=row["diabetes"],
            family_history=row["family_history"],
            smoking=row["smoking"],
            obesity=row["obesity"],
            alcohol_consumption=row["alcohol_consumption"],
            exercise_hours_per_week=row["exercise_hours_per_week"],
            diet=row["diet"],
            previous_heart_problems=row["previous_heart_problems"],
            medication_use=row["medication_use"],
            stress_level=row["stress_level"],
            sedentary_hours_per_day=row["sedentary_hours_per_day"],
            income=row["income"],
            bmi=row["bmi"],
            triglycerides=row["triglycerides"],
            physical_activity_days_per_week=row["physical_activity_days_per_week"],
            sleep_hours_per_day=row["sleep_hours_per_day"],
            country=row["country"],
            continent=row["continent"],
            hemisphere=row["hemisphere"],
            heart_attack_risk=row["heart_attack_risk"],
        )
        for _, row in df.iterrows()
    ]

    session.bulk_save_objects(patients)
    session.commit()
    print(f"{len(patients)} Datensätze wurden eingefügt.")


if __name__ == "__main__":
    initialize_database()
