import os
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import select
from database import SessionLocal, engine, Base
from models import PatientData  # Import the PatientData ORM model


def initialize_database():
    """
    Initializes the database.

    - Creates tables if they don't exist.
    - Checks whether the database is empty.
    - If empty, loads initial data from a local CSV file.
    """
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()

    # Check if there is already any data in the PatientData table
    result = session.execute(select(PatientData).limit(1)).fetchone()

    if result is None:
        print("Database is empty. Loading initial data from local CSV...")
        load_initial_data(session)
    else:
        print("Database is already populated.")

    session.close()


def load_initial_data(session: Session):
    """
    Loads initial patient data from a local CSV file into the database.

    Args:
        session (Session): Active SQLAlchemy database session.
    """
    # Relative path to the CSV file located in the working directory
    csv_file = os.path.join(os.getcwd(), "data.csv")

    # Ensure the file exists
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    print(f"CSV file found: {csv_file}")

    # Read the CSV into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Standardize column names to match database fields
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # Convert each row of the DataFrame to a PatientData object
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

    # Bulk insert all patients into the database
    session.bulk_save_objects(patients)
    session.commit()
    print(f"{len(patients)} records were successfully inserted into the database.")


# Run initialization if executed as a script
if __name__ == "__main__":
    initialize_database()
