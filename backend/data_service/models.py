from sqlalchemy import Column, Integer, Boolean, String, Float
from backend.data_service.database import Base  # Import the declarative base from the database module


class PatientData(Base):
    """
    SQLAlchemy model that defines the structure of the 'heart_risk' table.
    Each instance of this class represents a single patient data entry.
    """
    
    __tablename__ = "heart_risk"  # Name of the table in the database

    # --- Patient Identifier ---
    patient_id = Column(String, primary_key=True)  # Unique ID for the patient

    # --- Basic Demographics ---
    age = Column(Integer, nullable=False)  # Age in years
    sex = Column(String, nullable=False)   # "Male" or "Female"

    # --- Health Metrics ---
    cholesterol = Column(Integer, nullable=False)  # Cholesterol level (mg/dL)
    blood_pressure = Column(String, nullable=False)  # Format: "Systolic/Diastolic"
    heart_rate = Column(Integer, nullable=False)  # Resting heart rate (bpm)
    bmi = Column(Float, nullable=False)  # Body Mass Index
    triglycerides = Column(Integer, nullable=False)  # Triglycerides level (mg/dL)

    # --- Lifestyle & Medical History ---
    diabetes = Column(Boolean, nullable=False)  # Diagnosed with diabetes
    family_history = Column(Boolean, nullable=False)  # Family history of heart disease
    smoking = Column(Boolean, nullable=False)  # Smoker status
    obesity = Column(Boolean, nullable=False)  # Obesity status
    alcohol_consumption = Column(Boolean, nullable=False)  # Alcohol use
    exercise_hours_per_week = Column(Float, nullable=False)  # Weekly physical activity
    physical_activity_days_per_week = Column(Integer, nullable=False)  # Active days/week
    sedentary_hours_per_day = Column(Float, nullable=False)  # Sedentary time (hrs/day)
    sleep_hours_per_day = Column(Integer, nullable=False)  # Sleep duration (hrs/day)
    diet = Column(String, nullable=False)  # "Balanced", "Average", or "Unhealthy"
    medication_use = Column(Boolean, nullable=False)  # Ongoing medication
    stress_level = Column(Integer, nullable=False)  # Perceived stress level (0–20)
    previous_heart_problems = Column(Boolean, nullable=False)  # Past heart issues

    # --- Socioeconomic & Geographic Data ---
    income = Column(Integer, nullable=False)  # Annual income (USD)
    country = Column(String, nullable=False)  # Patient's country
    continent = Column(String, nullable=False)  # Patient's continent
    hemisphere = Column(String, nullable=False)  # "Northern Hemisphere" or "Southern Hemisphere"

    # --- Target Variable ---
    heart_attack_risk = Column(Boolean, nullable=False)  # Ground truth label for training
