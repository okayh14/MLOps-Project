from sqlalchemy import Column, Integer, Boolean, String, Float
from backend.data_service.database import Base  # Direkt aus database.py importieren


class PatientData(Base):
    __tablename__ = "heart_risk"

    patient_id = Column(String, primary_key=True, index=True)
    age = Column(Integer, nullable=False)
    sex = Column(String, nullable=False)
    cholesterol = Column(Integer, nullable=False)
    blood_pressure = Column(String, nullable=False)  # Werte wie "158/88"
    heart_rate = Column(Integer, nullable=False)
    diabetes = Column(Boolean, nullable=False)
    family_history = Column(Boolean, nullable=False)
    smoking = Column(Boolean, nullable=False)
    obesity = Column(Boolean, nullable=False)
    alcohol_consumption = Column(Boolean, nullable=False)
    exercise_hours_per_week = Column(Float, nullable=False)
    diet = Column(String, nullable=False)
    previous_heart_problems = Column(Boolean, nullable=False)
    medication_use = Column(Boolean, nullable=False)
    stress_level = Column(Integer, nullable=False)
    sedentary_hours_per_day = Column(Float, nullable=False)
    income = Column(Integer, nullable=False)
    bmi = Column(Float, nullable=False)
    triglycerides = Column(Integer, nullable=False)
    physical_activity_days_per_week = Column(Integer, nullable=False)
    sleep_hours_per_day = Column(Integer, nullable=False)
    country = Column(String, nullable=False)
    continent = Column(String, nullable=False)
    hemisphere = Column(String, nullable=False)
    heart_attack_risk = Column(Boolean, nullable=False)
