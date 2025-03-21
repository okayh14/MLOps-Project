import streamlit as st
import requests
import os

# Wir holen uns die URL des Orchestrators aus einer ENV-Variable
# (oder defaulten zu http://localhost:8003, falls nicht gesetzt)
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8003")

def main():
    st.title("Heart Disease Risk Predictor")

    st.markdown("""
        Dieses Dashboard erlaubt es dir, Patientendaten einzugeben und 
        eine Risikoeinschätzung für Herzinfarkte zu erhalten.
    """)

    # Formular für Patientendaten
    patient_id = st.text_input("Patient ID", value="P0001")
    age = st.number_input("Age", min_value=0, max_value=120, value=50)
    sex = st.selectbox("Sex", ["male", "female"])  
    cholesterol = st.number_input("Cholesterol", min_value=0, max_value=1000, value=200)
    blood_pressure = st.text_input("Blood Pressure (Systolic/Diastolic)", "120/80")
    heart_rate = st.number_input("Heart Rate", min_value=0, max_value=300, value=70)
    diabetes = st.checkbox("Diabetes?", value=False)
    family_history = st.checkbox("Family History?", value=False)
    smoking = st.checkbox("Smoking?", value=False)
    obesity = st.checkbox("Obesity?", value=False)
    alcohol_consumption = st.checkbox("Alcohol Consumption?", value=False)
    exercise_hours_per_week = st.number_input("Exercise hours/week", min_value=0.0, value=3.0)
    diet = st.text_input("Diet", "balanced")
    previous_heart_problems = st.checkbox("Previous heart problems?", value=False)
    medication_use = st.checkbox("Medication use?", value=False)
    stress_level = st.number_input("Stress Level (0-10?)", min_value=0, max_value=20, value=5)
    sedentary_hours_per_day = st.number_input("Sedentary Hours/Day", min_value=0.0, value=4.0)
    income = st.number_input("Income", value=30000)
    bmi = st.number_input("BMI", value=25.0)
    triglycerides = st.number_input("Triglycerides", min_value=0, max_value=10000, value=150)
    physical_activity_days_per_week = st.number_input("Physical Activity Days/Week", min_value=0, max_value=7, value=3)
    sleep_hours_per_day = st.number_input("Sleep Hours/Day", min_value=0, max_value=24, value=7)
    country = st.text_input("Country", "USA")
    continent = st.text_input("Continent", "North America")
    hemisphere = st.text_input("Hemisphere", "Western")

    if st.button("Predict Risk"):
        # Wir packen alle Felder in ein Dictionary, so wie es im Backend erwartet wird.
        patient_data = {
            "patient_id": patient_id,
            "age": age,
            "sex": sex,
            "cholesterol": cholesterol,
            "blood_pressure": blood_pressure,
            "heart_rate": heart_rate,
            "diabetes": diabetes,
            "family_history": family_history,
            "smoking": smoking,
            "obesity": obesity,
            "alcohol_consumption": alcohol_consumption,
            "exercise_hours_per_week": exercise_hours_per_week,
            "diet": diet,
            "previous_heart_problems": previous_heart_problems,
            "medication_use": medication_use,
            "stress_level": stress_level,
            "sedentary_hours_per_day": sedentary_hours_per_day,
            "income": income,
            "bmi": bmi,
            "triglycerides": triglycerides,
            "physical_activity_days_per_week": physical_activity_days_per_week,
            "sleep_hours_per_day": sleep_hours_per_day,
            "country": country,
            "continent": continent,
            "hemisphere": hemisphere,
            
        }

        # Rufe den Orchestrator an, um /start_inference zu triggern.
        try:
            # /start_inference erwartet eine Liste[Dict], also packen wir patient_data in eine Liste
            inference_url = f"{ORCHESTRATOR_URL}/start_inference"
            response = requests.post(inference_url, json=[patient_data], timeout=120)
            response.raise_for_status()
            result_json = response.json()

            st.subheader("Resultat der Vorhersage:")
            st.write(result_json.get("result_text", "Keine Antwort erhalten."))
            st.write("Detaillierte Ergebnisse:", result_json.get("final_results"))
            st.write("Durchschnittliche Risiko-Wahrscheinlichkeit:", result_json.get("mean_proba"))
        
        except requests.exceptions.RequestException as err:
            st.error(f"Fehler beim Request: {err}")

if __name__ == "__main__":
    main()
