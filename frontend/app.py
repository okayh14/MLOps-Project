import streamlit as st
import requests
import os
import random
import string

ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8003")

def generate_unique_id(length=8):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

def build_form():
    st.subheader("Patientendaten")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cholesterol = st.number_input("Cholesterol", min_value=0, max_value=1000, value=200)
        blood_pressure = st.text_input("Blood Pressure (Systolic/Diastolic)", "120/80")
        heart_rate = st.number_input("Heart Rate", min_value=0, max_value=300, value=70)
        diabetes = st.selectbox("Diabetes", ["False", "True"])
        family_history = st.selectbox("Family History", ["False", "True"])
        smoking = st.selectbox("Smoking", ["False", "True"])
        obesity = st.selectbox("Obesity", ["False", "True"])
        alcohol_consumption = st.selectbox("Alcohol Consumption", ["False", "True"])
        exercise_hours_per_week = st.number_input("Exercise Hours Per Week", min_value=0.0, value=3.0)
        diet = st.selectbox("Diet", ["Balanced", "Average", "Unhealthy"])

    with col2:
        previous_heart_problems = st.selectbox("Previous Heart Problems", ["False", "True"])
        medication_use = st.selectbox("Medication Use", ["False", "True"])
        stress_level = st.number_input("Stress Level (0-20)", min_value=0, max_value=20, value=5)
        sedentary_hours_per_day = st.number_input("Sedentary Hours Per Day", min_value=0.0, value=4.0)
        income = st.number_input("Income", min_value=0, value=30000)
        bmi = st.number_input("BMI", min_value=0.0, value=25.0)
        triglycerides = st.number_input("Triglycerides", min_value=0, max_value=10000, value=150)
        physical_activity_days_per_week = st.number_input("Physical Activity Days Per Week", min_value=0, max_value=7, value=3)
        sleep_hours_per_day = st.number_input("Sleep Hours Per Day", min_value=0, max_value=24, value=7)
        country = st.selectbox("Country", ["USA", "Germany", "France", "India", "Other"])
        continent = st.selectbox("Continent", ["North America", "Europe", "Asia", "Africa", "South America"])
        hemisphere = st.selectbox("Hemisphere", ["Northern Hemisphere", "Southern Hemisphere"])

    return {
        "age": age,
        "sex": sex,
        "cholesterol": cholesterol,
        "blood_pressure": blood_pressure,
        "heart_rate": heart_rate,
        "diabetes": diabetes == "True",
        "family_history": family_history == "True",
        "smoking": smoking == "True",
        "obesity": obesity == "True",
        "alcohol_consumption": alcohol_consumption == "True",
        "exercise_hours_per_week": exercise_hours_per_week,
        "diet": diet,
        "previous_heart_problems": previous_heart_problems == "True",
        "medication_use": medication_use == "True",
        "stress_level": stress_level,
        "sedentary_hours_per_day": sedentary_hours_per_day,
        "income": income,
        "bmi": bmi,
        "triglycerides": triglycerides,
        "physical_activity_days_per_week": physical_activity_days_per_week,
        "sleep_hours_per_day": sleep_hours_per_day,
        "country": country,
        "continent": continent,
        "hemisphere": hemisphere
    }

def main():
    st.title("📈 Herzinfarkt-Risiko Vorhersage & Datenpflege")
    mode = st.radio("Was möchtest du tun?", ["Prognose erstellen", "Neue Daten einpflegen & Training starten"])

    with st.form("patient_form"):
        form_data = build_form()

        if mode == "Neue Daten einpflegen & Training starten":
            heart_attack_risk = st.selectbox("Heart Attack Risk", ["False", "True"])
            form_data["heart_attack_risk"] = heart_attack_risk == "True"

        submitted = st.form_submit_button("Absenden")

        if submitted:
            form_data["patient_id"] = generate_unique_id()

            if mode == "Prognose erstellen":
                # Vorhersage benötigt formatierte Keys
                prediction_payload = [{
                    k.replace("_", " ").title(): v for k, v in form_data.items()
                }]

                try:
                    st.info("Starte Vorhersage...")
                    response = requests.post(f"{ORCHESTRATOR_URL}/start_inference", json=prediction_payload, timeout=500)
                    response.raise_for_status()
                    res = response.json()

                    st.success("Vorhersage abgeschlossen!")
                    st.write("**Zusammenfassung:**", res.get("result_text"))
                    st.write("**Durchschnittliches Risiko:**", res.get("mean_proba"))
                    st.write("**Modelle:**")
                    st.dataframe(res.get("final_results"))

                except Exception as e:
                    st.error(f"Fehler bei der Vorhersage: {e}")

            else:
                # Training benötigt originalen Payload (snake_case)
                try:
                    st.warning("Training wird gestartet... Bitte Seite nicht verlassen oder aktualisieren.")
                    with st.spinner("Lade neue Daten hoch und starte Training..."):
                        response = requests.post(f"{ORCHESTRATOR_URL}/trigger_upload", json=form_data, timeout=None)
                        response.raise_for_status()
                        result = response.json()

                    st.success("Training erfolgreich abgeschlossen!")
                    st.write(result)
                except Exception as e:
                    st.error(f"Fehler beim Hochladen oder Training: {e}")

if __name__ == "__main__":
    main()
