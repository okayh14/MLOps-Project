import streamlit as st
import requests
import os
import random
import string
import re
import pandas as pd
import plotly.express as px

# --- GLOBAL CONFIG ---
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8003")
st.set_page_config(layout="wide")

# --- CSS for error highlight and improved UI ---
st.markdown(
    """
    <style>
    .error-input input, .error-input select {
        border: 2px solid #ff4b4b !important;
        background-color: #fff0f0 !important;
    }
    
    .center-title {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .form-section {
        padding: 0;
        border: none;
        box-shadow: none;
        background-color: transparent;
    }
    
    .results-container {
        padding: 0;
        border: none;
        box-shadow: none;
        background-color: transparent;
    }
    
    .button-container {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-top: 20px;
    }
    
    .submit-button {
        background-color: #4CAF50 !important;
        color: white !important;
        padding: 10px 30px !important;
        border-radius: 30px !important;
        border: none !important;
        font-weight: 600 !important;
        width: 200px !important;
    }
    
    /* Remove white space between sections */
    div.block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Remove background color from streamlit containers */
    .stApp {
        background-color: transparent;
    }
    
    /* Remove padding between sections */
    section.main > div {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    
    /* Customize the submit button in forms */
    div.stButton button[kind="primaryFormSubmit"] {
        background-color: #4CAF50 !important;
        color: white !important;
        padding: 10px 30px !important;
        border-radius: 30px !important;
        border: none !important;
        font-weight: 600 !important;
        width: 200px !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# --- ID generator ---
def generate_unique_id(length=8):
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


# --- VALIDATION LOGIC ---
def validate_form(data):
    errors = {}

    if data["age"] is None or data["age"] <= 0:
        errors["age"] = "Please enter a valid age greater than 0."

    if not re.match(r"^\d{2,3}/\d{2,3}$", data["blood_pressure"] or ""):
        errors["blood_pressure"] = "Use format like 120/80 (systolic/diastolic)."

    # Modified validation - Added fields that cannot be zero
    required_fields = ["cholesterol", "heart_rate"]

    for field in required_fields:
        if data.get(field) in [None, ""]:
            errors[field] = "This field is required."

    # Validate ranges
    if data.get("cholesterol") is not None and data["cholesterol"] > 1000:
        errors["cholesterol"] = "Please enter a valid cholesterol level (< 1000 mg/dL)."

    if data.get("heart_rate") is not None and (
        data["heart_rate"] < 20 or data["heart_rate"] > 250
    ):
        errors["heart_rate"] = "Please enter a valid heart rate (20-250 bpm)."

    # Add validation for fields that cannot be zero
    if data.get("bmi") is not None and data["bmi"] <= 0:
        errors["bmi"] = "BMI must be greater than 0."

    if data.get("sleep_hours_per_day") is not None and data["sleep_hours_per_day"] <= 0:
        errors["sleep_hours_per_day"] = "Sleep hours must be greater than 0."

    return errors


# --- Check for empty required fields on form render ---
def check_empty_fields():
    # Get the session state keys that contain form field values
    form_fields = [
        key
        for key in st.session_state.keys()
        if key
        in [
            "age",
            "blood_pressure",
            "cholesterol",
            "heart_rate",
            "bmi",
            "sleep_hours_per_day",
        ]
    ]

    # Check if any required fields are empty and update errors
    for field in form_fields:
        value = st.session_state.get(field)
        if field == "age" and (value is None or value <= 0):
            st.session_state.errors[field] = "Please enter a valid age greater than 0."
        elif field == "blood_pressure" and not re.match(
            r"^\d{2,3}/\d{2,3}$", value or ""
        ):
            st.session_state.errors[
                field
            ] = "Use format like 120/80 (systolic/diastolic)."
        elif field == "cholesterol" and (value is None or value == ""):
            st.session_state.errors[field] = "This field is required."
        elif field == "heart_rate" and (value is None or value < 20):
            st.session_state.errors[
                field
            ] = "Please enter a valid heart rate (20-250 bpm)."
        elif field == "bmi" and (value is None or value <= 0):
            st.session_state.errors[field] = "BMI must be greater than 0."
        elif field == "sleep_hours_per_day" and (value is None or value <= 0):
            st.session_state.errors[field] = "Sleep hours must be greater than 0."


# --- FORM ---
def build_form(errors=None, is_training_mode=False):
    errors = errors or {}
    st.markdown('<div class="form-section">', unsafe_allow_html=True)
    st.subheader("📋 Patient Information")

    form_data = {}

    # Create 3 columns for more breathing space
    col1, col2, col3 = st.columns(3)

    # Column 1
    with col1:
        # Age field
        if "age" in errors:
            st.markdown('<div class="error-input">', unsafe_allow_html=True)
        form_data["age"] = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,  # Changed min_value to 1
            key="age",
            help="Enter the patient's age in years",
            placeholder="e.g. 45",
        )
        if "age" in errors:
            st.markdown("</div>", unsafe_allow_html=True)
            st.error(errors["age"])

        # Sex field
        form_data["sex"] = st.selectbox("Sex", ["Male", "Female"], key="sex")

        # Cholesterol field
        if "cholesterol" in errors:
            st.markdown('<div class="error-input">', unsafe_allow_html=True)
        form_data["cholesterol"] = st.number_input(
            "Cholesterol (mg/dL)",
            min_value=0,
            max_value=1000,
            key="cholesterol",
            help="Normal is below 200 mg/dL",
        )
        if "cholesterol" in errors:
            st.markdown("</div>", unsafe_allow_html=True)
            st.error(errors["cholesterol"])

        # Blood pressure field
        if "blood_pressure" in errors:
            st.markdown('<div class="error-input">', unsafe_allow_html=True)
        form_data["blood_pressure"] = st.text_input(
            "Blood Pressure (Systolic/Diastolic)",
            key="blood_pressure",
            placeholder="e.g. 120/80",
            help="Enter in format like 120/80",
        )
        if "blood_pressure" in errors:
            st.markdown("</div>", unsafe_allow_html=True)
            st.error(errors["blood_pressure"])

        # Heart rate field
        if "heart_rate" in errors:
            st.markdown('<div class="error-input">', unsafe_allow_html=True)
        form_data["heart_rate"] = st.number_input(
            "Heart Rate (bpm)",
            min_value=20,
            max_value=250,
            key="heart_rate",
            help="Normal resting heart rate is 60-100 bpm",
        )
        if "heart_rate" in errors:
            st.markdown("</div>", unsafe_allow_html=True)
            st.error(errors["heart_rate"])

    # Column 2
    with col2:
        form_data["diabetes"] = st.selectbox(
            "Diabetes", ["True", "False"], key="diabetes"
        )

        form_data["family_history"] = st.selectbox(
            "Family History", ["True", "False"], key="family_history"
        )

        form_data["smoking"] = st.selectbox("Smoking", ["True", "False"], key="smoking")

        form_data["obesity"] = st.selectbox("Obesity", ["True", "False"], key="obesity")

        # Stress level field - can be zero
        form_data["stress_level"] = st.number_input(
            "Stress Level (0-20)", min_value=0, max_value=20, key="stress"
        )

    # Column 3
    with col3:
        form_data["alcohol_consumption"] = st.selectbox(
            "Alcohol Consumption", ["True", "False"], key="alcohol_consumption"
        )

        # Exercise hours field - can be zero
        form_data["exercise_hours_per_week"] = st.number_input(
            "Exercise Hours Per Week", min_value=0.0, max_value=50.0, key="exercise"
        )

        form_data["diet"] = st.selectbox(
            "Diet", ["Balanced", "Average", "Unhealthy"], key="diet"
        )

        form_data["previous_heart_problems"] = st.selectbox(
            "Previous Heart Problems", ["True", "False"], key="prev_heart"
        )

        form_data["medication_use"] = st.selectbox(
            "Medication Use", ["True", "False"], key="meds"
        )

    # Create a new row for additional fields
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div class="form-section">', unsafe_allow_html=True)
    st.subheader("📊 Additional Health Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Sedentary hours field - can be zero
        form_data["sedentary_hours_per_day"] = st.number_input(
            "Sedentary Hours/Day", min_value=0.0, max_value=24.0, key="sedentary"
        )

        # Income field - can be zero
        form_data["income"] = st.number_input(
            "Income per Year ($)", min_value=0, key="income"
        )

    with col2:
        # BMI field - Updated min_value to 1
        if "bmi" in errors:
            st.markdown('<div class="error-input">', unsafe_allow_html=True)
        form_data["bmi"] = st.number_input(
            "BMI",
            min_value=1.0,
            max_value=50.0,
            key="bmi",
            help="BMI must be greater than 0",
        )
        if "bmi" in errors:
            st.markdown("</div>", unsafe_allow_html=True)
            st.error(errors["bmi"])

        # Triglycerides field - can be zero
        form_data["triglycerides"] = st.number_input(
            "Triglycerides", min_value=0, max_value=1000, key="trigly"
        )

    with col3:
        # Physical activity field - can be zero
        form_data["physical_activity_days_per_week"] = st.number_input(
            "Activity Days/Week", min_value=0, max_value=7, key="activity"
        )

        # Sleep hours field - Updated min_value to 1
        if "sleep_hours_per_day" in errors:
            st.markdown('<div class="error-input">', unsafe_allow_html=True)
        form_data["sleep_hours_per_day"] = st.number_input(
            "Sleep Hours/Day",
            min_value=1,
            max_value=24,
            key="sleep",
            help="Sleep hours must be greater than 0",
        )
        if "sleep_hours_per_day" in errors:
            st.markdown("</div>", unsafe_allow_html=True)
            st.error(errors["sleep_hours_per_day"])

    # Geographic information
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div class="form-section">', unsafe_allow_html=True)
    st.subheader("📍 Geographic Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        form_data["country"] = st.selectbox(
            "Country",
            [
                "Germany",
                "Argentina",
                "Brazil",
                "Spain",
                "Nigeria",
                "United Kingdom",
                "Australia",
                "France",
                "Canada",
                "China",
                "New Zealand",
                "Japan",
                "Italy",
                "Colombia",
                "Thailand",
                "South Africa",
                "Vietnam",
                "USA",
                "South Korea",
            ],
            key="country",
        )

    with col2:
        form_data["continent"] = st.selectbox(
            "Continent",
            ["North America", "Europe", "Asia", "Africa", "South America", "Australia"],
            key="continent",
        )

    with col3:
        form_data["hemisphere"] = st.selectbox(
            "Hemisphere", ["Northern Hemisphere", "Southern Hemisphere"], key="hemi"
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # Add ground truth option for retraining mode
    if is_training_mode:
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        st.subheader("📋 Training Data Information")
        form_data["heart_attack_risk"] = st.selectbox(
            "Heart Attack Risk (Actual Outcome)", ["True", "False"], key="har"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    return form_data


# --- RESULTS VISUALIZATION ---
def display_results(res):
    st.markdown('<div class="results-container">', unsafe_allow_html=True)

    # Display the prediction summary
    st.markdown("### 📋 Prediction Summary")
    st.write(res.get("result_text"))

    # Display the risk metric
    col1, col2 = st.columns([1, 3])
    with col1:
        risk_percentage = round(res.get("mean_proba", 0) * 100, 2)
        st.metric(
            label="Risk Probability",
            value=f"{risk_percentage}%",
            delta="Risk" if risk_percentage > 30 else "Low Risk",
        )

    # Create a color for the risk level
    risk_color = "#4CAF50"  # Green for low risk
    if risk_percentage > 30:
        risk_color = "#FFC107"  # Yellow for medium risk
    if risk_percentage > 60:
        risk_color = "#F44336"  # Red for high risk

    # Display the risk gauge
    with col2:
        st.markdown(
            f"""
            <div style="width:100%; background-color:#e0e0e0; height:20px; border-radius:10px; margin-top:30px;">
                <div style="width:{risk_percentage}%; background-color:{risk_color}; height:20px; border-radius:10px;">
                </div>
            </div>
            <div style="display:flex; justify-content:space-between; margin-top:5px;">
                <span>Low Risk</span>
                <span>Medium Risk</span>
                <span>High Risk</span>
            </div>
        """,
            unsafe_allow_html=True,
        )

    # Display model predictions
    st.markdown("### 🤖 Model Predictions")

    try:
        df = pd.DataFrame(res.get("final_results"))

        # Rename classifiers to Model 1-5
        model_names = {df["Classifier"].iloc[i]: f"Model {i+1}" for i in range(len(df))}
        df["Classifier"] = df["Classifier"].map(model_names)

        # Pastel colors for models
        colors = [
            "#95C1E1",
            "#95E1C1",
            "#E1C195",
            "#C195E1",
            "#E195C1",
        ]  # Pastel blue, green, orange, purple, pink
        colors_dict = {f"Model {i+1}": colors[i] for i in range(min(len(df), 5))}

        # Bar chart for model predictions with more compact layout
        fig = px.bar(
            df,
            y="Classifier",
            x="Probability_At_Risk",
            orientation="h",
            title="Risk Assessment by Model",
            labels={"Probability_At_Risk": "Risk Probability", "Classifier": "Model"},
            color="Classifier",
            color_discrete_map=colors_dict,
        )

        fig.update_layout(
            height=400,  # More compact height
            width=800,  # Consistent width
            yaxis={"categoryorder": "total ascending"},
            xaxis_title="Risk Probability (0-1)",
            yaxis_title="Model",
            showlegend=False,
            margin=dict(l=50, r=50, t=50, b=50),
            font=dict(size=16),  # Larger font size for axis labels
        )

        # Adjust bars appearance - no text inside bars
        fig.update_traces(
            textposition="none",  # No text in bars
            marker_line_width=1,  # Add border to bars
            marker_line_color="white",  # White border for contrast
        )

        # Make axes titles larger
        fig.update_xaxes(title_font=dict(size=18))
        fig.update_yaxes(title_font=dict(size=18))

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error displaying predictions: {str(e)}")

    st.markdown("</div>", unsafe_allow_html=True)


# Main function to run the app
def main():
    # Initialize session state
    if "form_data" not in st.session_state:
        st.session_state.form_data = {}
    if "errors" not in st.session_state:
        st.session_state.errors = {}
    if "results" not in st.session_state:
        st.session_state.results = None
    if "show_results" not in st.session_state:
        st.session_state.show_results = False
    if "error_displayed" not in st.session_state:
        st.session_state.error_displayed = False

    # Set page title
    st.markdown(
        "<h1 class='center-title'>Heart Attack Risk Prediction</h1>",
        unsafe_allow_html=True,
    )

    # Introduction / Welcome section
    st.markdown("### 👋 Welcome")
    st.markdown(
        """
This application is designed to support **healthcare professionals** in assessing the risk of heart attacks using AI-powered predictive models.
It offers two main functionalities:
"""
    )

    st.markdown("#### 🧠 Prediction Page")
    st.markdown(
        """
Use the **Prediction** page to enter patient information and receive a machine learning-based estimation of the heart attack risk.
This prediction is generated by multiple trained models, helping to guide early preventive care and risk management.

⚠️ **Disclaimer**: This is a clinical support tool, not a replacement for professional diagnosis. Always use medical judgment.
"""
    )

    st.markdown("#### 📥 Add Data & Retrain Page")
    st.markdown(
        """
On this page, you can contribute anonymized data for patients with a confirmed heart attack risk.
Each new entry automatically triggers a model retraining process to incorporate the most recent data and maintain predictive accuracy.

🕒 **Please note**: Retraining typically takes **6–10 minutes**, during which predictions are **temporarily unavailable**.  
For routine clinical use, we recommend scheduling data uploads **outside of core working hours**—for example, at the start or end of the day—so as not to interrupt diagnostic workflows.
"""
    )

    st.markdown("#### 💬 Support & Feedback")
    st.markdown(
        """
If you experience any issues or would like to provide feedback, please don't hesitate to contact our support team.
We're continuously improving the tool to better serve clinical needs. Supportmail: mlops@fh-swf.de
"""
    )

    st.markdown("<hr style='margin-top:2rem;'>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
