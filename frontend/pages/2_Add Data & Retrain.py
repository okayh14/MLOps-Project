import streamlit as st
import requests
import sys
import os

# Add parent directory to path to import shared methods from Home.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Home import validate_form, build_form, check_empty_fields, generate_unique_id

# --- GLOBAL CONFIG ---
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8003")

# --- PAGE SETUP ---
st.markdown(
    "<h1 class='center-title'>Add Data & Retrain Model</h1>", unsafe_allow_html=True
)

# --- RESET training results on page load ---
st.session_state.training_results = None

# --- Initialize session state ---
if "form_data" not in st.session_state:
    st.session_state.form_data = {}
if "errors" not in st.session_state:
    st.session_state.errors = {}
if "error_displayed" not in st.session_state:
    st.session_state.error_displayed = False

# --- Live validation of critical fields on page load ---
check_empty_fields()

# --- FORM SECTION ---
with st.form("training_form", clear_on_submit=False):
    # Build form with validation and training mode flag
    form_data = build_form(st.session_state.errors, is_training_mode=True)

    # Submit button in the center
    col1, col2, col3 = st.columns([5, 2, 5])
    with col2:
        st.markdown('<div class="button-container">', unsafe_allow_html=True)
        submitted = st.form_submit_button("Submit")
        st.markdown("</div>", unsafe_allow_html=True)

# --- HANDLE FORM SUBMISSION ---
if submitted:
    # Generate a unique ID for the patient
    form_data["patient_id"] = generate_unique_id()

    # Validate form fields
    errors = validate_form(form_data)

    # Display and store errors if present
    if errors:
        st.session_state.errors = errors
        st.session_state.error_displayed = True
        st.error("Please correct the highlighted fields.")
    else:
        # Clear any previous error state
        st.session_state.errors = {}
        st.session_state.error_displayed = False

        # Convert 'True'/'False' string to boolean for risk label
        if "heart_attack_risk" in form_data and isinstance(
            form_data["heart_attack_risk"], str
        ):
            form_data["heart_attack_risk"] = form_data["heart_attack_risk"] == "True"

        # --- Inform user about retraining impact ---
        st.warning(
            "⚠️ New patient data has triggered a model retraining process.\n\n"
            "Please do not refresh or leave this page during training. "
            "Predictions are temporarily unavailable until training is completed."
        )

        with st.spinner("Uploading data and retraining the model..."):
            try:
                # Send the training data to the orchestrator
                response = requests.post(
                    f"{ORCHESTRATOR_URL}/trigger_upload", json=form_data, timeout=None
                )
                response.raise_for_status()

                # Save training result in session state
                st.session_state.training_results = response.json()
                st.session_state.form_data = form_data

                # Notify user of success
                st.success(
                    "✅ Models have been successfully updated. You can now perform new predictions."
                )

            except Exception as e:
                st.error(f"❌ Error during training: {str(e)}")

# --- TRAINING RESULTS DISPLAY (now always reset on page load) ---
if st.session_state.training_results:
    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    st.markdown("### 📊 Training Results")

    result_data = st.session_state.training_results
    if isinstance(result_data, dict):
        for key, value in result_data.items():
            st.markdown(f"**{key.replace('_', ' ').title()}**: {value}")
    else:
        st.write(result_data)

    st.markdown("</div>", unsafe_allow_html=True)
