import streamlit as st
import requests
import sys
import os

# Add parent directory to path to import from Home.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Home import (
    validate_form,
    build_form,
    check_empty_fields,
    display_results,
    generate_unique_id,
)

# --- GLOBAL CONFIG ---
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8003")

# --- PAGE SETUP ---
st.markdown(
    "<h1 class='center-title'>Heart Attack Risk Prediction</h1>", unsafe_allow_html=True
)

# --- RESET prediction results on page load ---
st.session_state.results = None
st.session_state.show_results = False

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
with st.form("prediction_form", clear_on_submit=False):
    # Build form with any existing errors
    form_data = build_form(st.session_state.errors, is_training_mode=False)

    # Submit button
    col1, col2, col3 = st.columns([5, 2, 5])
    with col2:
        st.markdown('<div class="button-container">', unsafe_allow_html=True)
        submitted = st.form_submit_button("Submit")
        st.markdown("</div>", unsafe_allow_html=True)

# --- HANDLE FORM SUBMISSION ---
if submitted:
    # Generate a unique patient ID
    form_data["patient_id"] = generate_unique_id()

    # Validate the form data
    errors = validate_form(form_data)

    # Handle form errors
    if errors:
        st.session_state.errors = errors
        st.session_state.error_displayed = True
        st.error("Please correct the highlighted fields.")
    else:
        st.session_state.errors = {}
        st.session_state.error_displayed = False

        payload = [{k.replace("_", " ").title(): v for k, v in form_data.items()}]

        with st.spinner("Running prediction..."):
            try:
                response = requests.post(
                    f"{ORCHESTRATOR_URL}/start_inference", json=payload, timeout=500
                )
                response.raise_for_status()

                st.session_state.results = response.json()
                st.session_state.show_results = True
                st.success("Prediction complete!")
                st.session_state.form_data = form_data

            except Exception as e:
                st.error(f"Error while predicting: {str(e)}")

# --- SHOW RESULTS ---
if st.session_state.show_results and st.session_state.results:
    display_results(st.session_state.results)
