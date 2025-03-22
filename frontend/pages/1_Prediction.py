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

# Set page title
st.markdown(
    "<h1 class='center-title'>Heart Attack Risk Prediction</h1>", unsafe_allow_html=True
)

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

# Check for empty fields when the form is initially rendered
check_empty_fields()

# Form section
with st.form("prediction_form", clear_on_submit=False):
    # Build form with any existing errors
    form_data = build_form(st.session_state.errors, is_training_mode=False)

    # Submit button
    col1, col2, col3 = st.columns([5, 2, 5])
    with col2:
        st.markdown('<div class="button-container">', unsafe_allow_html=True)
        submitted = st.form_submit_button("Submit")
        st.markdown("</div>", unsafe_allow_html=True)

# Handle form submission
if submitted:
    # Generate a unique patient ID
    form_data["patient_id"] = generate_unique_id()

    # Validate the form data
    errors = validate_form(form_data)

    # Set error state
    if errors:
        st.session_state.errors = errors
        st.session_state.error_displayed = True
        st.error("Please correct the highlighted fields.")
    else:
        # Clear errors when form is valid
        st.session_state.errors = {}
        st.session_state.error_displayed = False

        # Convert snake_case to Title Case for API
        payload = [{k.replace("_", " ").title(): v for k, v in form_data.items()}]

        with st.spinner("Running prediction..."):
            try:
                # Make the prediction request
                response = requests.post(
                    f"{ORCHESTRATOR_URL}/start_inference", json=payload, timeout=500
                )
                response.raise_for_status()

                # Store results in session state
                st.session_state.results = response.json()
                st.session_state.show_results = True

                st.success("Prediction complete!")

                # Update session state
                st.session_state.form_data = form_data

            except Exception as e:
                st.error(f"Error while predicting: {str(e)}")

# Display results if available
if st.session_state.show_results and st.session_state.results:
    display_results(st.session_state.results)
