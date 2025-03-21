import pytest
import pandas as pd
from backend.data_service.data_preparation import data_preparation


@pytest.fixture
def sample_dataframe():
    """
    Creates a DataFrame that simulates the real data structure.
    """
    return pd.DataFrame(
        {
            "Patient ID": ["BMW7812", "CZE1114"],
            "Age": [67, 21],
            "Sex": ["Male", "Male"],
            "Cholesterol": [208, 389],
            "Blood Pressure": ["158/88", "165/93"],
            "Heart Rate": [72, 98],
            "Diabetes": [0, 1],
            "Family History": [0, 1],
            "Smoking": [1, 1],
            "Obesity": [0, 1],
            "Alcohol Consumption": [0, 1],
            "Exercise Hours Per Week": [4.17, 1.81],
            "Diet": ["Average", "Unhealthy"],
            "Previous Heart Problems": [0, 1],
            "Medication Use": [0, 0],
            "Stress Level": [9, 1],
            "Sedentary Hours Per Day": [6.61, 4.96],
            "Income": [261404, 285768],
            "BMI": [31.25, 27.19],
            "Triglycerides": [286, 235],
            "Physical Activity Days Per Week": [0, 1],
            "Sleep Hours Per Day": [6, 7],
            "Country": ["Argentina", "Canada"],
            "Continent": ["South America", "North America"],
            "Hemisphere": ["Southern Hemisphere", "Northern Hemisphere"],
            "Heart Attack Risk": [0, 0],
        }
    )


def test_data_preparation(sample_dataframe):
    """
    Integration test for the complete data_preparation pipeline
    using realistic input data.
    """
    result = data_preparation(sample_dataframe)

    # Check shape: no null values, no duplicates, 2 rows should remain
    assert result.shape[0] == 2

    # Check that column names are standardized
    assert all(" " not in col for col in result.columns)
    assert all(col == col.lower() for col in result.columns)

    # Check that patient_id and blood_pressure columns are removed
    assert "patient_id" not in result.columns
    assert "blood_pressure" not in result.columns

    # Check that new blood pressure columns are present
    assert "systolic_blood_pressure" in result.columns
    assert "diastolic_blood_pressure" in result.columns

    # Check correct blood pressure values
    assert result["systolic_blood_pressure"].tolist() == [158, 165]
    assert result["diastolic_blood_pressure"].tolist() == [88, 93]
    