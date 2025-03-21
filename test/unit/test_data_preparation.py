import pytest
import pandas as pd
from backend.data_service.data_preparation import (
    remove_null_values,
    remove_duplicates,
    standardize_column_names,
    remove_patient_id,
    split_blood_pressure,
)

# --------------------------------------
# Test remove_null_values
# --------------------------------------

def test_remove_null_values_removes_rows_with_nan():
    """
    Test that rows with null values are removed.
    """
    df = pd.DataFrame({
        "A": [1, None, 3],
        "B": [4, 5, None]
    })
    result = remove_null_values(df)
    assert result.isnull().sum().sum() == 0
    assert result.shape[0] == 1


# --------------------------------------
# Test remove_duplicates
# --------------------------------------

def test_remove_duplicates_removes_exact_copies():
    """
    Test that duplicate rows are removed.
    """
    df = pd.DataFrame({
        "A": [1, 1, 2],
        "B": [3, 3, 4]
    })
    result = remove_duplicates(df)
    assert result.shape == (2, 2)
    assert not result.duplicated().any()


# --------------------------------------
# Test standardize_column_names
# --------------------------------------

def test_standardize_column_names_lowercase_and_underscores():
    """
    Test that column names are converted to lowercase and spaces to underscores.
    """
    df = pd.DataFrame(columns=["Column One", "Column Two"])
    result = standardize_column_names(df)
    assert list(result.columns) == ["column_one", "column_two"]


# --------------------------------------
# Test remove_patient_id
# --------------------------------------

def test_remove_patient_id_removes_column_if_exists():
    """
    Test that 'patient_id' column is removed when present.
    """
    df = pd.DataFrame({
        "patient_id": ["X123", "Y456"],
        "age": [30, 45]
    })
    result = remove_patient_id(df)
    assert "patient_id" not in result.columns
    assert "age" in result.columns

def test_remove_patient_id_ignores_if_missing():
    """
    Test that function does not raise error if 'patient_id' is not in columns.
    """
    df = pd.DataFrame({"age": [30]})
    result = remove_patient_id(df)
    assert "age" in result.columns


# --------------------------------------
# Test split_blood_pressure
# --------------------------------------

def test_split_blood_pressure_creates_two_numeric_columns():
    """
    Test that a valid blood_pressure column is split into systolic and diastolic parts.
    """
    df = pd.DataFrame({
        "blood_pressure": ["120/80", "140/90"]
    })
    result = split_blood_pressure(df)
    assert "systolic_blood_pressure" in result.columns
    assert "diastolic_blood_pressure" in result.columns
    assert result["systolic_blood_pressure"].tolist() == [120, 140]
    assert result["diastolic_blood_pressure"].tolist() == [80, 90]
    assert "blood_pressure" not in result.columns

def test_split_blood_pressure_handles_invalid_format():
    """
    Test that invalid blood pressure formats do not create new columns.
    """
    df = pd.DataFrame({
        "blood_pressure": ["abc", "130", "120//80", None]
    })
    result = split_blood_pressure(df)

    # Columns should not be created because split failed
    assert "systolic_blood_pressure" not in result.columns
    assert "diastolic_blood_pressure" not in result.columns

def test_split_blood_pressure_ignores_if_column_missing():
    """
    Test that function does nothing if 'blood_pressure' column does not exist.
    """
    df = pd.DataFrame({"age": [50]})
    result = split_blood_pressure(df)
    assert "systolic_blood_pressure" not in result.columns
