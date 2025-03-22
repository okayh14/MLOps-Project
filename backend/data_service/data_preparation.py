import pandas as pd

def remove_null_values(df):
    """
    Removes rows from the DataFrame that contain any null (NaN) values.
    Ensures data completeness before further processing.
    """
    return df.dropna()


def remove_duplicates(df):
    """
    Removes duplicate rows from the DataFrame to avoid redundant training data.
    """
    return df.drop_duplicates()


def standardize_column_names(df):
    """
    Standardizes column names by converting them to lowercase and 
    replacing spaces with underscores. Ensures consistency across the pipeline.
    """
    df.columns = [str(col).lower().replace(" ", "_") for col in df.columns]
    return df


def remove_patient_id(df):
    """
    Removes the 'patient_id' column if it exists.
    This column is typically used for identification and not needed for training.
    """
    return df.drop(columns=["patient_id"], errors="ignore")


def split_blood_pressure(df):
    """
    Splits the 'blood_pressure' column into two separate columns:
    'systolic_blood_pressure' and 'diastolic_blood_pressure'.
    Assumes the format is 'systolic/diastolic' (e.g., '120/80').
    Non-numeric or malformed values are coerced to NaN.
    """
    if "blood_pressure" in df.columns and df["blood_pressure"].dtype == object:
        split_values = df["blood_pressure"].str.split("/", expand=True)

        # Ensure the split results in exactly two values
        if split_values.shape[1] == 2:
            df["systolic_blood_pressure"] = pd.to_numeric(
                split_values[0], errors="coerce"
            )
            df["diastolic_blood_pressure"] = pd.to_numeric(
                split_values[1], errors="coerce"
            )

        # Remove the original column after splitting
        df = df.drop(columns=["blood_pressure"])

    return df


def data_preparation(df: pd.DataFrame, drop_target: bool = False) -> pd.DataFrame:
    """
    Prepares patient data for machine learning by applying cleaning and transformation steps:
    - Removes null values and duplicates
    - Standardizes column names
    - Removes identifiers like 'patient_id'
    - Splits blood pressure into systolic and diastolic values
    - Optionally drops the target column ('heart_attack_risk') for inference

    Parameters:
        df (pd.DataFrame): Raw input data
        drop_target (bool): If True, the 'heart_attack_risk' column will be removed (e.g. for prediction)

    Returns:
        pd.DataFrame: Cleaned and processed dataset
    """
    df = remove_null_values(df)
    df = remove_duplicates(df)
    df = standardize_column_names(df)
    df = remove_patient_id(df)
    df = split_blood_pressure(df)

    # Remove the target variable when performing inference (optional)
    if drop_target and "heart_attack_risk" in df.columns:
        df = df.drop(columns=["heart_attack_risk"])

    return df
