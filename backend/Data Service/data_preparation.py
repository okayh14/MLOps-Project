import pandas as pd

path = "initial_data.csv"


def read_data(path):
    """Reads the CSV file and returns a DataFrame."""
    return pd.read_csv(path)


def remove_null_values(df):
    """Removes rows containing null values."""
    return df.dropna()


def remove_duplicates(df):
    """Removes duplicate rows."""
    return df.drop_duplicates()


def standardize_column_names(df):
    """
    Converts column names to lowercase and replaces spaces with underscores.
    """
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    return df


def remove_patient_id(df):
    """Removes the Patient ID column if it exists."""
    return df.drop(columns=["Patient ID"], errors="ignore")


def split_blood_pressure(df):
    """
    Splits the Blood Pressure column into Systolic and Diastolic BP
    if it exists and is a string.
    """
    if "blood_pressure" in df.columns and df["blood_pressure"].dtype == object:
        split_values = df["blood_pressure"].str.split("/", expand=True)

        if split_values.shape[1] == 2:  # Ensure we got exactly two columns
            df["systolic_blood_pressure"] = pd.to_numeric(
                split_values[0], errors="coerce"
            )
            df["diastolic_blood_pressure"] = pd.to_numeric(
                split_values[1], errors="coerce"
            )

        df = df.drop(columns=["blood_pressure"])  # Remove the original column

    return df


def data_preparation():
    """Loads, processes, and returns the cleaned data as JSON."""
    df = read_data(path)
    df = remove_null_values(df)
    df = remove_duplicates(df)
    df = standardize_column_names(df)
    df = remove_patient_id(df)
    df = split_blood_pressure(df)

    # Convert DataFrame to JSON-friendly format
    return df.to_dict(orient="records")
