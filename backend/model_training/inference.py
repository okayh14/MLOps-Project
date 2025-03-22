import pandas as pd
import joblib
import os
import json
from typing import List, Dict, Any


def prepare_and_predict(features: List[Dict[str, Any]], serialized_models_dir: str):
    """
    Loads serialized models and applies them to the provided feature set for inference.
    
    Parameters:
        features (List[Dict[str, Any]]): A list of feature dictionaries (one per patient).
        serialized_models_dir (str): Path to the directory containing serialized models and feature metadata.

    Returns:
        pd.DataFrame: Concatenated predictions from all models with corresponding model names.
    """

    # Convert input features into a DataFrame
    prediction_df = pd.DataFrame(features)

    # Dynamically load the expected column names used during training
    try:
        with open(os.path.join(serialized_models_dir, "model_features.json"), "r") as f:
            training_columns = json.load(f)["columns"]
        print("[INFO] Loaded training columns from model_features.json")
    except Exception as e:
        raise RuntimeError(f"Could not load model_features.json: {e}")

    # Validate that all required columns are present in the input
    missing_cols = set(training_columns) - set(prediction_df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in input: {missing_cols}")

    # Ensure the DataFrame has the columns in the same order as training
    prediction_df = prediction_df[training_columns]

    results = []

    # Loop through all serialized model files in the directory
    for filename in os.listdir(serialized_models_dir):
        if filename.endswith(".pkl"):
            model_path = os.path.join(serialized_models_dir, filename)

            # Load the model and predict probabilities
            model = joblib.load(model_path)
            probas = model.predict_proba(prediction_df)[
                :, 1
            ]  # Probability of positive class

            # Extract classifier name from filename
            classifier_name = filename.split("_v")[0]

            # Store predictions along with classifier name
            results.append(
                pd.DataFrame(
                    {"Probability_At_Risk": probas, "Classifier": classifier_name}
                )
            )

    # Combine all model results into a single DataFrame
    return pd.concat(results, ignore_index=True)
