import pandas as pd
import joblib
import os
import json
from typing import List, Dict, Any

def prepare_and_predict(features: List[Dict[str, Any]], serialized_models_dir: str):
    prediction_df = pd.DataFrame(features)

    # Dynamisch die erwarteten Trainings-Spalten laden
    try:
        with open(os.path.join(serialized_models_dir, "model_features.json"), "r") as f:
            training_columns = json.load(f)["columns"]
        print("[INFO] Loaded training columns from model_features.json")
    except Exception as e:
        raise RuntimeError(f"Could not load model_features.json: {e}")

    # Checke, ob alle erwarteten Spalten vorhanden sind
    missing_cols = set(training_columns) - set(prediction_df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in input: {missing_cols}")

    prediction_df = prediction_df[training_columns]

    results = []

    for filename in os.listdir(serialized_models_dir):
        if filename.endswith(".pkl"):
            model_path = os.path.join(serialized_models_dir, filename)
            model = joblib.load(model_path)
            probas = model.predict_proba(prediction_df)[:, 1]  # positive class
            classifier_name = filename.split("_v")[0]
            results.append(
                pd.DataFrame(
                    {"Probability_At_Risk": probas, "Classifier": classifier_name}
                )
            )

    return pd.concat(results, ignore_index=True)
