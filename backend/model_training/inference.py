import pandas as pd
import joblib
import os
from typing import List, Dict, Any


def prepare_and_predict(features: List[Dict[str, Any]], serialized_models_dir: str):
    prediction_df = pd.DataFrame(features)

    # Define the expected columns based on how the models were trained
    training_columns = [
        "diabetes", "medication_use", "sleep_hours_per_day", "family_history", "stress_level", 
        "country", "age", "smoking", "sedentary_hours_per_day", "continent", "sex", 
        "obesity", "income", "hemisphere", "cholesterol", "alcohol_consumption", "bmi", 
        "exercise_hours_per_week", "triglycerides", "heart_rate", "diet", 
        "physical_activity_days_per_week", "previous_heart_problems", 
        "systolic_blood_pressure", "diastolic_blood_pressure"
    ]

    # Ensure that the prediction data has the columns in the correct order
    prediction_df = prediction_df[training_columns]

    results = []
    import joblib
    for filename in os.listdir(serialized_models_dir):
        if filename.endswith('.pkl'):
            model_path = os.path.join(serialized_models_dir, filename)
            model = joblib.load(model_path)
            probas = model.predict_proba(prediction_df)[:, 1]  # Assuming the positive class is at index 1
            classifier_name = filename.split('_v')[0]
            results.append(pd.DataFrame({
                'Probability_At_Risk': probas,
                'Classifier': classifier_name
            }))
    return pd.concat(results, ignore_index=True) 