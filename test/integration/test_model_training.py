import pytest
import json
import os
from unittest.mock import patch
from backend.model_training.model_training import main

@pytest.mark.asyncio
async def test_main_integration(tmp_path):
    """
    Integration test that runs the entire training process (train_and_evaluate) 
    and model registration by calling the `main()` function.
    
    This test ensures that the pipeline is functioning end-to-end, even with mocked 
    dependencies, by validating that the result returned by the `main()` function 
    contains the expected status and results.
    
    Expected Behavior:
    - The function should successfully execute the training pipeline.
    - The `main()` function should return a status of 'success'.
    - The returned result should include an experiment name and a results dataframe.
    - The dataframe should contain some records (i.e., not be empty).
    """

    # Example input data for the test
    input_data = {
        "age": [45, 50, 60, 55, 52, 48, 62, 58, 53, 51],
        "sex": ["Male", "Female"] * 5,
        "heart_attack_risk": [0, 1] * 5,
    }
    
    # Save the input data as a JSON file
    input_json_path = tmp_path / "input_data.json"
    with open(input_json_path, "w") as f:
        json.dump(input_data, f)

    # Mocking `mlflow.start_run` and other functions to avoid actual experiments
    with patch("mlflow.start_run") as mock_start_run, \
         patch("mlflow.log_param"), patch("mlflow.log_metrics"), patch("mlflow.sklearn.log_model"), \
         patch("mlflow.register_model"), patch("mlflow.tracking.MlflowClient"):

        # Call the main function (which covers all pipeline steps)
        result = await main(input_data)

    # Verify the results
    assert result["status"] == "success"
    assert "experiment_name" in result
    assert "results_df" in result  # Checken, ob ein DataFrame zurückgegeben wurde
    assert len(result["results_df"]) > 0  # Sicherstellen, dass es Ergebnisse gibt
