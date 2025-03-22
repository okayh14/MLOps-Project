import pytest
import pandas as pd
import joblib
import json
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from backend.model_training.inference import prepare_and_predict

@pytest.mark.asyncio
async def test_inference_end_to_end(tmp_path):
    """
    End-to-end test for the inference pipeline.
    
    This test:
    - Trains a simple binary logistic regression model on the Iris dataset
    - Saves the model and its feature metadata to a temporary directory
    - Simulates a real prediction input with 5 samples
    - Runs inference using the prepare_and_predict() function
    - Verifies that the returned result is a valid DataFrame with expected structure
    """
    # Load example data and make the target binary (e.g., class 1 vs. the rest)
    iris = load_iris()
    X, y = iris.data, iris.target
    binary_y = (y == 1).astype(int)

    # Create a DataFrame with feature names
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)

    # Train a logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(df, binary_y)

    # Save the trained model
    model_path = tmp_path / "MyModel_v1.pkl"
    joblib.dump(model, model_path)

    # Save the feature metadata
    with open(tmp_path / "model_features.json", "w") as f:
        json.dump({"columns": feature_names}, f)

    # Prepare input data (simulate real prediction request)
    input_features = df.head(5).to_dict(orient="records")  # simulate 5 patients

    # Run prediction
    result = prepare_and_predict(input_features, str(tmp_path))

    # Validate the result
    assert isinstance(result, pd.DataFrame)
    assert "Probability_At_Risk" in result.columns
    assert "Classifier" in result.columns
    assert len(result) == 5
    assert result["Probability_At_Risk"].between(0, 1).all()
