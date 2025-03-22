import pytest
import pandas as pd
import json
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from backend.model_training.inference import prepare_and_predict


def create_dummy_model_and_features(tmp_path):
    """
    Creates a trained model and saves it along with model_features.json.
    Returns test input features and the serialization path.
    """
    # Load data and train a dummy model
    iris = load_iris()
    X, y = iris.data, iris.target
    model = LogisticRegression(max_iter=1000)
    model.fit(X, (y == 1).astype(int))  # make it binary for predict_proba

    # Define column names and path
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)

    # Save model
    model_path = tmp_path / "DummyModel_v1.pkl"
    joblib.dump(model, model_path)

    # Save expected features
    with open(tmp_path / "model_features.json", "w") as f:
        json.dump({"columns": feature_names}, f)

    # Return a subset of rows as input data
    input_features = df.head(3).to_dict(orient="records")
    return input_features, str(tmp_path)


def test_prepare_and_predict_success(tmp_path):
    """
    Test successful inference on valid input and existing model files.
    """
    input_features, model_dir = create_dummy_model_and_features(tmp_path)
    result = prepare_and_predict(input_features, model_dir)

    assert isinstance(result, pd.DataFrame)
    assert "Probability_At_Risk" in result.columns
    assert "Classifier" in result.columns
    assert len(result) == 3  # 3 input rows = 3 predictions


def test_prepare_and_predict_missing_model_features(tmp_path):
    """
    Test if missing model_features.json raises the correct error.
    """
    input_data = [{"feature_0": 1, "feature_1": 2}]
    with pytest.raises(RuntimeError, match="Could not load model_features.json"):
        prepare_and_predict(input_data, str(tmp_path))


def test_prepare_and_predict_missing_columns(tmp_path):
    """
    Test if missing columns in input raises a ValueError.
    """
    input_features, model_dir = create_dummy_model_and_features(tmp_path)

    # Remove one required column from the input
    incomplete_input = [
        {k: v for k, v in sample.items() if k != "feature_0"}
        for sample in input_features
    ]

    with pytest.raises(ValueError, match="Missing columns in input"):
        prepare_and_predict(incomplete_input, model_dir)


def test_prepare_and_predict_no_models(tmp_path):
    """
    Test that the function raises an error if no .pkl models are present.
    """
    # Create valid model_features.json but no models
    with open(tmp_path / "model_features.json", "w") as f:
        json.dump({"columns": ["feature_0", "feature_1"]}, f)

    input_data = [{"feature_0": 1.0, "feature_1": 2.0}]

    # Expect ValueError because there are no models to concatenate
    with pytest.raises(ValueError, match="No objects to concatenate"):
        prepare_and_predict(input_data, str(tmp_path))
