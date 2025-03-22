import os
import pandas as pd
import pytest
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from backend.model_training.model_registry import (
    register_top_models,
    serialize_and_compress_models,
    clean_model_registry_and_folder,
)

pytestmark = pytest.mark.asyncio

@pytest.fixture
def iris_model():
    """
    Creates and returns a trained Logistic Regression model using the Iris dataset.
    """
    iris = load_iris()
    X, y = iris.data, iris.target
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model


async def test_full_model_registry_flow(tmp_path, iris_model):
    """
    Full integration test for the model registry pipeline.

    This test:
    - Logs a trained model to MLflow
    - Constructs a results DataFrame for model selection
    - Registers the top model using custom logic
    - Serializes and compresses the registered model to disk
    - Verifies that serialized files exist
    - Cleans up both the MLflow registry and local files
    - Asserts that at least one model and one file were deleted
    """
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(iris_model, artifact_path="model")
        run_id = run.info.run_id

    df = pd.DataFrame({
        "run_id": [run_id],
        "model_name": ["LogisticRegression"],
        "encoder": ["None"],
        "scaler": ["None"],
        "feat_selection": ["None"],
        "fbeta_1_5": [0.93],
        "C": [1.0],
        "max_iter": [1000],
        "solver": ["lbfgs"],
    })

    await register_top_models(df, experiment_name="integration_test_exp", top_n=1)

    serialized_dir = tmp_path / "models"
    os.makedirs(serialized_dir, exist_ok=True)

    await serialize_and_compress_models(str(serialized_dir))

    # Ensure that at least one serialized file exists
    files = list(serialized_dir.glob("*.pkl"))
    assert len(files) > 0, "No serialized model files found"

    # Clean up registry and local serialized files
    models_deleted, files_deleted = await clean_model_registry_and_folder(serialized_dir)

    assert models_deleted >= 1, "Expected at least one model to be deleted from registry"
    assert files_deleted >= 1, "Expected serialized files to be deleted"
