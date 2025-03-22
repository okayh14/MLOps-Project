import pytest
import os
import pandas as pd
import mlflow
import mlflow.sklearn
import shutil
from mlflow.tracking import MlflowClient
from unittest.mock import patch, MagicMock
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from backend.model_training.model_registry import (
    register_top_models,
    serialize_and_compress_models,
    clean_model_registry_and_folder,
)

# Mark all tests in this file as async-compatible
pytestmark = pytest.mark.asyncio


async def test_register_top_models_empty_df():
    """
    Ensures that no model is registered when the input DataFrame is empty.
    """
    with patch("mlflow.register_model") as mock_register:
        await register_top_models(pd.DataFrame(), "test_exp")
        mock_register.assert_not_called()


async def test_register_top_models_success():
    """
    Tests successful registration of a valid model using a mock results DataFrame.
    """
    df = pd.DataFrame(
        {
            "run_id": ["abc123"],
            "model_name": ["LogisticRegression"],
            "encoder": ["Label"],
            "scaler": ["None"],
            "feat_selection": ["None"],
            "fbeta_1_5": [0.9],
            "C": [0.1],
            "max_iter": [100],
            "solver": ["lbfgs"],
        }
    )

    with patch("mlflow.register_model") as mock_register, patch(
        "mlflow.tracking.MlflowClient.transition_model_version_stage"
    ), patch("mlflow.tracking.MlflowClient.update_model_version"):
        await register_top_models(df, "test_exp")
        mock_register.assert_called_once()


async def test_serialize_and_compress_models_real(tmp_path):
    """
    Trains and registers a real model using sklearn and MLflow,
    then verifies that the model is serialized to disk as a .pkl file.
    """
    iris = load_iris()
    X, y = iris.data, iris.target
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(model, artifact_path="model")
        run_id = run.info.run_id

    model_name = "DummyTestModel"
    try:
        mlflow.register_model(
            model_uri=f"runs:/{run_id}/model", name=model_name,
        )
    except mlflow.exceptions.MlflowException:
        pass  # Ignore if it already exists

    # Create target directory and serialize models
    target_dir = tmp_path / "serialized"
    os.makedirs(target_dir, exist_ok=True)
    await serialize_and_compress_models(str(target_dir))

    # Check if any .pkl files were created
    files = list(target_dir.glob("*.pkl"))
    assert len(files) > 0


async def test_clean_model_registry_and_folder(tmp_path):
    """
    Tests that both registered models and serialized local files are deleted correctly.
    """
    # Create a dummy file to simulate existing serialized model
    file_path = tmp_path / "dummy.pkl"
    file_path.write_text("delete me")

    # Mock a dummy model version in staging
    dummy_version = MagicMock()
    dummy_version.version = "1"
    dummy_version.current_stage = "Staging"

    dummy_model = MagicMock()
    dummy_model.name = "TestModel"
    dummy_model.latest_versions = [dummy_version]

    # Patch all client interactions
    with patch(
        "mlflow.tracking.MlflowClient.search_registered_models",
        return_value=[dummy_model],
    ), patch(
        "mlflow.tracking.MlflowClient.get_latest_versions", return_value=[dummy_version]
    ), patch(
        "mlflow.tracking.MlflowClient.transition_model_version_stage"
    ), patch(
        "mlflow.tracking.MlflowClient.delete_model_version"
    ), patch(
        "mlflow.tracking.MlflowClient.delete_registered_model"
    ):
        models_deleted, files_deleted = await clean_model_registry_and_folder(tmp_path)

        # Verify both model and file were deleted
        assert models_deleted == 1
        assert files_deleted >= 1
