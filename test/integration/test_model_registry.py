# tests/integration/test_model_registry_flow.py

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
    iris = load_iris()
    X, y = iris.data, iris.target
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model


async def test_full_model_registry_flow(tmp_path, iris_model):
    # === Step 1: Log Model ===
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(iris_model, artifact_path="model")
        run_id = run.info.run_id

    # === Step 2: Create results_df ===
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

    # === Step 3: Register model ===
    await register_top_models(df, experiment_name="integration_test_exp", top_n=1)

    # === Step 4: Serialize to disk ===
    serialized_dir = tmp_path / "models"
    os.makedirs(serialized_dir, exist_ok=True)

    await serialize_and_compress_models(str(serialized_dir))

    files = list(serialized_dir.glob("*.pkl"))
    assert len(files) > 0, "No serialized model files found"

    # === Step 5: Clean registry + serialized files ===
    models_deleted, files_deleted = await clean_model_registry_and_folder(serialized_dir)

    assert models_deleted >= 1, "Expected at least one model to be deleted from registry"
    assert files_deleted >= 1, "Expected serialized files to be deleted"
