import os
import json
import pytest
import tempfile
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
import pandas as pd
import shutil
# Import the app, patch SERIALIZED_MODELS_DIR later
from backend.model_training.app import app

# FastAPI test client
client = TestClient(app)

@pytest.fixture(scope="module")
def test_model_dir():
    """
    Creates a temporary directory for testing and patches the app's model directory.
    """
    with tempfile.TemporaryDirectory() as test_dir:
        # Patch the constant in the app module
        import backend.model_training.app as app_module
        original_dir = app_module.SERIALIZED_MODELS_DIR
        
        # Set temporary directory for model serialization
        app_module.SERIALIZED_MODELS_DIR = test_dir
        
        # Übergib das Verzeichnis an die Tests
        yield test_dir
        
        # Restore original directory path
        app_module.SERIALIZED_MODELS_DIR = original_dir
        # Cleanup is handled automatically

@pytest.fixture
def sample_training_data():
    """
    Provides example input data for training endpoint tests.
    """
    return [
        {"feature1": 1.0, "feature2": 2.0, "label": 0},
        {"feature1": 3.0, "feature2": 4.0, "label": 1},
        {"feature1": 5.0, "feature2": 6.0, "label": 0},
    ]

@pytest.fixture
def sample_inference_data():
    """
    Provides example input data for inference endpoint tests.
    """
    return [
        {"feature1": 1.0, "feature2": 2.0},
        {"feature1": 3.0, "feature2": 4.0},
    ]

@pytest.fixture
def sample_inference_results():
    """
    Returns a sample prediction result DataFrame.
    """
    return pd.DataFrame({
        "Id": [1, 2],
        "Probability_At_Risk": [0.2, 0.8],
        "Prediction": [0, 1]
    })

def test_check_registry_no_models(test_model_dir):
    """
    Tests /check endpoint when no models are available.
    """
    with patch("os.walk") as mock_walk:
        mock_walk.return_value = [(test_model_dir, [], [])]
        response = client.get("/check")
        assert response.status_code == 200
        assert response.json() == {"message": "No models available."}

def test_check_registry_with_models(test_model_dir):
    """
    Tests /check endpoint when models are available.
    """
    with patch("os.walk") as mock_walk:
        mock_walk.return_value = [(test_model_dir, [], ["model1.pkl", "model2.pkl"])]
        response = client.get("/check")
        assert response.status_code == 200
        assert response.json() == {"message": "Models available."}

def test_check_registry_exception(test_model_dir):
    """
    Tests /check endpoint when an exception is raised.
    """
    with patch("os.path.exists", side_effect=Exception("Test exception")):
        response = client.get("/check")
        assert response.status_code == 500
        assert "Test exception" in response.json()["detail"]

@pytest.mark.asyncio
async def test_train_success(test_model_dir, sample_training_data):
    """
    Tests successful training via /train endpoint.
    """
    with patch("backend.model_training.app.clean_model_registry_and_folder", new_callable=AsyncMock) as mock_clean, \
         patch("backend.model_training.app.main", new_callable=AsyncMock) as mock_main, \
         patch("backend.model_training.app.register_top_models", new_callable=AsyncMock) as mock_register, \
         patch("backend.model_training.app.serialize_and_compress_models", new_callable=AsyncMock) as mock_serialize:
        
        mock_main.return_value = {
            "results_df": pd.DataFrame({"model": ["model1"], "metric": [0.9]}),
            "experiment_name": "test_experiment"
        }
        
        response = client.post("/train", json=sample_training_data)
        
        mock_clean.assert_called_once()
        mock_main.assert_called_once_with(sample_training_data)
        mock_register.assert_called_once()
        mock_serialize.assert_called_once_with(test_model_dir)
        
        assert response.status_code == 200
        assert response.json() == {"message": "Training started successfully."}

@pytest.mark.asyncio
async def test_train_exception(test_model_dir):
    """
    Tests training failure due to an exception in /train endpoint.
    """
    with patch("backend.model_training.app.clean_model_registry_and_folder", new_callable=AsyncMock) as mock_clean:
        mock_clean.side_effect = Exception("Training error")
        
        response = client.post("/train", json=[])
        
        assert response.status_code == 500
        assert "Training error" in response.json()["detail"]

# Tests für den /inference Endpunkt
@pytest.mark.asyncio
async def test_inference_success(test_model_dir, sample_inference_data, sample_inference_results):
    """
    Tests successful inference via /inference endpoint.
    """
    with patch("os.listdir") as mock_listdir, \
         patch("backend.model_training.app.prepare_and_predict") as mock_predict:
        
        mock_listdir.return_value = ["model1.pkl", "model2.pkl"]
        mock_predict.return_value = sample_inference_results
        
        response = client.post("/inference", json=sample_inference_data)
        
        mock_predict.assert_called_once()
        
        assert response.status_code == 200
        result = response.json()
        assert "final_results" in result
        assert "mean_proba" in result
        assert len(result["final_results"]) == 2
        assert result["mean_proba"] == 0.5  # (0.2 + 0.8) / 2

@pytest.mark.asyncio
async def test_inference_no_models(test_model_dir, sample_inference_data):
    """
    Tests /inference when no models are available.
    """
    with patch("os.listdir") as mock_listdir, \
         patch("backend.model_training.app.serialize_and_compress_models", new_callable=AsyncMock) as mock_serialize, \
         patch("backend.model_training.app.prepare_and_predict") as mock_predict:
        
        mock_listdir.return_value = []
        mock_predict.return_value = pd.DataFrame()
        
        response = client.post("/inference", json=sample_inference_data)
        
        mock_serialize.assert_called_once()
        assert response.status_code == 404
        assert response.json() == {"message": "No models available for inference."}

@pytest.mark.asyncio
async def test_inference_exception(test_model_dir, sample_inference_data):
    """
    Tests /inference when an exception occurs during processing.
    """
    with patch("os.listdir") as mock_listdir:
        mock_listdir.side_effect = Exception("Inference error")
        
        response = client.post("/inference", json=sample_inference_data)
        
        assert response.status_code == 500
        assert "Inference error" in response.json()["detail"]
