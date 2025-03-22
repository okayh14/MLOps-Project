import os
import json
import pytest
import tempfile
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
import pandas as pd
import shutil

# Importiere die app, aber patche SERIALIZED_MODELS_DIR später
from backend.model_training.app import app

# TestClient für FastAPI
client = TestClient(app)

@pytest.fixture(scope="module")
def test_model_dir():
    """Erstellt ein temporäres Verzeichnis für die Tests und patcht den Pfad in der app."""
    # Temporäres Verzeichnis für Tests erstellen
    test_dir = "./test_serialized_models"
    os.makedirs(test_dir, exist_ok=True)
    
    # Patch the constant in the app module
    original_dir = None
    
    # Speichere den originalen Wert
    import backend.model_training.app as app_module
    original_dir = app_module.SERIALIZED_MODELS_DIR
    
    # Setze temporären Wert
    app_module.SERIALIZED_MODELS_DIR = test_dir
    
    # Übergib das Verzeichnis an die Tests
    yield test_dir
    
    # Stelle ursprünglichen Wert wieder her
    app_module.SERIALIZED_MODELS_DIR = original_dir
    
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir, ignore_errors=True)

@pytest.fixture
def sample_training_data():
    return [
        {"feature1": 1.0, "feature2": 2.0, "label": 0},
        {"feature1": 3.0, "feature2": 4.0, "label": 1},
        {"feature1": 5.0, "feature2": 6.0, "label": 0},
    ]

@pytest.fixture
def sample_inference_data():
    return [
        {"feature1": 1.0, "feature2": 2.0},
        {"feature1": 3.0, "feature2": 4.0},
    ]

@pytest.fixture
def sample_inference_results():
    return pd.DataFrame({
        "Id": [1, 2],
        "Probability_At_Risk": [0.2, 0.8],
        "Prediction": [0, 1]
    })

# Test für den /check Endpunkt
def test_check_registry_no_models(test_model_dir):
    """Test für /check wenn keine Modelle verfügbar sind."""
    with patch("os.walk") as mock_walk:
        mock_walk.return_value = [(test_model_dir, [], [])]
        response = client.get("/check")
        assert response.status_code == 200
        assert response.json() == {"message": "No models available."}

def test_check_registry_with_models(test_model_dir):
    """Test für /check wenn Modelle verfügbar sind."""
    with patch("os.walk") as mock_walk:
        mock_walk.return_value = [(test_model_dir, [], ["model1.pkl", "model2.pkl"])]
        response = client.get("/check")
        assert response.status_code == 200
        assert response.json() == {"message": "Models available."}

def test_check_registry_exception(test_model_dir):
    """Test für /check wenn eine Exception auftritt."""
    with patch("os.path.exists", side_effect=Exception("Test exception")):
        response = client.get("/check")
        assert response.status_code == 500
        assert "Test exception" in response.json()["detail"]

# Tests für den /train Endpunkt
@pytest.mark.asyncio
async def test_train_success(test_model_dir, sample_training_data):
    """Test für erfolgreiche Modelltraining."""
    # Mocks für die asynchronen Funktionen
    with patch("backend.model_training.app.clean_model_registry_and_folder", new_callable=AsyncMock) as mock_clean, \
         patch("backend.model_training.app.main", new_callable=AsyncMock) as mock_main, \
         patch("backend.model_training.app.register_top_models", new_callable=AsyncMock) as mock_register, \
         patch("backend.model_training.app.serialize_and_compress_models", new_callable=AsyncMock) as mock_serialize:
        
        # Mock für Trainingsergebnisse
        mock_main.return_value = {
            "results_df": pd.DataFrame({"model": ["model1"], "metric": [0.9]}),
            "experiment_name": "test_experiment"
        }
        
        response = client.post("/train", json=sample_training_data)
        
        # Überprüfung, ob alle Funktionen aufgerufen wurden
        mock_clean.assert_called_once()
        mock_main.assert_called_once_with(sample_training_data)
        mock_register.assert_called_once()
        mock_serialize.assert_called_once_with(test_model_dir)
        
        assert response.status_code == 200
        assert response.json() == {"message": "Training started successfully."}

@pytest.mark.asyncio
async def test_train_exception(test_model_dir):
    """Test für Fehler beim Modelltraining."""
    with patch("backend.model_training.app.clean_model_registry_and_folder", new_callable=AsyncMock) as mock_clean:
        mock_clean.side_effect = Exception("Training error")
        
        response = client.post("/train", json=[])
        
        assert response.status_code == 500
        assert "Training error" in response.json()["detail"]

# Tests für den /inference Endpunkt
@pytest.mark.asyncio
async def test_inference_success(test_model_dir, sample_inference_data, sample_inference_results):
    """Test für erfolgreiche Inferenz."""
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
    """Test für Inferenz wenn keine Modelle vorhanden sind."""
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
    """Test für Fehler bei der Inferenz."""
    with patch("os.listdir") as mock_listdir:
        mock_listdir.side_effect = Exception("Inference error")
        
        response = client.post("/inference", json=sample_inference_data)
        
        assert response.status_code == 500
        assert "Inference error" in response.json()["detail"]
