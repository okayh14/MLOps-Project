from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
import os
from mlflow.tracking import MlflowClient
from backend.model_training.model_registry import (
    register_top_models,
    serialize_and_compress_models,
    clean_model_registry_and_folder,
)
from backend.model_training.model_training import main
from backend.model_training.inference import prepare_and_predict

# Initialize FastAPI app
app = FastAPI()

# Initialize MLflow client
client = MlflowClient()

# Path where models will be serialized and stored
SERIALIZED_MODELS_DIR = "./serialized_models"

# URL for the data service (can be overridden with environment variable)
DATA_SERVICE_URL = os.getenv("DATA_SERVICE_URL", "http://data_service:8001")


@app.post("/train")
async def start_training(training_data: List[Dict[str, Any]]):
    """
    Starts the model training pipeline:
    - Cleans existing models
    - Trains models on provided data
    - Registers top-performing models
    - Serializes and compresses models for inference
    """
    try:
        await clean_model_registry_and_folder(SERIALIZED_MODELS_DIR)
        train_results = await main(training_data)
        await register_top_models(
            train_results["results_df"], train_results["experiment_name"]
        )
        await serialize_and_compress_models(SERIALIZED_MODELS_DIR)

        return {"message": "Training started successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inference")
async def inference(features: List[Dict[str, Any]]):
    """
    Runs inference on the provided patient features using the latest models.
    - If no serialized models are found, tries to regenerate them.
    - Returns prediction results and mean probability of risk.
    """
    try:
        print("\n[DEBUG] Received features:")
        print(features)

        if not os.listdir(SERIALIZED_MODELS_DIR):
            print("[DEBUG] No serialized models found. Re-serializing...")
            await serialize_and_compress_models()

        final_results = prepare_and_predict(features, SERIALIZED_MODELS_DIR)

        if final_results.empty:
            print("[DEBUG] No results from prepare_and_predict.")
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"message": "No models available for inference."},
            )

        mean_proba = final_results["Probability_At_Risk"].mean()
        print("[DEBUG] Inference complete. Mean proba:", mean_proba)

        return {
            "final_results": final_results.to_dict(orient="records"),
            "mean_proba": mean_proba,
        }

    except Exception as e:
        import traceback

        print("\n[ERROR] Inference crashed with exception:")
        traceback.print_exc()  # This prints the full stack trace for debugging
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/check")
async def check_registry():
    """
    Health check endpoint to verify availability of serialized models.
    - Creates the model directory and serializes models if necessary.
    - Returns the number of model files found.
    """
    try:
        # Ensure the model directory exists and serialize models if missing
        if not os.path.exists(SERIALIZED_MODELS_DIR):
            os.makedirs(SERIALIZED_MODELS_DIR, exist_ok=True)
            await serialize_and_compress_models(SERIALIZED_MODELS_DIR)

        # Count model files in the directory
        file_count = sum(len(files) for _, _, files in os.walk(SERIALIZED_MODELS_DIR))
        if file_count < 1:
            return {"message": "No models available."}
        else:
            return {"message": "Models available."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
