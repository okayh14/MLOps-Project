from fastapi import FastAPI, HTTPException
import httpx
import asyncio
import logging
import os
import pickle
import dill
import pandas as pd
import json
from pydantic import BaseModel
import mlflow
import zipfile
from typing import List, Dict, Any
import time
import traceback

# Setup logging
logging.basicConfig()
logger = logging.getLogger(__name__)

app = FastAPI()

DATA_SERVICE_URL = os.getenv("DATA_SERVICE_URL", "http://data_service:8001")
MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://model_training:8002")

loaded_models = {}

async def assess_heart_disease_risk(result, mean_proba, threshold=0.5):
    result_df = pd.DataFrame(result)
    max_confidence = result_df["Probability_At_Risk"].max()
    risk_count = (result_df["Probability_At_Risk"] > threshold).sum()
    models_count = len(result_df)
    percentage_agreeing = (risk_count / models_count) * 100

    if risk_count == 0:
        return (
            f"This patient is not considered at risk for heart disease according to all {models_count} models, "
            f"with the highest risk prediction being {max_confidence:.2f} and the mean probability being {mean_proba:.2f}."
        )
    elif risk_count == models_count:
        return (
            f"This patient is considered at high risk for heart disease by all {models_count} models, "
            f"with a unanimous likelihood of {max_confidence:.2f} and the mean probability being {mean_proba:.2f}."
        )
    elif risk_count > models_count / 2:
        return (
            f"A majority of models ({risk_count} out of {models_count}, {percentage_agreeing:.2f}%) suggest this "
            f"patient may be at risk for heart disease, with the highest prediction at {max_confidence:.2f} and the mean probability being {mean_proba:.2f}. "
            f"Further clinical evaluation is recommended."
        )
    else:
        return (
            f"There is a divided opinion among the models regarding the risk for heart disease: {risk_count} out "
            f"of {models_count} ({percentage_agreeing:.2f}%) indicate a potential risk. Given the lack of a strong "
            f"consensus, further investigation and possibly additional diagnostic testing are advised. The highest "
            f"risk prediction is {max_confidence:.2f} and the mean probability being {mean_proba:.2f}."
        )

@app.get("/data")
async def get_data():
    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.get(DATA_SERVICE_URL + "/data")
            response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"An error occurred: {e}")

    return None

@app.post("/upload")
async def upload_data(data: dict):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(DATA_SERVICE_URL + "/patients", json=data)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/trigger_upload")
async def upload_retrain(data: dict):
    try:
        response_upload = await upload_data(data)
        response_training = await trigger_training()
        return {
            'response_upload':response_upload.get('status'),
            'response_training':response_training.get('message')
        }
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    
@app.post("/call_training")
async def orchestrator_train_models(train_data: dict): 
    try:
        async with httpx.AsyncClient(timeout=60000.0) as client:
            resp = await client.post(f"{MODEL_SERVICE_URL}/train", json=train_data)
            resp.raise_for_status()
            return {"success": True, "status_code": resp.status_code}
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/call_inference")
async def call_prediction(features: List[Dict[str, Any]]):
    try:
        async with httpx.AsyncClient(timeout=60000.0) as client:
            response = await client.post(
                MODEL_SERVICE_URL + "/inference", json=features
            )
            response.raise_for_status()  
            response_data = response.json()

        logger.info(f"Inference Response: {response_data}")

        if "final_results" not in response_data or "mean_proba" not in response_data:
            raise HTTPException(
                status_code=400,
                detail=f"Inference response malformed: {response_data}"
            )

        result_text = await assess_heart_disease_risk(
            response_data["final_results"], response_data["mean_proba"]
        )

        return {
            "result_text": result_text,
            "final_results": response_data["final_results"],
            "mean_proba": response_data["mean_proba"],
        }

    except Exception as e:
        logger.error(f"An error occurred in call_prediction: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


async def _trigger_training_logic():
    """Internal function with the training trigger logic"""
    data = await get_data()
    if not data:
        raise HTTPException(status_code=400, detail="No data for training.")
    resp = await orchestrator_train_models(data)
    return {"message": "Training successful.", "info": resp}


@app.post("/start_training") 
async def trigger_training():
    """API endpoint for training"""
    try:
        result = await _trigger_training_logic()
        return result
    except Exception as e:
        logger.error(f"Error in trigger_training: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/start_inference")
async def start_prediction(features: List[Dict[str, Any]]):
    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.post(
                DATA_SERVICE_URL + "/clean",
                json=features,
                params={"drop_target": True}  
            )
            response.raise_for_status()
            clean_features = response.json()

        prediction_response = await call_prediction(clean_features)
        return prediction_response
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """
    Prüft /check bei model_training. 
    Falls "No models available.", ruft /start_training auf.
    """
    logger.info("Startup: Warte auf model_training...")
    start_time = time.time()
    timeout = 10  
    retry_interval = 2

    while True:
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                resp = await client.get(f"{MODEL_SERVICE_URL}/check")
                
                try:
                    data = resp.json()
                    if data.get("message") == "No models available.":
                        logger.info("Keine Modelle gefunden. Trigger Training...")
                        await _trigger_training_logic()
                    else:
                        logger.info("Modelle vorhanden. Kein Training nötig.")
                    break
                except json.JSONDecodeError:
                    logger.error(f"JSON decode error in startup_event. Raw text: {await resp.aread()}")
                    break

        except Exception as e:
            if time.time() - start_time > timeout:
                logger.error("Timeout: model_training nicht erreichbar.")
                logger.error(traceback.format_exc())
                break
            logger.warning("model_training noch nicht erreichbar... retrying...")
            await asyncio.sleep(retry_interval)

