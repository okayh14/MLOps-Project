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
from mlflow import MlflowClient
import zipfile

# Setup logging
logging.basicConfig()
logger = logging.getLogger(__name__)

app = FastAPI()

client = MlflowClient()


DATA_SERVICE_URL = "http://localhost:8001"
MODEL_SERVICE_URL = "http://localhost:8002"

# Define a global variable to hold loaded models
loaded_models = {}

def assess_heart_disease_risk(result, threshold=0.5):
    max_confidence = result['Probability_At_Risk'].max()
    risk_count = (result['Probability_At_Risk'] > threshold).sum()
    models_count = len(result)
    percentage_agreeing = (risk_count / models_count) * 100

    if risk_count == 0:
        return (f"This patient is not considered at risk for heart disease according to all {models_count} models, "
                f"with the highest risk prediction being {max_confidence:.2f}.")
    elif risk_count == models_count:
        return (f"This patient is considered at high risk for heart disease by all {models_count} models, "
                f"with a unanimous likelihood of {max_confidence:.2f}.")
    elif risk_count > models_count / 2:
        return (f"A majority of models ({risk_count} out of {models_count}, {percentage_agreeing:.2f}%) suggest this "
                f"patient may be at risk for heart disease, with the highest prediction at {max_confidence:.2f}. "
                f"Further clinical evaluation is recommended.")
    else:
        return (f"There is a divided opinion among the models regarding the risk for heart disease: {risk_count} out "
                f"of {models_count} ({percentage_agreeing:.2f}%) indicate a potential risk. Given the lack of a strong "
                f"consensus, further investigation and possibly additional diagnostic testing are advised. The highest "
                f"risk prediction is {max_confidence:.2f}.")

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

@app.post("/clean")
async def clean_data(data: dict):
    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.post(DATA_SERVICE_URL + "/clean", json=data)
            response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"An error occurred: {e}")

@app.post("/upload")
async def upload_data(data: dict):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(DATA_SERVICE_URL + "/patients", json=data)
            response.raise_for_status()
            train_data = await get_data()
            if train_data:
                training_response = await call_training(train_data)
                return {
                    "success": True,
                    "upload_status": response.status_code,
                    "training_status": training_response["status_code"]
                }
            else:
                raise HTTPException(status_code=500, detail="Failed to fetch data for training")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/call_training")
async def call_training(train_data: dict):
    try:
        async with httpx.AsyncClient(timeout=60000.0) as client:
            response = await client.post(MODEL_SERVICE_URL + "/train", json=train_data)
            response.raise_for_status()
            return {"success": True, "status_code": response.status_code}
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/call_inference")
async def call_inference(features: dict):
    try:
        async with httpx.AsyncClient(timeout=60000.0) as client:
            response = await client.post(MODEL_SERVICE_URL + "/inference", json=features)
            result_text = assess_heart_disease_risk(response)
        return { "result_text": result_text, "final_results": response}
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/start_training")
async def start_training():
    train_data = await get_data()
    await call_training(train_data)


@app.post("/start_inference")
async def start_inference(features: dict):
    clean_features = await clean_data(features)
    response = await call_inference(clean_features)
    return response

@app.on_event("startup")
async def startup_event():
    global loaded_models
    #if not loaded_models:
       # load_models()
     #   if not loaded_models:
        #    await load_registry()
      #      if not loaded_models:
         #       await start_training()