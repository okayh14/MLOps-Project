from fastapi import FastAPI
import httpx
import asyncio
import logging

# Setup logging
logging.basicConfig()
logger = logging.getLogger(__name__)

app = FastAPI()

DATA_CLEAN_URL = "http://localhost:8001/clean"
MODEL_TRAIN_URL = "http://localhost:8002/train"

clean_data = None

@app.post("/clean_data")
async def cleaning(raw_data):
    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            clean_data = await client.post(DATA_CLEAN_URL, data=raw_data)
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
    return None


@app.post("/start_training")
async def training(raw_data):
    try:
        #data = cleaning(raw_data)
        async with httpx.AsyncClient(timeout=600.0) as client:
            await client.post(MODEL_TRAIN_URL, data=clean_data)
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
    return None

def load_model():
    #temp anlegen
    #mlflow as run starten
    #model registry aufrufen
    #models in temp ablegen

@app.post("/start_inference")
async def inference(features: dict)
    try:

