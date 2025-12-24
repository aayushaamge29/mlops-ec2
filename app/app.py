import logging
import time
import uuid
import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

APP_VERSION = "v1.1.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

app = FastAPI(title="Iris ML API", version=APP_VERSION)

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
model = joblib.load(MODEL_PATH)

class PredictRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": APP_VERSION
    }

@app.post("/predict")
def predict(data: PredictRequest):
    request_id = str(uuid.uuid4())
    start = time.time()

    try:
        features = [[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]]

        prediction = model.predict(features)[0]
        latency = time.time() - start

        logging.info(
            f"request_id={request_id} prediction={prediction} latency={latency:.3f}s"
        )

        return {
            "request_id": request_id,
            "prediction": int(prediction),
            "version": APP_VERSION
        }

    except Exception as e:
        logging.error(f"request_id={request_id} error={str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")
