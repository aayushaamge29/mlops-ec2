from fastapi import FastAPI
import joblib
import logging
import time

logging.basicConfig(level=logging.INFO)

app = FastAPI()
model = joblib.load("model.pkl")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: dict):
    start = time.time()

    features = [
        data["sepal_length"],
        data["sepal_width"],
        data["petal_length"],
        data["petal_width"]
    ]

    pred = model.predict([features])

    latency = time.time() - start
    logging.info(f"Prediction={pred[0]}, latency={latency:.3f}s")

    return {"prediction": int(pred[0])}
