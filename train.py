import os
import mlflow
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

# CI-safe local tracking
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("iris-mlops")

X, y = load_iris(return_X_y=True)

with mlflow.start_run():
    model = RandomForestClassifier()
    model.fit(X, y)

    acc = model.score(X, y)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_param("model_type", "RandomForest")

# ✅ Explicit artifact save (Docker-friendly)
os.makedirs("app", exist_ok=True)
joblib.dump(model, "app/model.pkl")

print("Training completed, model saved to app/model.pkl")
