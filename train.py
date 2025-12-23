import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

# 👇 IMPORTANT: local mlflow storage (CI-safe)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("iris-mlops")

X, y = load_iris(return_X_y=True)

with mlflow.start_run():
    model = RandomForestClassifier()
    model.fit(X, y)

    acc = model.score(X, y)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

joblib.dump(model, "app/model.pkl")
