import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import joblib

# IMPORTANT: local MLflow tracking
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Set / create experiment
mlflow.set_experiment("iris-mlops")

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Log params & metrics
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc)

    # Log model
    mlflow.sklearn.log_model(model, "model")

    # Save model for API
    os.makedirs("app", exist_ok=True)
    joblib.dump(model, "app/model.pkl")

    print(f"✅ Model trained | Accuracy={acc}")