import mlflow
import mlflow.sklearn
import numpy as np

# Tracking URI
mlflow.set_tracking_uri("http://52.53.238.162:5000")

# Load model from registry
model_name = "LinearRegressionModel"
model_version = 1  # choose the version you registered
model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")

# Example prediction
X_new = np.array([[0.5], [1.5]])
predictions = model.predict(X_new)
print("Predictions:", predictions)
