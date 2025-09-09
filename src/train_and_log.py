import mlflow
import mlflow.sklearn
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Set MLflow tracking URI to your remote server
mlflow.set_tracking_uri("http://52.53.238.162:5000")

# Create or set experiment
mlflow.set_experiment("simple_linear_regression")

# Generate sample data
X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

with mlflow.start_run() as run:
    # Parameters
    fit_intercept = True
    mlflow.log_param("fit_intercept", fit_intercept)

    # Train model
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mlflow.log_metric("mse", mse)

    # Log model
    mlflow.sklearn.log_model(model, artifact_path="linear_model", registered_model_name="LinearRegressionModel")

    print(f"Run ID: {run.info.run_id}")



