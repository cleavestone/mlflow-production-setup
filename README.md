# Remote MLflow Setup with AWS (EC2 + S3 + RDS)

## Project Overview

This project demonstrates how to set up a **production-ready MLflow tracking server** using AWS services:

- **EC2**: Hosting the MLflow server
- **RDS (PostgreSQL)**: Backend store for experiment metadata
- **S3**: Artifact storage for models, plots, and datasets
- **MLflow Python Client**: Logging experiments and models remotely

This setup enables:

- Centralized experiment tracking
- Versioned models
- Scalable artifact management

```
+--------------------+           +-------------------+           +----------------+
|  MLflow Client     |  ---->    |   EC2 MLflow      |  ---->    |   S3 Bucket     |
| (Training Script)  |           |   Server          |           |  (Artifacts)   |
|                    |           |                   |           |                |
+--------------------+           +-------------------+           +----------------+
                                     |
                                     v
                               +-------------+
                               |  RDS Postgres|
                               | (Metadata)   |
                               +-------------+
```
## Step 1: Launch EC2 Instance

1. Go to **AWS EC2 Console** → Launch a new instance.
2. Choose **Ubuntu 22.04 LTS** or **Amazon Linux 2**.
3. Assign a **security group** allowing inbound ports:
   - `5000` → MLflow server
   - `22` → SSH
4. Connect via SSH:

```bash
ssh -i "your-key.pem" ubuntu@<EC2_PUBLIC_IP>
```
## Step 2: Update Packages and Install Dependencies

1. **Update and upgrade system packages**

```bash
sudo apt update && sudo apt upgrade -y
```
*** Install Python 3 and pip
```bash
sudo apt install python3-pip -y
```
*** Install MLflow and PostgreSQL driver
```bash
pip3 install mlflow psycopg2-binary boto3
```

## Step 3: Setup S3 Bucket for Artifacts

1. Go to **AWS S3** → Create a bucket (e.g., `mlflow-artifacts-cleave`).

2. Ensure your **EC2 IAM role** or **AWS credentials** have access to the bucket.

3. Test S3 access:

```bash
aws s3 ls s3://mlflow-artifacts-cleave

## Step 4: Setup RDS PostgreSQL

1. Go to **AWS RDS Console** → Launch a **PostgreSQL** instance.

2. Configure the database:
   - **DB Name**: `mlflow_db`
   - **Username**: `mlflow_user`
   - **Password**: `mlflow_pass`
   - **Public accessibility**: Yes (or configure properly with VPC)

3. Note the **endpoint**, e.g.:

## Step 5: Start MLflow Server on EC2

Run the following command to start the MLflow server:

```bash
mlflow server \
  --backend-store-uri postgresql+psycopg2://mlflow_user:mlflow_pass@mlflow-db.cx26g4a04z9y.us-west-1.rds.amazonaws.com:5432/mlflow_db \
  --default-artifact-root s3://mlflow-artifacts-cleave \
  --host 0.0.0.0 \
  --port 5000

## Step 6: Use MLflow Python Client to Log Experiments Remotely

```python
import mlflow
import mlflow.sklearn

# Point MLflow client to the remote server
mlflow.set_tracking_uri("http://<EC2_PUBLIC_IP>:5000")

# Set or create an experiment
mlflow.set_experiment("iris-experiment")
```

### Notes

- Replace `<EC2_PUBLIC_IP>` with your EC2 instance's public IP.
- This allows your Python scripts to log **metrics**, **parameters**, and **models** directly to the **remote MLflow server**.

## Step 7: Log Model, Metrics, and Parameters with MLflow

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

# Load dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Start an MLflow run
with mlflow.start_run():
    # Train model
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    
    # Evaluate and log metric
    acc = clf.score(X_test, y_test)
    mlflow.log_metric("accuracy", acc)
    
    # Log and register model
    mlflow.sklearn.log_model(clf, "model", registered_model_name="IrisRF")
```

## Step 8: Load Registered Model in Pipeline

```python
import mlflow.pyfunc

# Load the registered model from MLflow Model Registry
model = mlflow.pyfunc.load_model("models:/IrisRF/Production")

# Make predictions
predictions = model.predict(X_test)
```
