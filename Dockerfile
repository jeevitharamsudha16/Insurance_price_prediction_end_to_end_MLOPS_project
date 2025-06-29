FROM python:3.10-slim

WORKDIR /app
COPY . /app

# Install system dependencies for MLflow (important!)
RUN apt-get update && apt-get install -y build-essential && \
    pip install --upgrade pip && \
    pip install mlflow gunicorn && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Ensure volume/dir for backend store and artifact root exist
RUN mkdir -p /app/mlruns

# Main command: bind to Render's dynamic port
CMD mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port $PORT
