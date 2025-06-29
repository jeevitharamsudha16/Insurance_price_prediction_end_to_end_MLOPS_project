FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y build-essential && \
    pip install --upgrade pip && \
    pip install mlflow && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/mlruns

# 👇 Use ENTRYPOINT instead of CMD to force override
ENTRYPOINT ["mlflow", "server"]
CMD [
  "--backend-store-uri", "sqlite:///mlflow.db",
  "--default-artifact-root", "./mlruns",
  "--host", "0.0.0.0",
  "--port", "5000"
]
