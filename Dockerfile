FROM python:3.10

WORKDIR /app
COPY . /app

RUN pip install mlflow gunicorn

CMD mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port $PORT
