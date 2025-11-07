# AutoGluon + MLflow + DVC ortamı
FROM python:3.11-slim

WORKDIR /workspace

# Gereken sistem bağımlılıkları
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# Python bağımlılıkları
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir mlflow==2.14.0 autogluon==1.2.0 dvc[gdrive]==3.53.0

COPY . .

CMD ["python", "app/train.py"]
