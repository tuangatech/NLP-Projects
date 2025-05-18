# ingestion/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install ingestion-specific dependencies
# build context is set to ./ingestion (directory) in Docker-compose.yml
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ingestion code
COPY ingest.py .

# Entrypoint for ingestion
ENTRYPOINT ["python", "ingest.py"]