# app/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install app-specific dependencies, with build context is set to ./app in Docker-compose.yml
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app structure
COPY backend/ backend/
COPY frontend/ frontend/

# Copy entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose ports
EXPOSE 8000 8501

# Entrypoint for app. Replace ./entrypoint.sh with /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]   