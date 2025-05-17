#!/bin/bash

set -e

# Start FastAPI server
echo "> Starting FastAPI RAG backend..."
uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload &
FASTAPI_PID=$!

# Wait for FastAPI to be ready
until curl -s http://localhost:8000/healthz > /dev/null; do
  echo "> Waiting for FastAPI to start..."
  sleep 3
done

# Start Streamlit
echo "> Starting Streamlit UI..."
streamlit run frontend/app.py --server.port 8501 --browser.gatherUsageStats false --server.headless true --server.runOnSave true

# Cleanup (not reached unless Streamlit stops)
kill $FASTAPI_PID