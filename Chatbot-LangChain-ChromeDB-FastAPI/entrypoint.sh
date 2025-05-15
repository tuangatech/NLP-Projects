#!/bin/bash

# Wait for OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
  echo "Error: OPENAI_API_KEY is not set."
  exit 1
fi

# Wait for ChromaDB to be ready
until curl -s http://chroma:8000/api/v1/heartbeat > /dev/null; do
  echo "Waiting for ChromaDB..."
  sleep 5
done

# Run ingestion only once
if [ ! -f "/chroma_initialized" ]; then
  echo "Running initial ingestion..."
  python ingestion_pipeline.py
  touch /chroma_initialized
else
  echo "ChromaDB already initialized. Skipping ingestion."
fi

# Start FastAPI and Streamlit in parallel
echo "Starting services..."
uvicorn backend.api:app --host 0.0.0.0 --port 8000 &
streamlit run frontend/app.py --server.port 8501