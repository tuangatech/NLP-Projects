#!/bin/sh
set -e
# Log timestamp and attempt
echo "[$(date)] Checking ChromaDB health..."
response=$(curl -s http://localhost:8000/api/v2/heartbeat)
if echo "$response" | grep -q "heartbeat"; then
  echo "OK"
  exit 0
else
  echo "FAIL: $response"
  exit 1
fi