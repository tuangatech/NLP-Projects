# chromadb-custom/Dockerfile
FROM chromadb/chroma:latest

# Install curl and grep using apt-get
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl gnupg libgl1 libsm6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies needed for healthcheck
# RUN apk add --no-cache curl grep