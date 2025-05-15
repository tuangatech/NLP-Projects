# RAG Chatbot with Conversational Memory using LangChain

In a previous article, I walked through deploying a FastAPI model as a prediction service on AWS using Docker, ECS with Fargate, and ALB. That setup was great for getting an API live in the cloud, but for this project, we’re keeping it local — fast to spin up and perfect for experimenting.

This time, I’m building a Retrieval-Augmented Generation (RAG) chatbot that can reference your PDFs and hold the conversation with users. The goal is to show how to:
- Use LangChain to wire together chunking, embeddings, and prompt chains
- Use a vector database (ChromaDB) to store and retrieve knowledge efficiently
- Use FastAPI to .......
- Use Streamlit to build a simple real-time UI that streams chatbot replies as they generate

By keeping everything local with Docker Compose, we skip cloud deployment headaches and focus on making the core pipeline work — ingestion, retrieval, re-ranking, chat memory, and citation formatting — all running on your own machine.

Let’s break down how it works and how you can tweak it for your own projects.

## High Level Solution
This RAG chatbot design focuses on delivering a powerful yet manageable solution for local deployment, making it easy to get up and running with your own PDF knowledge. It's built to give you accurate and reliable answers by using a pretty smart retrieval system with re-ranking and checks against hallucinations, all packaged in a flexible, Docker-based setup that's easy to update and maintain.

- Dockerized setup using docker-compose
  - One container runs the chatbot (FastAPI + Streamlit)
  - Another runs ChromaDB for vector search
- PDFs are mounted, not baked in → update docs without rebuilding anything
- All logic (ingestion, search, chat) handled by LangChain
- PDF files are read, split into chunks, being embedded and stored in ChromeDB
- FastAPI backend handles vector search, LLM prompts, and API endpoints
- Streamlit frontend streams chatbot replies in real-time
- Retrieval pipeline includes Searching top chunks from ChromaDB, Re-ranking them with a cross-encoder and Build final prompt with history + context + few-shot examples.
- Responses are validated with a lightweight verifier to catch hallucinations
- Conversation history is saved to SQLite


## Implementation
### 1. Ingestion Pipeline 
1.1 Docker & Volumes
- Docker‑first: Two containers via Docker Compose
  - App container: FastAPI + Streamlit code
  - ChromaDB container: self‑hosted vector store
- Mounted PDF volume: drop new PDFs in, no image rebuild needed

1.2 PDF → Text Chunks
- Loader: PyMuPDFLoader for fast, reliable PDF parsing
- Chunking: LangChain’s RecursiveCharacterTextSplitter
  - ~500 tokens per chunk
  - 100‑token overlap to keep context

1.3 Embeddings & Storage
- Embedding model: OpenAI text-embedding-ada-002
- Vector store: ChromaDB with
  - HNSW indexing for millisecond‑scale nearest‑neighbor queries
  - Persistent Docker volume (won’t lose data on restarts)
- Backup: cron/rsync of the Chroma volume to host backup dir

1.4 “Run‑once” Logic
- Detect existing embeddings via a flag file or metadata. Or storing ingestion timestamps or checksums in a small SQLite
- Re‑ingestion only when you flip the flag or hit a “refresh” API endpoint
- Avoids re‑embedding every startup

### 2. Chat Runtime
2.1 Safety & Input Validation
- Sanitize & trim incoming queries (limit length, strip weird chars) to prevent prompt injection
- Rate‑limit at the API level, even if “only local”, ready for production

2.2 Memory & Session
- Hybrid memory: token‑limit + last k‑turns strategy for fast, short chats
- Session persistence: dump conversation logs to SQLite
  - Helpful for debugging
  - Can resurrect halfway‑done sessions

2.3 Retrieval & Re‑Ranking
- Vector search: similarity + optional metadata filters
- Re‑rank (Post-Retrieval Refinement) top 10 hits with a cross‑encoder (e.g., ms‑marco‑MiniLM‑L‑6‑v2)
- Narrow down to the 3 most relevant chunks

2.4 Chain & Prompt Assembly
- LCEL (LangChain Expression Lang) for modular, composable “runnables”
- Prompt builder stitches together:
  - Chat history
  - Re‑ranked context
  - Citation instructions
  - Few‑shot examples 

2.5 LLM Configuration
- Model: `gpt‑4o‑mini` (low temp ~0.2) for consistency, less ... creative
- Streaming: LangChain’s `stream()` hooked into Streamlit for live typing in Streamlit

2.6 Hallucination Checks & Citations
- Citation parser: pull page number metadata from chunks → append [1] style refs
- Verifier step: “Does the answer match the retrieved context chunks?” via a quick `gpt-4.1-nano-2025-04-14` check. May use embedding‑cosine threshold for basic validation at no cost.
- Output Parsing: Use StrOutputParser() or custom logic to format responses.

2.7 Backend & Frontend Glue
- FastAPI
  - Async routes for streaming + DB queries
  - Validate requests using Pydantic models
  - `/health` + `/metrics` endpoints
- Docker healthcheck: make sure ChromaDB is alive before spinning up the app
- Streamlit UI: renders streamed tokens in real time
- The application itself consists of a FastAPI backend providing the RAG logic and API endpoints, including a `/health` endpoint for monitoring. For user interaction, a Streamlit frontend is utilized. To simplify the deployment for this local setup, both the FastAPI backend and the Streamlit frontend will reside within the same Docker container, built from a single Dockerfile that includes all necessary dependencies.

### 3. Benefits
- Hot‑swap PDFs without Docker image rebuilds or redundant re-embedding
- Persistent data: embeddings + chat logs survive reboots, backed up via rsync
- Docker Compose and LangChain foster a clean design with clearly separated, maintainable services (vector DB, API, and UI).
- Fast prototyping: Streamlit + FastAPI in one image → minimal infra overhead
- Context‑aware chats: Real conversational memory (persisted in SQlite), re-ranking plus decent OpenAI models means it should actually understand context.
- Cost‑friendly for small/local projects (no managed vector DB bills)
- Concurrency‐ready: FastAPI async + HNSW indexes scale to multiple users
- Safety nets: input validation + hallucination checks + citation formatting
- Responsive UX: streaming gives that “chat‑app” feel instead of batch replies


