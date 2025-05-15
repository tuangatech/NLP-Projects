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




### 1. Ingestion Pipeline: From PDFs to Searchable Knowledge
- PDF loading and parsing: consider `PyMuPDFLoader` (uses Fitz) for potentially better speed, accuracy, and metadata extraction (like table of contents, font styles which can infer headers)
- Docker best practices: data should be decoupled from containers, mount pdfs.
- Text Splitting (Chunking): 
  - RecursiveCharacterTextSplitter is a good default. Aim for semantically complete chunks. Or LangChain `TextSplitter`?
  - Experiment with chunk sizes (e.g., 500-1000 tokens) and overlap (e.g., 100-200 tokens). 
  - Consider layout-aware chunking. If possible, try to split based on sections, paragraphs, or even smaller logical units identified during parsing. For example, if you can detect headings, use them as split points. This keeps related context together.
  - Best Practice (Efficiency): Overlap helps ensure context isn't lost at the edges of chunks but increases storage. Find the right balance.
- Metadata Extraction for filtering and citations: file name, page number, section title. Store this metadata alongside the vector and the raw chunk_text.
- Embedding Model 
  - use the same at ingest and query
  - OpenAI `text-embedding-ada-002`: good performance, $0.10 per 1M tokens. 
  - Efficiency: Batch your embedding calls to the model provider.
- Vector Database Population: `Chroma DB` developer-friendly. Use batch upserting to the vector DB. Build a vector index with Hierarchical Navigable Small World (HNSW), it is very fast and efficient.
- Orchestration (Ingestion Trigger) for new files -- Will not implement.

### 2. Vector Database: The Knowledge Core
- Schema: chunk_text, embedding, metadata
- Ensure your metadata fields are indexed
- Regularly backup your vector database if it's self-hosted.
- Persistent volumes ensure ChromaDB data survives container restarts. Simpler than manual backups/restore workflows.

### 3. Conversation Memory: Short-Term Context
- ConversationBufferMemory: Simple, stores last k messages. Prone to abrupt context loss.
- ConversationTokenBufferMemory: Better, as it limits by token count, aligning with LLM limits. 1000-2000 tokens for history (separate from retrieval context) is a good starting point
- Ensure conversation memory is keyed by a unique session_id or user_id.

### 4. Retriever: Finding Relevant Knowledge
- Strategy: search_type="similarity": Standard cosine similarity.
- Number of Chunks (k): Retrieve a configurable number of chunks (e.g., top 3-5). Too few might miss context; too many can exceed token limits or introduce noise.
- (advanced) Re-ranking: After initial retrieval (e.g., top 20 chunks), use a more sophisticated re-ranking model (like a cross-encoder or even a smaller LLM) to pick the best 3-5 chunks. This can significantly improve relevance. LangChain has components for this.
- MultiQueryRetriever: Use an LLM to generate several variations of the user's query from different perspectives, perform a search for each, and take the union of results.

### 5. Prompt Template: Guiding the LLM
- Structure
```
System Prompt: You are a helpful AI assistant. Answer the user's query based ONLY on the provided context documents. If the information is not in the context, say "I do not have enough information to answer that." List the source file and page number for your answer.

Retrieved Context: ...
Chat History: ...
```
- Few-Shot Examples (Advanced): If you have common query types, you can **include a few examples of good question-answer pairs** in the prompt.

### 6. LLM: The Brain
- LLM: `gpt-4o-mini-2024-07-18`: $0.15 per 1M input tokens.
- Temperature: For factual Q&A from context, use a lower temperature (e.g., 0.0 - 0.3) to reduce creativity/hallucinations.
- Streaming: Use streaming APIs for a better user experience (words appear as they are generated). LangChain supports this.
- Error Handling: Implement retries and fallbacks for LLM API calls.

### 7. Post-Processing: Refining the Output
- Citation Generation: post-processing step can parse these citations from answer.
- Verification: 
  - Extract the cited chunk(s).
  - Make **another LLM** call asking it to verify if the generated answer statement is supported by only that specific cited chunk. This adds latency but boosts faithfulness.

### 8. Deployment: Serving the Chatbot
- API Layer (FastAPI):
  - Best Practice (Efficiency): Use FastAPI's async def for your endpoints to handle I/O-bound operations (LLM calls, vector DB queries) concurrently without blocking.
  - Implement proper request validation (using Pydantic models).
- LangChain Runnable: Use LangChain Expression Language (LCEL) to define your chain as a Runnable. This makes it easy to invoke, stream, and batch.
- Containerization (Docker): Standard practice.
Hosting with ECS/Fargate: Good for long-running applications, persistent WebSocket connections, more control. Autoscaling based on CPU/memory or custom metrics.
- Best Practice: Ensure proper health checks for your FastAPI application (e.g., a /health endpoint ECS/ALB can ping).

### Streamlit:
- Streamlit and FastAPI can coexist in the same Python environment.
- Runs FastAPI and Streamlit in parallel via entrypoint
- Streamlit connects to FastAPI endpoints for RAG responses
- UX : Add a "Regenerate" button and display retrieved chunks in an expandable sidebar for transparency.
- If different container: challenges of cross-container networking setup (e.g., CORS, service discovery).
- When to split? If you need horizontal scaling (e.g., multiple Streamlit instances).

### Local Dev Tools :
- Use dotenv to manage secrets (e.g., OPENAI_API_KEY).
- Add a /reload endpoint to refresh ChromaDB data without restarting.

### Others
1. Monitoring & Logging:

- Log user queries, retrieved chunks, final LLM responses, latencies of each step (retrieval, LLM generation), token counts.
-  LangSmith is excellent for tracing and debugging LangChain applications.

2. Evaluation Framework:

- Create a "golden dataset" of question-answer pairs based on your PDFs.
- Metrics: RAGAs (faithfulness, answer relevancy, context precision/recall)

3. Security: Sanitize inputs.

### Concerns:
- How to detect "How are you?" with questions? Or just type question only?


=======

Building a RAG Chatbot with Conversational Memory for local Deployment. Chatbot's knowledge comes from some pdf input files. Tech stack includes:
- Orchestration framework: LangChain
- PDF loading and parsing: `PyMuPDFLoader`
- Mount the folder of PDF files as a volume (outside the container) instead of baking them into the Docker image
- Text Splitting: pick the best between `RecursiveCharacterTextSplitter` and LangChain `TextSplitter`
- Chunk sizes 500 tokens and overlap 100 tokens
- Embedding model: OpenAI `text-embedding-ada-002`
- Vector database: Self-hosted ChromaDB with persistent volume, avoid copying ChromaDB into the Docker Image. Data is indexed by HNSW. A way to backup data, not regenerating data everytime restarting docker container.
- Separating the app and ChromaDB into distinct containers using `Docker Compose` with a single docker-compose.yml to coordinate the services
- Avoid Re-Embedding PDFs, skip ingestion if ChromaDB already exists
- Adding health checks in Docker Compose to ensure ChromaDB is ready before starting the app
- Orchestration: Manual Ingestion Trigger for new files
- LLM: OpenAI `gpt-4o-mini-2024-07-18`
- Backend RAG logic and APIs with FastAPI. Add /health endpoint in FastAPI
- Frontend layer with Streamlit. Run Streamlit and FastAPI in the same container for simplicity. Use a single Dockerfile with both dependencies.
- Conversational Memory: LangChain's memory components
- Session Persistence: Store chat history in SQLite, in the same container with FastAPI and StreamLit?


AWS Deployment Options: to run your embedding model, vector database, and serve the API - AWS ECS (Fargate) with Docker
Framework: LangChain for orchestration
Memory: LangChain's ConversationBufferMemory
Frontend: Streamlit or Gradio deployed on the same instance

### Implementation Architect
1. PDF Processing Pipeline
   └── Document loaders (PyPDF2/pdfplumber)
   └── Text chunking (LangChain TextSplitter)
   └── Embedding generation
   └── Storage in Chroma DB

2. Chat System
   └── User query embedding
   └── Chroma DB similarity search
   └── Relevant context retrieval
   └── Conversation memory (last N exchanges)
   └── OpenAI API prompt construction & completion
   └── Response to user