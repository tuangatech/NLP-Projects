# backend/api.py
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnableMap, RunnableLambda
from langchain_core.documents import Document
import logging

from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
import numpy as np
import os
from pprint import pprint
from typing import List, Optional, Dict
from .vectorstore import get_retriever
from .prompting import format_context, format_chat_history, build_prompt


# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Chatbot API")
class QueryRequest(BaseModel):
    question: str = Field(
        ...,            # Ellipsis (...) = Required field
        min_length=5,   # Minimum length of the question
        max_length=100  # Maximum length of the question
    )
    chat_history: List[Dict[str, str]] = []

# Load environment variables from .env file
load_dotenv("/app/.env")

# Load environment variables
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GPT_MODEL_1 = os.getenv("GPT_MODEL_1", "gpt-4o-mini-2024-07-18")
GPT_MODEL_2 = os.getenv("GPT_MODEL_2", "gpt-4.1-nano-2025-04-14")
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
TOP_K = 2   # Number of top documents for reranking

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY must be set.")
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# Base retriever (similarity search) from vectorstore
retriever = get_retriever(search_kwargs={"k": 6})
# Initialize CrossEncoder for reranking
cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

# metadata strucuture: {'file_name': 'i1040.pdf', 'page_number': 1, 'chunk_index_in_doc': 2, 'section_title': '2024 Changes'}
# Function to rerank documents based on the question and their content
def rerank_fn(inputs: dict) -> list[Document]:
    question = inputs["question"]
    docs = inputs["docs"]
    try:
        pairs = [[question, doc.page_content] for doc in docs]
        scores = cross_encoder.predict(pairs)
        scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        top_k_docs = [doc for doc, score in scored_docs[:TOP_K]]
        for i, (doc, score) in enumerate(scored_docs[:TOP_K]):
            logger.info(f"Rank {i+1}: Section: {doc.metadata.get('section_title', 'N/A')}, Score: {score:.2f}")
        return top_k_docs
    except Exception as e:
        logger.error(f"Error during reranking: {e}")
        return []

reranker = (
    RunnableMap({
        # This will call retriever.invoke(input["question"])
        "docs": RunnableLambda(lambda x: x["question"]) | retriever,  
        "question": lambda x: x["question"]
    }) 
    | RunnableLambda(rerank_fn)
)

def rerank_docs(docs, question):
    return reranker.invoke({"docs": docs, "question": question})

def retrieve_docs(question: str):
    return retriever.invoke(question)

# Validate the answer based on the context and the answer provided by the LLM
# This is a simple cosine similarity check between the answer and the context
def validate_answer(answer: str, docs: list[Document], threshold=0.75) -> bool:
    if not answer or not docs:
        return False
    try:
        context_str = " ".join(doc.page_content for doc in docs)
        ctx_emb = embeddings.embed_query(context_str)
        ans_emb = embeddings.embed_query(answer)
        similarity = np.dot(ctx_emb, ans_emb) / (np.linalg.norm(ctx_emb) * np.linalg.norm(ans_emb))
        return similarity > threshold
    except Exception as e:
        logger.warning(f"Validation failed: {e}")
        return False

# Initialize LLM
llm = ChatOpenAI(
    model=GPT_MODEL_1,
    temperature=0.1,
    openai_api_key=OPENAI_API_KEY
)

def call_model(prompt_text: str):
    response = llm.invoke(prompt_text)
    return response.content

chain = (
    # Initializes a dictionary with two keys: "question" and "chat_history" 
    # and maps them to the corresponding values from the input
    RunnableMap({
        "question": lambda x: x["question"],
        "chat_history": lambda x: x.get("chat_history", [])
    })
    # Calls retrieve_docs(question) to find the most similar documents to the question
    | RunnableLambda(lambda x: {
        **x,
        "docs": retrieve_docs(x["question"])
    })
    # Calls rerank_docs(docs, question) to rerank the retrieved documents
    | RunnableLambda(lambda x: {
        **x,
        "docs": rerank_docs(x["docs"], x["question"])
    })
    | RunnableLambda(lambda x: {
        **x,
        "context": format_context(x["docs"]),
        "formatted_chat_history": format_chat_history(x["chat_history"]),
    })
    | RunnableLambda(lambda x: {
        **x,
        "prompt": build_prompt(x["context"], x["formatted_chat_history"], x["question"])
    })
    # Calls the LLM with the formatted prompt and stores the raw output
    | RunnableLambda(lambda x: {
        **x,
        "raw_output": call_model(x["prompt"]),
    })
    # Return the final result as a dictionary with keys: "answer", "is_verified", and "docs"
    # "answer" is the raw output from the LLM
    | RunnableLambda(lambda x: {
        "answer": x["raw_output"],
        "is_verified": bool(validate_answer(x["raw_output"], x["docs"])),
        "docs": x["docs"]
    })
)

@app.post("/query")
async def query_rag(request: QueryRequest):
    question = request.question
    chat_history = request.chat_history
    logger.info(f"Question: {question}")

    if not question or len(question.strip()) < 5:
        raise HTTPException(status_code=400, detail="Invalid or too short question.")
    
    try:
        result = chain.invoke({
            "question": question,
            "chat_history": chat_history
        })

        if not result or not result["docs"]:
            logger.warning("No relevant documents found.")
            raise HTTPException(status_code=404, detail="No relevant documents found.")

        logger.info(f"> Result:")
        pprint(result, width=50, indent=2, sort_dicts=True)

        response = {
            "question": question,
            "answer": result["answer"],
            "verified": result["is_verified"],
            "source_documents": [
                {
                    "file_name": doc.metadata.get("file_name", "unknown"),
                    "section_title": doc.metadata.get("section_title", "N/A"),
                    "page_number": doc.metadata.get("end_page", "N/A"),
                    "page_content": doc.page_content
                }
                for doc in result["docs"]
            ]
        }

        return response
    except Exception as e:
        logger.error(f"Internal server error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error occurred.")
    
@app.get("/healthz")
def health_check():
    return {"status": "ok"}