from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from chromadb import HttpClient
import logging
from .config import CHROMA_HOST, CHROMA_PORT, COLLECTION_NAME, EMBEDDING_MODEL

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize and return a Chroma vector store instance connected to the remote server
def init_vectorstore(embeddings=None):
    # Initialize embeddings and vector store
    try:
        if embeddings is None:
            embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

        client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        
        vectorstore = Chroma(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
        )
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}")
        raise
    
    return vectorstore

# Get a retriever from the vectorstore
def get_retriever(vectorstore=None, search_kwargs=None):
    if vectorstore is None:
        vectorstore = init_vectorstore()
    
    if search_kwargs is None:
        search_kwargs = {"k": 6}

    return vectorstore.as_retriever(search_kwargs=search_kwargs)