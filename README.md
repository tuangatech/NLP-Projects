# NLP Projects Portfolio

Welcome! Below are hands-on NLP side projects I've built to deepen my skills and experiment with useful real-world applications. Each one built from different stacks, from fine-tuning LLMs to building full-stack RAG pipelines.

---

## 1. RAG Chatbot with Conversational Memory using LangChain

**Use Case**  
A chatbot that supports contextual conversations on tax guidance documents (Forms 1040 & 1065). Users can chat "Tell me what are taxpayer rights?", the chatbot will find relevant information from 2 documents and respond with citation to users.

**Tech Stack**  
- LangChain for chaining chunking, embedding, prompting  
- ChromaDB for semantic retrieval  
- FastAPI backend with `/query` and `/healthcheck` endpoints  
- Streamlit frontend with real-time response UI  
- Docker Compose for local orchestration



## 2. Data Enrichment with OpenAI

**Use Case**  
A book resale business needed to clean and enrich book data coming from inconsistent sources. This project automates metadata completion using AI, reducing manual effort by 80%.

**Tech Stack**  
- Web crawling to retrieve book info using Bing and Google Search
- Prompt generation from search snippets  
- OpenAI LLMs for structured info extraction  
- Post-processing and DB integration for clean records  
- Python + Requests + OpenAI API



## 3. Intent Detection with RoBERTa

**Use Case**  
Detect customer intent in messages like "How could I get my money back?" by mapping to intents like `get_refund`. Tailored for chatbot or support automation.

**Tech Stack**  
- Fine-tuned `RoBERTa` on a customer support dataset
- PyTorch + HuggingFace Transformers  
- Training loop with early stopping and evaluation  
- Jupyter/Colab for experimentation



## 4. RAG Pipeline with Haystack

**Use Case**  
Q&A assistant for company policies using the UNG Employee Handbook PDF. The system fetches relevant context and feeds it to an LLM for grounded answers.

**Tech Stack**  
- Haystack pipeline with `sentence-transformers/all-MiniLM-L6-v2`  
- InMemoryDocumentStore for vector search  
- `GPT-4o-mini` for generative response  
- End-to-end semantic search + generation pipeline



## 5. Recipe Generator with GPT-2

**Use Case**  
Give it ingredients like "eggs, mushroom, butter, sugar", and it generates full cooking instructions using them. Built for creativity and fun.

**Tech Stack**  
- Fine-tuned GPT-2 using a recipes dataset  
- PyTorch Datasets and Dataloaders  
- Text preprocessing and tokenization  
- Basic training loop with generation script



## 6. Sentiment Analysis with BiLSTM

**Use Case**  
Classify Amazon product reviews as Positive or Negative. Trains a custom BiLSTM with attention to understand emotional tone in reviews.

**Tech Stack**  
- BiLSTM + Attention layer  
- PyTorch Dataset/Dataloader setup  
- Gradient clipping to avoid exploding gradients  
- Sentiment classification using binary cross-entropy loss

---

🚀 Each project represents a focused exploration into either LLMs, NLP techniques, or full-stack AI systems. Contributions, questions, or feedback are welcome!
