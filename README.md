# 📚 Multi-Document RAG Assistant (Llama-3)

A production-ready Retrieval-Augmented Generation (RAG) web application. This tool allows users to upload multiple documents (PDF, JSON, TXT) and engage in a context-aware chat powered by Llama-3 and ChromaDB.

## 🚀 Key Features
- **Multi-Format Support**: Ingests and processes PDF, JSON, and Text files simultaneously.
- **Local Inference**: Powered by Llama-3 via Ollama for 100% data privacy.
- **Vector Intelligence**: Uses `all-MiniLM-L6-v2` for high-accuracy semantic search.
- **Streamlit UI**: A clean, responsive chat interface with source citations.

## 🛠️ Tech Stack
- **LLM**: Llama-3 (8B Instruct)
- **Orchestration**: LangChain
- **Vector Store**: ChromaDB
- **Backend/UI**: Streamlit & FastAPI-ready logic
- **Containerization**: Docker (See Dockerfile)

## 🔧 Setup & Usage
1. **Prepare Ollama**: Ensure `ollama serve` is running and `llama3` is pulled.
2. **Install Dependencies**: `pip install -r requirements.txt`
3. **Launch App**: `streamlit run app.py`

## 📂 Project Architecture
The system follows a standard RAG pipeline: 
Document Ingestion → Text Chunking → Vector Embedding → Semantic Retrieval → LLM Generation.
