"""
Professional RAG Streamlit Web Application
Supports PDF, TXT, and JSON file uploads with real-time chat interface
Enhanced with conversational memory, adaptive chunking, and improved retrieval stability
"""

import streamlit as st
import os
import json
import shutil
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
import requests

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory

# File handling imports
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

CHROMA_DB_DIR = Path("./chroma_db_streamlit")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "llama3"

# Adaptive chunking strategy
CHUNK_SIZE_DEFAULT = 500
CHUNK_OVERLAP_DEFAULT = 100
SMALL_DOC_THRESHOLD = 2000  # chars - bypass chunking for very small docs
MEDIUM_DOC_CHUNK_SIZE = 800  # Larger chunks for medium docs

# Retrieval settings
RETRIEVAL_K_MIN = 3
RETRIEVAL_K_DEFAULT = 8  # Increased from 3 for better context coverage
RETRIEVAL_SEARCH_TYPE = "mmr"  # "similarity" or "mmr" (max marginal relevance)
RETRIEVAL_FETCH_K = 20  # Fetch more candidates for MMR to select from

# Streamlit page config
st.set_page_config(
    page_title="RAG Chat Application",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 20px;
    }
    .stChatMessage {
        padding: 10px;
        border-radius: 10px;
    }
    .sidebar-title {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "document_processed" not in st.session_state:
        st.session_state.document_processed = False
    
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    
    if "memory" not in st.session_state:
        st.session_state.memory = None
    
    if "document_content" not in st.session_state:
        st.session_state.document_content = None
    
    if "current_file_name" not in st.session_state:
        st.session_state.current_file_name = None
    
    if "ollama_available" not in st.session_state:
        st.session_state.ollama_available = False
    
    if "embeddings_model" not in st.session_state:
        st.session_state.embeddings_model = None
    
    if "retrieval_config" not in st.session_state:
        st.session_state.retrieval_config = {
            "k": RETRIEVAL_K_DEFAULT,
            "search_type": RETRIEVAL_SEARCH_TYPE,
        }


initialize_session_state()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def check_ollama_connection() -> bool:
    """Check if Ollama service is running"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def extract_text_from_pdf(file) -> str:
    """Extract text from PDF file"""
    if not PDF_SUPPORT:
        raise ImportError("PyPDF2 not installed. Install with: pip install PyPDF2")
    
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        raise ValueError(f"Error reading PDF: {str(e)}")


def extract_text_from_txt(file) -> str:
    """Extract text from TXT file"""
    try:
        return file.read().decode('utf-8')
    except UnicodeDecodeError:
        return file.read().decode('latin-1')
    except Exception as e:
        raise ValueError(f"Error reading TXT: {str(e)}")


def extract_text_from_json(file) -> str:
    """Extract text from JSON file"""
    try:
        data = json.loads(file.read().decode('utf-8'))
        
        # Convert JSON to readable text
        if isinstance(data, dict):
            text = json.dumps(data, indent=2)
        elif isinstance(data, list):
            text = "\n".join([json.dumps(item, indent=2) for item in data])
        else:
            text = str(data)
        
        return text
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error reading JSON: {str(e)}")


def create_text_chunks(text: str) -> List[str]:
    """
    Split text into chunks using adaptive strategy.
    
    Strategy:
    - Very small docs (< SMALL_DOC_THRESHOLD): skip chunking to preserve context
    - Medium docs: use standard chunking with default parameters
    - Large docs: keep current chunking strategy
    """
    text_length = len(text)
    
    # If document is very small, skip chunking to avoid over-fragmentation
    if text_length < SMALL_DOC_THRESHOLD:
        return [text]
    
    # Determine chunk size based on document length
    if text_length < 50000:  # Less than ~50KB
        chunk_size = CHUNK_SIZE_DEFAULT
    else:
        chunk_size = MEDIUM_DOC_CHUNK_SIZE
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=CHUNK_OVERLAP_DEFAULT,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)


# ============================================================================
# MODULAR RAG PIPELINE BUILDERS
# ============================================================================

def build_embeddings() -> HuggingFaceEmbeddings:
    """
    Build or retrieve embeddings model.
    Caches in session state to avoid reloading.
    """
    if st.session_state.embeddings_model is None:
        with st.spinner("📦 Loading embeddings model..."):
            st.session_state.embeddings_model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL
            )
    return st.session_state.embeddings_model


def build_vectorstore(text_chunks: List[str]) -> Chroma:
    """
    Build Chroma vector store with production-grade robustness.
    
    Key improvements:
    - Forces memory cleanup of previous vectorstore
    - Clears persistent directory to avoid lock issues
    - Handles edge cases gracefully
    """
    # 1. Force clear the existing objects from memory
    st.session_state.vectorstore = None
    st.session_state.rag_chain = None
    
    # 2. Clear the persistent directory
    persist_directory = CHROMA_DB_DIR
    if persist_directory.exists():
        try:
            shutil.rmtree(persist_directory)
            time.sleep(0.5)  # Wait for OS to release locks
        except Exception as e:
            st.warning(f"⚠️ Could not clear old vectorstore: {e}")
    
    # 3. Get embeddings
    embeddings = build_embeddings()
    
    # 4. Create fresh vector store
    try:
        vectorstore = Chroma.from_texts(
            texts=text_chunks,
            embedding=embeddings,
            persist_directory=str(persist_directory),
            collection_name="documents"
        )
        return vectorstore
    except Exception as e:
        st.error(f"❌ Failed to create vectorstore: {e}")
        return None


def build_retriever(vectorstore: Chroma):
    """
    Build retriever with configurable search settings.
    
    Supports:
    - Dynamic k selection based on corpus size
    - MMR (Max Marginal Relevance) for diversity
    - Standard similarity search
    
    Returns: Chroma retriever object
    """
    if vectorstore is None:
        return None
    
    # Dynamically determine k based on chunk count
    config = st.session_state.retrieval_config
    k = config["k"]
    search_type = config["search_type"]
    
    # For MMR search, we fetch more candidates then select top k
    if search_type == "mmr":
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": RETRIEVAL_FETCH_K,
                "lambda_mult": 0.5  # Balance relevance vs diversity
            }
        )
    else:
        # Standard similarity search
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
    
    return retriever


def build_rag_chain(vectorstore: Chroma) -> Optional[ConversationalRetrievalChain]:
    """
    Build ConversationalRetrievalChain with conversational memory.
    
    Key improvements:
    - Uses ConversationalRetrievalChain instead of stateless RetrievalQA
    - Maintains ConversationBufferMemory for multi-turn context
    - Improved prompt template that enforces grounding
    - Handles Ollama initialization errors gracefully
    
    Returns: Initialized chain or None on error
    """
    if vectorstore is None:
        st.error("❌ No vectorstore available")
        return None
    
    # Initialize LLM
    try:
        llm = Ollama(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.1
        )
    except Exception as e:
        st.error(f"❌ Failed to initialize LLM: {str(e)}")
        return None
    
    # Build retriever
    retriever = build_retriever(vectorstore)
    if retriever is None:
        return None
    
    # Initialize or reset memory
    if st.session_state.memory is None:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    
    # Improved prompt template with better grounding
    qa_prompt = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template="""You are a helpful AI assistant. Use ONLY the provided context documents to answer questions.

CRITICAL RULES:
1. Base all answers strictly on the provided context
2. If context doesn't address the question, clearly state: "This information is not in the provided documents"
3. Never make assumptions or use outside knowledge
4. Quote or paraphrase from context when possible
5. If uncertain, acknowledge the uncertainty: "The documents are unclear about..."
6. Maintain consistency with earlier conversation turns

Context from documents:
{context}

Chat history:
{chat_history}

Question: {question}

Answer (based ONLY on context):""",
    )
    
    # Build the conversational chain
    try:
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=st.session_state.memory,
            chain_type="stuff",
            get_chat_history=lambda h: h,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            return_source_documents=True,
            verbose=False
        )
        return chain
    except Exception as e:
        st.error(f"❌ Failed to create RAG chain: {str(e)}")
        return None


def process_document(uploaded_files) -> bool:
    """
    Process uploaded documents and create RAG chain.
    
    Improvements:
    - Uses new modular RAG builders
    - Adaptive chunking strategy
    - Proper error handling
    - Resets conversation memory for new documents
    """
    try:
        all_text = ""
        file_names = [f.name for f in uploaded_files]
        
        with st.spinner(f"📖 Reading {len(uploaded_files)} document(s)..."):
            for uploaded_file in uploaded_files:
                text_content = load_document(uploaded_file)
                if text_content:
                    # Add a header so the LLM knows which file it's reading
                    all_text += f"\n\n--- Source: {uploaded_file.name} ---\n{text_content}"
            
            if not all_text:
                st.error("❌ No text content extracted from documents")
                return False
            
            st.session_state.document_content = all_text
            st.session_state.current_file_name = ", ".join(file_names)
        
        # Adaptive chunking
        with st.spinner("✂️ Processing document with adaptive chunking..."):
            chunks = create_text_chunks(all_text)
            st.session_state.chunks_count = len(chunks)
            st.info(f"📊 Created {len(chunks)} chunks (adaptive strategy)")
        
        # Build vectorstore
        with st.spinner("🧠 Creating embeddings (this may take a minute on first run)..."):
            vectorstore = build_vectorstore(chunks)
            if vectorstore is None:
                return False
            st.session_state.vectorstore = vectorstore
        
        # Build RAG chain with conversational memory
        with st.spinner("🔗 Setting up RAG chain with memory..."):
            # Reset memory for new document
            st.session_state.memory = None
            
            rag_chain = build_rag_chain(vectorstore)
            if rag_chain is None:
                return False
            st.session_state.rag_chain = rag_chain
        
        st.session_state.document_processed = True
        st.session_state.messages = []  # Reset messages for new documents
        
        return True
    
    except Exception as e:
        st.error(f"❌ Error processing documents: {str(e)}")
        return False


def load_document(uploaded_file) -> Optional[str]:
    """Load and extract text from uploaded file"""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_extension == 'pdf':
            return extract_text_from_pdf(uploaded_file)
        elif file_extension == 'txt':
            return extract_text_from_txt(uploaded_file)
        elif file_extension == 'json':
            return extract_text_from_json(uploaded_file)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        return None


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("# 📚 RAG Chat Application")
    with col2:
        if st.session_state.ollama_available:
            st.markdown("🟢 **Ollama Running**")
        else:
            st.markdown("🔴 **Ollama Offline**")
    
    st.markdown("Upload a document and chat with it using AI")
    st.divider()
    
    # ========================================
    # SIDEBAR - FILE UPLOAD & STATUS
    # ========================================
    
    with st.sidebar:
        st.markdown("## ⚙️ Setup & Control")
        
        # Ollama status check
        if st.button("🔍 Check Ollama Status", key="check_ollama"):
            st.session_state.ollama_available = check_ollama_connection()
            st.rerun()
        
        if not st.session_state.ollama_available:
            st.warning(
                "⚠️ **Ollama Not Running**\n\n"
                "Please start Ollama in a terminal:\n"
                "`ollama serve`\n\n"
                "Then make sure Llama-3 is available:\n"
                "`ollama pull llama3`"
            )
        
        st.markdown("---")
        st.markdown("## 🎛️ Retrieval Tuning")
        
        # Configurable retrieval parameters
        config = st.session_state.retrieval_config
        k_value = st.slider(
            "Retrieval k (documents to fetch)",
            min_value=RETRIEVAL_K_MIN,
            max_value=20,
            value=config["k"],
            help="Higher k = more context but potentially more noise"
        )
        config["k"] = k_value
        
        search_type = st.selectbox(
            "Search strategy",
            ["similarity", "mmr"],
            index=0 if config["search_type"] == "similarity" else 1,
            help="MMR = Max Marginal Relevance (more diverse results)"
        )
        config["search_type"] = search_type
        
        if config["search_type"] != search_type or config["k"] != k_value:
            st.info("💡 Tuning applied to next query")
        
        st.markdown("---")
        st.markdown("## 📤 Upload Document")
        
        # File upload - support multiple files
        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "txt", "json"],
            accept_multiple_files=True,
            help="Supported formats: PDF, TXT, JSON"
        )
        
        if uploaded_files:
            total_size = sum(f.size for f in uploaded_files) / 1024
            st.info(f"📄 **{len(uploaded_files)} file(s) selected** | Total size: {total_size:.2f} KB")
            for file in uploaded_files:
                st.caption(f"  • {file.name} ({file.size / 1024:.2f} KB)")
            
            if st.button("🚀 Process Documents", key="process_doc"):
                if not st.session_state.ollama_available:
                    st.error("❌ Ollama service is not running!")
                else:
                    success = process_document(uploaded_files)
                    if success:
                        st.success(
                            f"✅ Documents processed successfully!\n\n"
                            f"Chunks: {st.session_state.chunks_count}\n"
                            f"Files: {st.session_state.current_file_name}"
                        )
                        st.rerun()
        
        # Document status
        st.markdown("---")
        st.markdown("## 📊 Document Status")
        
        if st.session_state.document_processed:
            st.success(f"✅ Document loaded: {st.session_state.current_file_name}")
            st.info(f"Chunks: {st.session_state.chunks_count}")
            
            if st.button("🔄 Load New Document", key="new_doc"):
                st.session_state.document_processed = False
                st.session_state.vectorstore = None
                st.session_state.rag_chain = None
                st.session_state.memory = None  # Reset memory
                st.session_state.document_content = None
                st.session_state.messages = []
                st.rerun()
        else:
            st.info("ℹ️ No document loaded yet. Upload and process a file to get started.")
        
        st.markdown("---")
        st.markdown("## ℹ️ Help")
        with st.expander("How to use"):
            st.markdown("""
            1. **Upload** a PDF, TXT, or JSON file
            2. **Process** the document to create embeddings
            3. **Tune** retrieval parameters if desired
            4. **Chat** with the AI about your document
            
            ### Improvements in this version:
            - **Conversational Memory**: Maintains context across multiple turns
            - **Adaptive Chunking**: Small docs stay intact, large docs are split intelligently
            - **Better Grounding**: Prompt enforces context-only answers
            - **Retrieval Tuning**: Adjust k and search strategy on the fly
            - **MMR Search**: Optional diversity-aware retrieval
            """)
        
        with st.expander("Configuration"):
            st.markdown(f"""
            **Current Settings:**
            - Retrieval k: {config['k']}
            - Search type: {config['search_type']}
            - Chunk size: {CHUNK_SIZE_DEFAULT} (small docs: adaptive)
            - Embeddings: {EMBEDDING_MODEL.split('/')[-1]}
            - LLM: {LLM_MODEL}
            """)
    
    # ========================================
    # MAIN AREA - CHAT INTERFACE
    # ========================================
    
    if not st.session_state.document_processed:
        st.info(
            "👈 **Get Started**\n\n"
            "1. Click 'Upload Document' in the sidebar\n"
            "2. Select a PDF, TXT, or JSON file\n"
            "3. Click 'Process Document'\n"
            "4. Start asking questions!"
        )
        return
    
    # Chat header
    st.markdown(f"### 💬 Chat with: {st.session_state.current_file_name}")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📌 Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**[{i}]** {source[:200]}...")
    
        # Chat input
    if prompt := st.chat_input("Ask a question about the document..."):
        if not st.session_state.ollama_available:
            st.error("❌ Ollama service is not running!")
            return
        
        # Add user message to chat
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response with improved error handling
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    # ConversationalRetrievalChain uses different input format
                    result = st.session_state.rag_chain(
                        {"question": prompt},
                        return_only_outputs=False
                    )
                    
                    # Extract response and sources
                    response_text = result.get("answer", "No response generated")
                    sources = []
                    
                    if result.get("source_documents"):
                        sources = [
                            doc.page_content[:200]
                            for doc in result["source_documents"]
                            if doc.page_content  # Skip empty documents
                        ]
                    
                    # Defensive check for empty response
                    if not response_text or response_text.strip() == "":
                        response_text = "❌ The model did not generate a response. Please try again."
                    
                    # Display response
                    st.markdown(response_text)
                    
                    # Show sources if available
                    if sources:
                        with st.expander("📌 Sources"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**[{i}]** {source}...")
                    else:
                        # Note when no sources were found
                        st.caption("ℹ️ No relevant context found for this query")
                    
                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text,
                        "sources": sources
                    })
                
                except requests.exceptions.Timeout:
                    st.error(
                        "❌ Ollama timed out. The model may be busy.\n\n"
                        "Try:\n"
                        "- Simplifying your question\n"
                        "- Waiting a moment and retrying\n"
                        "- Restarting Ollama: `ollama serve`"
                    )
                
                except requests.exceptions.ConnectionError:
                    st.error(
                        "❌ Cannot connect to Ollama.\n\n"
                        "Start it with: `ollama serve`"
                    )
                
                except Exception as e:
                    st.error(f"❌ Error generating response: {str(e)}")
                    st.info("💡 Make sure: Ollama is running and Llama-3 model is pulled (`ollama pull llama3`)")


if __name__ == "__main__":
    main()
