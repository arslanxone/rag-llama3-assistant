#!/usr/bin/env python3
"""
RAG System for Espresso Evolution Document
Uses Llama-3 with local Ollama instance for chat interactions
"""

import os
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


def setup_rag_system(document_path: str, collection_name: str = "espresso_evolution"):
    """
    Initialize the RAG system with document and vector store
    """
    print("🚀 Initializing RAG System...")
    
    # 1. Read the document
    print(f"📄 Loading document: {document_path}")
    with open(document_path, 'r', encoding='utf-8') as f:
        document_content = f.read()
    
    # 2. Split text into chunks
    print("✂️  Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_text(document_content)
    print(f"   Created {len(chunks)} chunks")
    
    # 3. Create embeddings
    print("🧠 Initializing embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # 4. Create vector store
    print("🗄️  Creating vector store...")
    persist_directory = Path("./chroma_db")
    persist_directory.mkdir(exist_ok=True)
    
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(persist_directory)
    )
    
    print(f"✅ Vector store created with {len(chunks)} documents")
    return vectorstore, embeddings


def initialize_llm():
    """
    Initialize Llama-3 LLM via Ollama
    """
    print("\n🤖 Initializing Llama-3 LLM...")
    print("   Make sure Ollama is running and Llama-3 is pulled locally")
    print("   Install ollama from: https://ollama.ai")
    print("   Run: ollama pull llama3")
    print("   Then: ollama serve")
    
    llm = Ollama(
        model="llama3",
        base_url="http://localhost:11434",  # Default Ollama port
        temperature=0.7
    )
    
    return llm


def create_rag_chain(vectorstore, llm):
    """
    Create the RAG retrieval chain
    """
    print("\n🔗 Setting up RAG retrieval chain...")
    
    # Custom prompt template
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are an expert on coffee machine history and espresso technology.
Use the following context from the Espresso Evolution document to answer the question.
If the information is not in the context, say you don't have that information.

Context:
{context}

Question: {question}

Answer:"""
    )
    
    # Create retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True
    )
    
    return qa_chain


def load_existing_vectorstore(embeddings, collection_name: str = "espresso_evolution"):
    """
    Load an existing vector store from disk
    """
    print("📚 Loading existing vector store...")
    persist_directory = Path("./chroma_db")
    
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory)
    )
    
    return vectorstore


def chat_loop(qa_chain):
    """
    Interactive chat loop
    """
    print("\n" + "="*60)
    print("💬 Espresso Evolution RAG Chat")
    print("="*60)
    print("Type your questions about espresso machine history.")
    print("Type 'exit' or 'quit' to end the chat.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Thanks for chatting! Goodbye!")
                break
            
            print("\n🔍 Searching and generating response...\n")
            
            # Get response
            result = qa_chain.invoke({"query": user_input})
            
            print(f"Assistant: {result['result']}")
            
            # Show sources
            if result.get('source_documents'):
                print("\n📌 Sources:")
                for i, doc in enumerate(result['source_documents'], 1):
                    # Show first 100 chars of source
                    source_text = doc.page_content[:100].replace('\n', ' ')
                    print(f"   [{i}] {source_text}...")
            
            print()
        
        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("   Make sure Ollama is running with Llama-3 model!")
            print()


def main():
    """
    Main entry point
    """
    document_path = "Espresso_Evolution.txt"
    
    # Check if document exists
    if not os.path.exists(document_path):
        print(f"❌ Error: {document_path} not found!")
        return
    
    # Check if vector store already exists
    persist_directory = Path("./chroma_db")
    
    if persist_directory.exists():
        print("Found existing vector store. Loading...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = load_existing_vectorstore(embeddings)
    else:
        print("No existing vector store found. Creating new one...")
        vectorstore, embeddings = setup_rag_system(document_path)
    
    # Initialize LLM
    try:
        llm = initialize_llm()
        
        # Create RAG chain
        qa_chain = create_rag_chain(vectorstore, llm)
        
        # Start chat loop
        chat_loop(qa_chain)
    
    except Exception as e:
        print(f"\n❌ Error initializing LLM: {str(e)}")
        print("\n⚠️  Please make sure:")
        print("   1. Ollama is installed (https://ollama.ai)")
        print("   2. Llama-3 model is pulled: ollama pull llama3")
        print("   3. Ollama service is running: ollama serve")


if __name__ == "__main__":
    main()
