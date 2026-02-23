#!/bin/bash
# Quick start script for Streamlit RAG Application

echo "🚀 Streamlit RAG Application - Quick Start"
echo "=========================================="
echo ""

# Check if rag_env exists
if ! conda env list | grep -q "rag_env"; then
    echo "❌ Conda environment 'rag_env' not found!"
    echo "Please run: conda create -n rag_env python=3.11 -y"
    exit 1
fi

# Activate environment
source activate rag_env 2>/dev/null || conda activate rag_env

# Check if dependencies are installed
python -c "import streamlit, langchain, chromadb" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Dependencies not found. Installing..."
    pip install -r requirements.txt
fi

# Check if Ollama is running
echo ""
echo "⏳ Checking Ollama connection..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama is running"
    
    # Check if llama3 model exists
    if curl -s http://localhost:11434/api/tags | grep -q "llama3"; then
        echo "✅ Llama-3 model is available"
    else
        echo "⚠️  Llama-3 model not found. Run: ollama pull llama3"
    fi
else
    echo "⚠️  Ollama is not running!"
    echo "   Please start it in another terminal: ollama serve"
    echo ""
    echo "Continue anyway? The app will show connection errors until Ollama is running."
fi

echo ""
echo "🌐 Starting Streamlit Application..."
echo "📍 App will open at: http://localhost:8501"
echo ""
streamlit run app.py
