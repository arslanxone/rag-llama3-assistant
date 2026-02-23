#!/bin/bash
# Quick start script for RAG system

echo "🚀 Espresso Evolution RAG Chat - Quick Start"
echo "==========================================="
echo ""

# Check if rag_env exists
if ! conda env list | grep -q "rag_env"; then
    echo "❌ Conda environment 'rag_env' not found!"
    echo "Please run: conda create -n rag_env python=3.11 -y"
    exit 1
fi

# Activate environment
source activate rag_env

# Check if dependencies are installed
python -c "import langchain, chromadb, sentence_transformers, ollama" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Dependencies not found. Installing..."
    pip install -r requirements.txt
fi

# Check if Espresso_Evolution.txt exists
if [ ! -f "Espresso_Evolution.txt" ]; then
    echo "❌ Espresso_Evolution.txt not found in current directory!"
    exit 1
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
    echo "❌ Ollama is not running!"
    echo "   Please start it in another terminal: ollama serve"
    exit 1
fi

echo ""
echo "🤖 Starting RAG Chat System..."
python rag_chat.py
