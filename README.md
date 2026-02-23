# Espresso Evolution RAG System

A Retrieval-Augmented Generation (RAG) system that lets you chat with the Espresso Evolution document using Llama-3.

## Prerequisites

- Conda/Miniconda installed
- Ollama installed (https://ollama.ai)
- Llama-3 model downloaded

## Setup Instructions

### 1. Activate the Conda Environment
```bash
conda activate rag_env
```

### 2. Install Ollama and Llama-3

If you haven't already, install Ollama from [ollama.ai](https://ollama.ai)

Then pull the Llama-3 model:
```bash
ollama pull llama3
```

### 3. Start Ollama Service

In a separate terminal, start the Ollama service:
```bash
ollama serve
```

This will run Ollama on `http://localhost:11434` (default port)

### 4. Run the RAG Chat

In your original terminal (with rag_env activated), run:
```bash
python rag_chat.py
```

## How It Works

1. **Document Loading**: Reads `Espresso_Evolution.txt`
2. **Text Chunking**: Splits the document into manageable chunks (500 chars with 100 char overlap)
3. **Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` to create embeddings
4. **Vector Store**: Stores chunks in ChromaDB for fast retrieval
5. **RAG Chain**: Combines document context with Llama-3 for intelligent responses

## Example Questions

- "Who invented the espresso machine?"
- "What is crema and when was it first created?"
- "What is the difference between steam-based and lever-based espresso machines?"
- "What pressure does modern espresso machines use?"
- "Who was Achille Gaggia and what did he contribute?"

## Project Structure

```
rag_test/
├── Espresso_Evolution.txt    # Source document
├── rag_chat.py               # Main RAG application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── chroma_db/                # Vector store (created on first run)
    └── ...                   # ChromaDB files
```

## Troubleshooting

### "Connection refused" error
- Make sure Ollama is running: `ollama serve`
- Verify it's listening on `http://localhost:11434`

### Slow responses on first run
- The first run creates embeddings and the vector store - this takes time
- Subsequent runs will load from the saved vector store and be much faster

### Model not found error
- Make sure you've pulled Llama-3: `ollama pull llama3`
- Check available models: `ollama list`

### Out of memory errors
- Llama-3 requires significant RAM (typically 4-8GB+)
- You can use a lighter model like `ollama pull mistral` if needed
- Then modify the model name in `rag_chat.py` (line ~63)

## Advanced Usage

### Using a Different Model

To use a different model, modify line 63 in `rag_chat.py`:
```python
llm = Ollama(
    model="mistral",  # Change this to your model name
    base_url="http://localhost:11434",
    temperature=0.7
)
```

### Adjusting Retrieval Parameters

Modify the `search_kwargs` in `create_rag_chain()` function:
```python
retriever=vectorstore.as_retriever(search_kwargs={"k": 5})  # Change k for more/fewer results
```

### Clearing the Vector Store

If you want to rebuild the vector store:
```bash
rm -rf chroma_db/
python rag_chat.py
```

## Performance Tips

- First run: ~10-30 seconds (depends on document size and system)
- Subsequent runs: ~1-2 seconds (loads from cache)
- Llama-3 responses: ~5-30 seconds per query (depends on system specs)

## License

This RAG system is provided as-is for educational purposes.
