# RAG (Retrieval-Augmented Generation) System

A multimodal RAG system that can retrieve information from Supabase, index it in Qdrant vector database, and generate responses using a multimodal LLM that can process both text and images.

## Features

- **Supabase Integration**: Retrieves documents from Supabase database
- **Qdrant Vector Database**: Indexes document chunks for semantic search
- **Multimodal LLM**: Uses Google's Gemini model to process both text and images
- **LangGraph Pipeline**: Orchestrates the RAG workflow using LangGraph
- **FastAPI Endpoints**: Provides API access to the RAG system
- **Document Processing**: Chunks and embeds documents for efficient retrieval

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Set up environment variables by creating a `.env` file:

```
# Google Gemini API (comma-separated list of API keys for rotation)
GOOGLE_API_KEYS_CSV=key1,key2,key3

# Qdrant Vector Database
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
# For local Qdrant (no API key needed)
# QDRANT_URL=http://localhost:6333

# Supabase Configuration
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
SUPABASE_PAGE_TABLE=your_table_name

# RAG API Configuration
RAG_API_HOST=0.0.0.0
RAG_API_PORT=8000
```

## Usage

### Indexing Documents

Before using the RAG system, you need to index documents from Supabase:

```python
from RAG.data.document_processor import DocumentProcessor

processor = DocumentProcessor()
processor.process_documents()
```

### Querying the RAG System

```python
from RAG.models.rag_pipeline import RAGPipeline

# Initialize the RAG pipeline
pipeline = RAGPipeline()

# Text-only query
result = pipeline.run(query="What is retrieval-augmented generation?")
print(result["response"])

# Multimodal query with image
result = pipeline.run(
    query="What is shown in this image?",
    image_path="path/to/image.jpg"  # or URL
)
print(result["response"])
```

### Running the API Server

```bash
python -m RAG.api.app
```

This will start the FastAPI server at `http://0.0.0.0:8000`.

### API Endpoints

- `POST /query`: Query the RAG system with text and optional image
- `POST /index`: Index all documents from Supabase
- `POST /index/since/{timestamp}`: Index documents since a specific timestamp
- `GET /collection/info`: Get information about the Qdrant collection
- `DELETE /collection`: Delete the Qdrant collection

### Running the Demo

```bash
# Index documents and run interactive demo
python -m RAG.demo --index

# Query with text only
python -m RAG.demo --query "What is retrieval-augmented generation?"

# Query with text and image
python -m RAG.demo --query "What is shown in this image?" --image "path/to/image.jpg"

# Run interactive demo
python -m RAG.demo
```

## Architecture

The RAG system consists of the following components:

1. **Data Layer**:
   - `supabase_connector.py`: Retrieves documents from Supabase
   - `qdrant_connector.py`: Manages the Qdrant vector database
   - `document_processor.py`: Processes documents for indexing

2. **Model Layer**:
   - `llm_interface.py`: Interfaces with OpenAI's multimodal LLM
   - `rag_pipeline.py`: Orchestrates the RAG workflow using LangGraph

3. **API Layer**:
   - `app.py`: Provides FastAPI endpoints for the RAG system

4. **Utilities**:
   - `image_utils.py`: Utilities for image processing
   - `env_utils.py`: Utilities for environment variable handling

## Customization

### Using a Different Vector Database

To use a different vector database, create a new connector class similar to `QdrantConnector` and update the `DocumentProcessor` class to use it.

### Using a Different LLM

To use a different LLM, update the `MultimodalLLM` class to interface with your preferred model.

### Customizing the RAG Pipeline

The RAG pipeline is implemented using LangGraph, which makes it easy to customize the workflow. You can modify the `RAGPipeline` class to add or remove steps, or to change the behavior of existing steps.

## License

[Your License Here]
