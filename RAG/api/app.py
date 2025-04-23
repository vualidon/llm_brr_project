"""
FastAPI application for the RAG system.
Provides endpoints for querying the RAG system and managing the vector database.
"""

import os
import shutil
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime

from RAG.utils.env_utils import get_api_config, load_env_vars
from RAG.utils.image_utils import save_image, is_valid_image
from RAG.models.rag_pipeline import RAGPipeline
from RAG.data.document_processor import DocumentProcessor
from RAG.data.supabase_connector import SupabaseConnector
from RAG.data.qdrant_connector import QdrantConnector

# Create upload directory
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="RAG API",
    description="API for Retrieval-Augmented Generation with multimodal capabilities",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize components
try:
    env_vars = load_env_vars()
    rag_pipeline = RAGPipeline()
    document_processor = DocumentProcessor()
    supabase = SupabaseConnector()
    qdrant = QdrantConnector()
except Exception as e:
    print(f"Error initializing components: {e}")
    # We'll continue and handle errors in the endpoints

@app.get("/")
async def root():
    """Root endpoint that returns a welcome message."""
    return {"message": "Welcome to the RAG API"}

@app.post("/query")
async def query(
    query: str = Form(...),
    image: Optional[UploadFile] = File(None)
):
    """
    Query the RAG system with text and optional image.
    
    Args:
        query: Text query.
        image: Optional image file.
        
    Returns:
        RAG response.
    """
    try:
        image_path = None
        
        # Process image if provided
        if image:
            # Validate image
            contents = await image.read()
            if not is_valid_image(contents):
                raise HTTPException(status_code=400, detail="Invalid image file")
            
            # Save image
            image_path = os.path.join(UPLOAD_DIR, f"{datetime.now().timestamp()}_{image.filename}")
            with open(image_path, "wb") as f:
                f.write(contents)
        
        # Run RAG pipeline
        result = rag_pipeline.run(query=query, image_path=image_path)
        
        # Clean up image file
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
        
        # Return response
        return {
            "query": query,
            "response": result["response"],
            "num_documents": len(result["retrieved_documents"]),
            "error": result["error"]
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error processing query: {str(e)}"}
        )

@app.post("/index")
async def index_documents(background_tasks: BackgroundTasks):
    """
    Index all documents from Supabase in the vector database.
    This is a long-running operation, so it runs in the background.
    
    Returns:
        Status message.
    """
    try:
        # Start indexing in the background
        background_tasks.add_task(document_processor.process_documents)
        
        return {"message": "Indexing started in the background"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error starting indexing: {str(e)}"}
        )

@app.post("/index/since/{timestamp}")
async def index_documents_since(
    timestamp: str,
    background_tasks: BackgroundTasks
):
    """
    Index documents from Supabase created or updated since the given timestamp.
    
    Args:
        timestamp: ISO format timestamp string.
        
    Returns:
        Status message.
    """
    try:
        # Validate timestamp format
        try:
            datetime.fromisoformat(timestamp)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid timestamp format. Use ISO format (YYYY-MM-DDTHH:MM:SS)")
        
        # Start indexing in the background
        background_tasks.add_task(document_processor.process_new_documents_since, timestamp)
        
        return {"message": f"Indexing documents since {timestamp} started in the background"}
    except HTTPException as e:
        raise e
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error starting indexing: {str(e)}"}
        )

@app.get("/collection/info")
async def get_collection_info():
    """
    Get information about the Qdrant collection.
    
    Returns:
        Collection information.
    """
    try:
        info = qdrant.get_collection_info()
        return info
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error getting collection info: {str(e)}"}
        )

@app.delete("/collection")
async def delete_collection():
    """
    Delete the Qdrant collection.
    
    Returns:
        Status message.
    """
    try:
        qdrant.delete_collection()
        return {"message": "Collection deleted successfully"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error deleting collection: {str(e)}"}
        )

def start():
    """Start the FastAPI server."""
    config = get_api_config()
    uvicorn.run(
        "RAG.api.app:app",
        host=config["host"],
        port=config["port"],
        reload=config["debug"]
    )

if __name__ == "__main__":
    start()
