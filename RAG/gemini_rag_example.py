"""
Example script demonstrating how to use the Gemini-based RAG system.
"""

import os
from dotenv import load_dotenv
import logging
import argparse
from datetime import datetime

from RAG.models.gemini_interface import GeminiLLM
from RAG.models.gemini_embeddings import GeminiEmbeddings
from RAG.data.document_processor import DocumentProcessor
from RAG.models.rag_pipeline import RAGPipeline
from RAG.data.qdrant_connector import QdrantConnector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_environment():
    """
    Check if the required environment variables are set.
    """
    load_dotenv()
    
    required_vars = ["GOOGLE_API_KEYS_CSV", "QDRANT_URL", "SUPABASE_URL", "SUPABASE_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file.")
        return False
    
    return True

def test_gemini_llm():
    """
    Test the Gemini LLM interface.
    """
    print("\n=== Testing Gemini LLM ===")
    try:
        llm = GeminiLLM()
        
        # Test with text only
        query = "What is retrieval-augmented generation?"
        print(f"Query: {query}")
        
        response = llm.generate_response(query, max_tokens=100)
        print(f"Response: {response}")
        
        return True
    except Exception as e:
        print(f"Error testing Gemini LLM: {e}")
        return False

def test_gemini_embeddings():
    """
    Test the Gemini embeddings.
    """
    print("\n=== Testing Gemini Embeddings ===")
    try:
        embeddings = GeminiEmbeddings()
        
        # Test with a single text
        query = "What is retrieval-augmented generation?"
        print(f"Generating embedding for: {query}")
        
        query_embedding = embeddings.embed_query(query)
        print(f"Embedding dimension: {len(query_embedding)}")
        print(f"First few values: {query_embedding[:5]}")
        
        return True
    except Exception as e:
        print(f"Error testing Gemini embeddings: {e}")
        return False

def test_rag_pipeline():
    """
    Test the RAG pipeline.
    """
    print("\n=== Testing RAG Pipeline ===")
    try:
        # Initialize the RAG pipeline
        pipeline = RAGPipeline(
            llm_model="gemini-2.0-flash",
            num_results=3
        )
        
        # Test with a query
        query = "What is retrieval-augmented generation?"
        print(f"Query: {query}")
        
        result = pipeline.run(query=query)
        
        print(f"Retrieved {len(result['retrieved_documents'])} documents")
        print(f"Response: {result['response']}")
        
        return True
    except Exception as e:
        print(f"Error testing RAG pipeline: {e}")
        return False

def index_documents():
    """
    Index documents from Supabase in Qdrant.
    """
    print("\n=== Indexing Documents ===")
    try:
        processor = DocumentProcessor()
        
        start_time = datetime.now()
        print(f"Started indexing at {start_time}")
        
        num_chunks = processor.process_documents()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"Indexed {num_chunks} document chunks in {duration}")
        
        return True
    except Exception as e:
        print(f"Error indexing documents: {e}")
        return False

def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description="Gemini RAG Example")
    parser.add_argument("--test-llm", action="store_true", help="Test the Gemini LLM")
    parser.add_argument("--test-embeddings", action="store_true", help="Test the Gemini embeddings")
    parser.add_argument("--test-pipeline", action="store_true", help="Test the RAG pipeline")
    parser.add_argument("--index", action="store_true", help="Index documents from Supabase")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    # Check environment variables
    if not check_environment():
        return
    
    # Run tests
    if args.all or args.test_llm:
        test_gemini_llm()
    
    if args.all or args.test_embeddings:
        test_gemini_embeddings()
    
    if args.all or args.test_pipeline:
        test_rag_pipeline()
    
    if args.all or args.index:
        index_documents()
    
    # If no arguments provided, show help
    if not (args.test_llm or args.test_embeddings or args.test_pipeline or args.index or args.all):
        parser.print_help()

if __name__ == "__main__":
    main()
