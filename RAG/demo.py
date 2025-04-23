"""
Demo script for the RAG system.
Shows how to use the RAG pipeline with text and image queries.
"""

import os
import argparse
from typing import Optional
from dotenv import load_dotenv

from RAG.models.rag_pipeline import RAGPipeline
from RAG.data.document_processor import DocumentProcessor
from RAG.data.supabase_connector import SupabaseConnector
from RAG.data.qdrant_connector import QdrantConnector

def setup_rag_system():
    """
    Set up the RAG system components.

    Returns:
        Tuple of (RAGPipeline, DocumentProcessor, SupabaseConnector, QdrantConnector)
    """
    load_dotenv()

    # Check for required environment variables
    required_vars = ["GOOGLE_API_KEYS_CSV", "QDRANT_URL", "SUPABASE_URL", "SUPABASE_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file.")
        exit(1)

    # Initialize components
    try:
        rag_pipeline = RAGPipeline()
        document_processor = DocumentProcessor()
        supabase = SupabaseConnector()
        qdrant = QdrantConnector()

        return rag_pipeline, document_processor, supabase, qdrant
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        exit(1)

def index_documents(document_processor):
    """
    Index documents from Supabase in the vector database.

    Args:
        document_processor: DocumentProcessor instance.
    """
    print("Indexing documents from Supabase...")
    try:
        num_chunks = document_processor.process_documents()
        print(f"Successfully indexed {num_chunks} document chunks.")
    except Exception as e:
        print(f"Error indexing documents: {e}")

def query_rag(rag_pipeline, query: str, image_path: Optional[str] = None):
    """
    Query the RAG system with text and optional image.

    Args:
        rag_pipeline: RAGPipeline instance.
        query: Text query.
        image_path: Optional path to an image file.
    """
    print(f"\nQuery: {query}")
    if image_path:
        print(f"Image: {image_path}")

    try:
        result = rag_pipeline.run(query=query, image_path=image_path)

        print("\nRetrieved Documents:")
        for i, doc in enumerate(result["retrieved_documents"][:3]):  # Show top 3
            print(f"  {i+1}. {doc.get('title', 'Untitled')} (Score: {doc.get('score', 0):.4f})")

        print("\nResponse:")
        print(result["response"])
    except Exception as e:
        print(f"Error querying RAG system: {e}")

def main():
    """Main function for the demo script."""
    parser = argparse.ArgumentParser(description="RAG System Demo")
    parser.add_argument("--index", action="store_true", help="Index documents from Supabase")
    parser.add_argument("--query", type=str, help="Text query for the RAG system")
    parser.add_argument("--image", type=str, help="Path to an image file for multimodal queries")

    args = parser.parse_args()

    # Set up RAG system
    rag_pipeline, document_processor, supabase, qdrant = setup_rag_system()

    # Index documents if requested
    if args.index:
        index_documents(document_processor)

    # Query RAG system if requested
    if args.query:
        query_rag(rag_pipeline, args.query, args.image)

    # If no arguments provided, run interactive demo
    if not (args.index or args.query):
        print("RAG System Interactive Demo")
        print("==========================")

        # Check if collection exists and has documents
        collection_info = qdrant.get_collection_info()
        if not collection_info or collection_info.get("points_count", 0) == 0:
            print("No documents found in the vector database.")
            index_now = input("Would you like to index documents from Supabase now? (y/n): ")
            if index_now.lower() == "y":
                index_documents(document_processor)

        # Interactive query loop
        while True:
            print("\nEnter a query (or 'exit' to quit):")
            query = input("> ")

            if query.lower() in ["exit", "quit", "q"]:
                break

            use_image = input("Include an image? (y/n): ")
            image_path = None

            if use_image.lower() == "y":
                image_path = input("Enter image path or URL: ")

            query_rag(rag_pipeline, query, image_path)

if __name__ == "__main__":
    main()
