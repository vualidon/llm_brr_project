"""
Qdrant vector database connector for RAG system.
Handles document indexing and vector search operations.
"""

import os
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import uuid

class QdrantConnector:
    """
    Connector for Qdrant vector database operations.
    """
    
    def __init__(self, collection_name: str = "documents"):
        """
        Initialize the Qdrant connector with credentials from environment variables.
        
        Args:
            collection_name: Name of the Qdrant collection to use.
        """
        load_dotenv()
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = collection_name
        
        if not self.qdrant_url:
            raise ValueError("Missing QDRANT_URL in environment variables.")
        
        # Initialize Qdrant client
        if self.qdrant_api_key:
            self.client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
        else:
            self.client = QdrantClient(url=self.qdrant_url)
        
        print(f"Qdrant client initialized with URL: {self.qdrant_url}")
    
    def create_collection(self, vector_size: int = 1536):
        """
        Create a new collection in Qdrant if it doesn't exist.
        
        Args:
            vector_size: Dimension of the vector embeddings.
        """
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print(f"Created new Qdrant collection: {self.collection_name}")
        else:
            print(f"Collection {self.collection_name} already exists.")
    
    def upsert_documents(self, documents: List[Dict[str, Any]], vectors: List[List[float]]):
        """
        Insert or update documents in the Qdrant collection.
        
        Args:
            documents: List of document dictionaries.
            vectors: List of embedding vectors corresponding to the documents.
        """
        if len(documents) != len(vectors):
            raise ValueError("Number of documents and vectors must match.")
        
        points = []
        for i, (doc, vector) in enumerate(zip(documents, vectors)):
            # Generate a deterministic ID based on document content to avoid duplicates
            doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, doc.get('url', '') or str(i)))
            
            # Prepare payload with document metadata
            payload = {
                "id": doc_id,
                "title": doc.get("title", ""),
                "url": doc.get("url", ""),
                "content": doc.get("content", ""),
                "image_link": doc.get("image_link", ""),
                "crawled_at": doc.get("crawled_at", "")
            }
            
            # Create point
            point = PointStruct(
                id=doc_id,
                vector=vector,
                payload=payload
            )
            points.append(point)
        
        # Upsert points to collection
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"Upserted {len(points)} documents to Qdrant collection {self.collection_name}")
    
    def search(self, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents in the Qdrant collection.
        
        Args:
            query_vector: Embedding vector of the query.
            limit: Maximum number of results to return.
            
        Returns:
            List of document dictionaries with similarity scores.
        """
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )
        
        results = []
        for scored_point in search_result:
            # Extract payload and add score
            doc = scored_point.payload
            doc["score"] = scored_point.score
            results.append(doc)
        
        return results
    
    def delete_collection(self):
        """
        Delete the Qdrant collection.
        """
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            print(f"Deleted Qdrant collection: {self.collection_name}")
        except Exception as e:
            print(f"Error deleting collection {self.collection_name}: {e}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            Dictionary with collection information.
        """
        try:
            info = self.client.get_collection(collection_name=self.collection_name)
            return {
                "name": info.name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return {}


if __name__ == "__main__":
    # Simple test with dummy data
    try:
        connector = QdrantConnector(collection_name="test_collection")
        
        # Create collection
        connector.create_collection(vector_size=4)  # Small size for testing
        
        # Test documents and vectors
        test_docs = [
            {"title": "Test Document 1", "url": "http://example.com/1", "content": "This is a test document."},
            {"title": "Test Document 2", "url": "http://example.com/2", "content": "Another test document."}
        ]
        test_vectors = [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8]
        ]
        
        # Upsert documents
        connector.upsert_documents(test_docs, test_vectors)
        
        # Get collection info
        info = connector.get_collection_info()
        print(f"Collection info: {info}")
        
        # Search
        search_results = connector.search([0.1, 0.2, 0.3, 0.4], limit=2)
        print(f"Search results: {search_results}")
        
        # Clean up
        connector.delete_collection()
        
    except Exception as e:
        print(f"Error in Qdrant connector test: {e}")
