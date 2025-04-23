"""
Document processor for RAG system.
Handles document chunking, embedding generation, and indexing.
"""

import os
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
import time
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from RAG.data.supabase_connector import SupabaseConnector
from RAG.data.qdrant_connector import QdrantConnector
from RAG.models.gemini_embeddings import GeminiEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DocumentProcessor:
    """
    Processes documents for RAG system by chunking, embedding, and indexing.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "text-embedding-3-small",
        qdrant_collection: str = "documents"
    ):
        """
        Initialize the document processor.

        Args:
            chunk_size: Size of document chunks.
            chunk_overlap: Overlap between chunks.
            embedding_model: Name of the OpenAI embedding model to use.
            qdrant_collection: Name of the Qdrant collection to use.
        """
        load_dotenv()

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        # Initialize embedding model
        self.embeddings = GeminiEmbeddings(
            model_name="embedding-001",
            task_type="RETRIEVAL_DOCUMENT"
        )

        # Initialize connectors
        self.supabase = SupabaseConnector()
        self.qdrant = QdrantConnector(collection_name=qdrant_collection)

        # Create Qdrant collection if it doesn't exist
        # Using 1536 as the default dimension for OpenAI embeddings
        self.qdrant.create_collection(vector_size=1536)

        print(f"Document processor initialized with chunk size {chunk_size}, overlap {chunk_overlap}")

    def _create_langchain_documents(self, raw_docs: List[Dict[str, Any]]) -> List[Document]:
        """
        Convert raw document dictionaries to LangChain Document objects.

        Args:
            raw_docs: List of raw document dictionaries from Supabase.

        Returns:
            List of LangChain Document objects.
        """
        documents = []
        for doc in raw_docs:
            # Extract content and metadata
            content = doc.get("content", "")
            if not content:
                continue

            metadata = {
                "id": doc.get("id", ""),
                "title": doc.get("title", ""),
                "url": doc.get("url", ""),
                "image_link": doc.get("image_link", ""),
                "crawled_at": doc.get("crawled_at", "")
            }

            # Create LangChain Document
            documents.append(Document(page_content=content, metadata=metadata))

        return documents

    def process_documents(self, raw_docs: Optional[List[Dict[str, Any]]] = None) -> int:
        """
        Process documents from Supabase, chunk them, generate embeddings, and index in Qdrant.

        Args:
            raw_docs: Optional list of raw documents. If None, fetches from Supabase.

        Returns:
            Number of chunks indexed.
        """
        # Get documents from Supabase if not provided
        if raw_docs is None:
            raw_docs = self.supabase.get_all_documents()
            print(f"Retrieved {len(raw_docs)} documents from Supabase")

        if not raw_docs:
            print("No documents to process")
            return 0

        # Convert to LangChain Documents
        langchain_docs = self._create_langchain_documents(raw_docs)
        print(f"Created {len(langchain_docs)} LangChain documents")

        # Split documents into chunks
        chunks = self.text_splitter.split_documents(langchain_docs)
        print(f"Split into {len(chunks)} chunks")

        # Process in batches to avoid rate limits
        batch_size = 100
        total_chunks = len(chunks)
        processed_chunks = 0

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i+batch_size]

            # Extract text content for embedding
            texts = [chunk.page_content for chunk in batch]

            # Generate embeddings
            print(f"Generating embeddings for batch {i//batch_size + 1}/{(total_chunks-1)//batch_size + 1}")
            embeddings = self.embeddings.embed_documents(texts)

            # Prepare documents for Qdrant
            qdrant_docs = []
            for j, chunk in enumerate(batch):
                doc = {
                    "id": chunk.metadata.get("id", f"chunk_{i+j}"),
                    "title": chunk.metadata.get("title", ""),
                    "url": chunk.metadata.get("url", ""),
                    "content": chunk.page_content,
                    "image_link": chunk.metadata.get("image_link", ""),
                    "crawled_at": chunk.metadata.get("crawled_at", ""),
                    "chunk_index": i + j
                }
                qdrant_docs.append(doc)

            # Index in Qdrant
            self.qdrant.upsert_documents(qdrant_docs, embeddings)
            processed_chunks += len(batch)

            print(f"Indexed {processed_chunks}/{total_chunks} chunks")

            # Sleep to avoid rate limits
            if i + batch_size < total_chunks:
                time.sleep(1)

        return processed_chunks

    def process_new_documents_since(self, timestamp: str) -> int:
        """
        Process documents created or updated since the given timestamp.

        Args:
            timestamp: ISO format timestamp string.

        Returns:
            Number of chunks indexed.
        """
        # Get documents from Supabase
        raw_docs = self.supabase.get_documents_since(timestamp)
        print(f"Retrieved {len(raw_docs)} documents from Supabase since {timestamp}")

        # Process documents
        return self.process_documents(raw_docs)


if __name__ == "__main__":
    # Simple test
    try:
        processor = DocumentProcessor(
            chunk_size=500,  # Smaller chunks for testing
            chunk_overlap=50
        )

        # Process all documents
        num_chunks = processor.process_documents()
        print(f"Processed and indexed {num_chunks} chunks")

    except Exception as e:
        print(f"Error in document processor test: {e}")
