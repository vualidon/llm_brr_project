"""
Gemini embeddings for the RAG system.
Provides a LangChain-compatible embeddings class for Google's Gemini embeddings.
"""

import os
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
import time
import logging
from google import genai
from langchain_core.embeddings import Embeddings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GeminiEmbeddings(Embeddings):
    """
    LangChain-compatible embeddings class for Google's Gemini embeddings.
    """
    
    def __init__(
        self,
        model_name: str = "embedding-001",
        task_type: str = "RETRIEVAL_DOCUMENT",
        title: Optional[str] = None
    ):
        """
        Initialize the Gemini embeddings.
        
        Args:
            model_name: Name of the embedding model to use.
            task_type: Type of task for the embeddings.
            title: Optional title for the document.
        """
        load_dotenv()
        
        # Load comma-separated keys for rotation
        keys_csv = os.getenv("GOOGLE_API_KEYS_CSV")
        if not keys_csv:
            raise ValueError("GOOGLE_API_KEYS_CSV environment variable not set or empty.")

        self.api_keys = [key.strip() for key in keys_csv.split(',') if key.strip()]
        if not self.api_keys:
            raise ValueError("No valid API keys found in GOOGLE_API_KEYS_CSV.")

        self.current_key_index = 0
        self.client = None  # Will be initialized by _initialize_client
        self.model_name = model_name
        self.task_type = task_type
        self.title = title

        logging.info(f"Loaded {len(self.api_keys)} API keys for Gemini embeddings.")

        # Initialize the client with the first key
        self._initialize_client(self.current_key_index)
    
    def _initialize_client(self, key_index: int):
        """
        Initialize or re-initialize the Gemini client with the specified key.
        
        Args:
            key_index: Index of the API key to use.
        """
        if not 0 <= key_index < len(self.api_keys):
            raise IndexError("Invalid API key index provided.")

        api_key = self.api_keys[key_index]
        try:
            # Create a new client instance
            self.client = genai.Client(api_key=api_key)
            logging.info(f"Google Generative AI client initialized successfully with key index {key_index} for embeddings.")
            self.current_key_index = key_index  # Update the current index tracker
        except Exception as e:
            logging.error(f"Failed to initialize Google Generative AI client with key index {key_index}: {e}")
            raise RuntimeError(f"Failed to initialize client with key index {key_index}") from e
    
    def _switch_to_next_key(self) -> bool:
        """
        Switch to the next available API key and re-initialize the client.
        
        Returns:
            True if successfully switched, False otherwise.
        """
        initial_index = self.current_key_index
        next_index = (self.current_key_index + 1) % len(self.api_keys)

        logging.warning(f"Attempting to switch API key from index {initial_index} to {next_index}.")

        # Try initializing with the next key
        try:
            self._initialize_client(next_index)
            logging.info(f"Successfully switched API key to index {next_index}.")
            return True  # Successfully switched
        except Exception as e:
            logging.error(f"Failed to switch to API key index {next_index}: {e}. Staying with index {initial_index}.")
            return False  # Failed to switch
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            texts: List of document texts to embed.
            
        Returns:
            List of embedding vectors.
        """
        embeddings = []
        
        for text in texts:
            # Try to get embedding with retry logic
            embedding = self._get_embedding_with_retry(text)
            embeddings.append(embedding)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a query.
        
        Args:
            text: Query text to embed.
            
        Returns:
            Embedding vector.
        """
        # For Gemini, we use the same embedding method for both documents and queries
        return self._get_embedding_with_retry(text)
    
    def _get_embedding_with_retry(self, text: str) -> List[float]:
        """
        Get embedding for a text with retry logic.
        
        Args:
            text: Text to embed.
            
        Returns:
            Embedding vector.
        """
        # Retry loop - try each key once
        start_key_index = self.current_key_index
        for attempt in range(len(self.api_keys)):
            current_attempt_index = (start_key_index + attempt) % len(self.api_keys)
            
            # Ensure the client is initialized with the correct key for this attempt
            if attempt > 0 or self.client is None:
                if not self._switch_to_next_key():
                    logging.error("Failed to initialize next key. Cannot proceed.")
                    # Return a zero vector as fallback (not ideal but prevents crashing)
                    return [0.0] * 768  # Default dimension for Gemini embeddings
            
            try:
                # Get embedding from Gemini
                result = self.client.embeddings.get_embedding(
                    model=f"models/{self.model_name}",
                    text=text,
                    task_type=self.task_type,
                    title=self.title
                )
                
                # Extract and return the embedding values
                return result.values
                
            except Exception as e:
                if "rate limit" in str(e).lower() or "quota exceeded" in str(e).lower():
                    logging.warning(f"Rate limit or quota exceeded for key index {self.current_key_index}: {e}. Attempting to switch key.")
                    time.sleep(1)  # Brief pause before trying the next key
                    continue
                elif attempt == len(self.api_keys) - 1:
                    # Last attempt failed
                    logging.error(f"All {len(self.api_keys)} API keys failed to generate embedding. Error: {e}")
                    # Return a zero vector as fallback (not ideal but prevents crashing)
                    return [0.0] * 768  # Default dimension for Gemini embeddings
                else:
                    # Try the next key
                    logging.warning(f"Error with key index {self.current_key_index}: {e}. Trying next key.")
                    time.sleep(1)
                    continue
        
        # If we get here, all keys failed
        logging.error("Failed to generate embedding after trying all available API keys.")
        return [0.0] * 768  # Default dimension for Gemini embeddings


if __name__ == "__main__":
    # Simple test
    try:
        embeddings = GeminiEmbeddings()
        
        # Test with a single text
        query_embedding = embeddings.embed_query("What is retrieval-augmented generation?")
        print(f"Query embedding dimension: {len(query_embedding)}")
        print(f"First few values: {query_embedding[:5]}")
        
        # Test with multiple texts
        docs = [
            "Retrieval-augmented generation (RAG) is a technique that combines retrieval and generation.",
            "RAG systems retrieve relevant documents and use them to generate more accurate responses."
        ]
        doc_embeddings = embeddings.embed_documents(docs)
        print(f"Generated {len(doc_embeddings)} document embeddings")
        print(f"Document embedding dimension: {len(doc_embeddings[0])}")
        
    except Exception as e:
        print(f"Error in Gemini embeddings test: {e}")
