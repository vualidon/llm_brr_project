"""
Supabase connector for RAG system.
Handles retrieving data from Supabase for indexing in the vector database.
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from supabase import create_client, Client, ClientOptions

class SupabaseConnector:
    """
    Connector for retrieving data from Supabase for the RAG system.
    """
    
    def __init__(self):
        """
        Initialize the Supabase connector with credentials from environment variables.
        """
        load_dotenv()
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        self.supabase_table = os.getenv("SUPABASE_PAGE_TABLE")
        
        if not all([self.supabase_url, self.supabase_key, self.supabase_table]):
            raise ValueError("Missing Supabase configuration. Please check your .env file.")
        
        try:
            options = ClientOptions(schema="public")
            self.supabase_client: Client = create_client(self.supabase_url, self.supabase_key, options=options)
            print("Supabase client initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize Supabase client: {e}")
            raise
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        Retrieve all documents from the Supabase table.
        
        Returns:
            List of document dictionaries.
        """
        try:
            response = self.supabase_client.table(self.supabase_table).select("*").execute()
            if hasattr(response, 'data'):
                return response.data
            return []
        except Exception as e:
            print(f"Error retrieving documents from Supabase: {e}")
            return []
    
    def get_documents_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve documents by their IDs.
        
        Args:
            ids: List of document IDs to retrieve.
            
        Returns:
            List of document dictionaries.
        """
        if not ids:
            return []
        
        try:
            response = self.supabase_client.table(self.supabase_table).select("*").in_("id", ids).execute()
            if hasattr(response, 'data'):
                return response.data
            return []
        except Exception as e:
            print(f"Error retrieving documents by IDs from Supabase: {e}")
            return []
    
    def get_documents_since(self, timestamp: str) -> List[Dict[str, Any]]:
        """
        Retrieve documents created or updated since the given timestamp.
        
        Args:
            timestamp: ISO format timestamp string.
            
        Returns:
            List of document dictionaries.
        """
        try:
            response = self.supabase_client.table(self.supabase_table).select("*").gte("crawled_at", timestamp).execute()
            if hasattr(response, 'data'):
                return response.data
            return []
        except Exception as e:
            print(f"Error retrieving documents since {timestamp} from Supabase: {e}")
            return []
    
    def search_documents(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for documents containing the query string in title or content.
        
        Args:
            query: Search query string.
            
        Returns:
            List of document dictionaries.
        """
        try:
            # This is a simple implementation. For more complex search,
            # consider using Supabase's full-text search capabilities.
            response = self.supabase_client.table(self.supabase_table).select("*").or_(
                f"title.ilike.%{query}%,content.ilike.%{query}%"
            ).execute()
            
            if hasattr(response, 'data'):
                return response.data
            return []
        except Exception as e:
            print(f"Error searching documents in Supabase: {e}")
            return []


if __name__ == "__main__":
    # Simple test
    try:
        connector = SupabaseConnector()
        documents = connector.get_all_documents()
        print(f"Retrieved {len(documents)} documents from Supabase")
        
        if documents:
            print("\nSample document:")
            sample = documents[0]
            print(f"Title: {sample.get('title', 'N/A')}")
            print(f"URL: {sample.get('url', 'N/A')}")
            print(f"Image: {sample.get('image_link', 'N/A')}")
            content = sample.get('content', '')
            print(f"Content preview: {content[:100]}..." if content else "No content")
    except Exception as e:
        print(f"Error in Supabase connector test: {e}")
