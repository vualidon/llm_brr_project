"""
Utility functions for environment variable handling in the RAG system.
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

def load_env_vars() -> Dict[str, str]:
    """
    Load environment variables from .env file.
    
    Returns:
        Dictionary of environment variables.
    """
    load_dotenv()
    
    # Required environment variables for RAG system
    required_vars = [
        "OPENAI_API_KEY",
        "QDRANT_URL",
        "SUPABASE_URL",
        "SUPABASE_KEY",
        "SUPABASE_PAGE_TABLE"
    ]
    
    # Optional environment variables with defaults
    optional_vars = {
        "QDRANT_API_KEY": None,
        "RAG_API_HOST": "0.0.0.0",
        "RAG_API_PORT": "8000"
    }
    
    # Check required variables
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Collect all variables
    env_vars = {}
    for var in required_vars:
        env_vars[var] = os.getenv(var)
    
    for var, default in optional_vars.items():
        env_vars[var] = os.getenv(var, default)
    
    return env_vars

def get_api_config() -> Dict[str, Any]:
    """
    Get API configuration from environment variables.
    
    Returns:
        Dictionary with API configuration.
    """
    load_dotenv()
    
    return {
        "host": os.getenv("RAG_API_HOST", "0.0.0.0"),
        "port": int(os.getenv("RAG_API_PORT", "8000")),
        "debug": os.getenv("RAG_API_DEBUG", "False").lower() == "true"
    }

def get_qdrant_config() -> Dict[str, Any]:
    """
    Get Qdrant configuration from environment variables.
    
    Returns:
        Dictionary with Qdrant configuration.
    """
    load_dotenv()
    
    config = {
        "url": os.getenv("QDRANT_URL"),
        "collection_name": os.getenv("QDRANT_COLLECTION", "documents")
    }
    
    # Add API key if provided
    api_key = os.getenv("QDRANT_API_KEY")
    if api_key:
        config["api_key"] = api_key
    
    return config

def get_openai_config() -> Dict[str, str]:
    """
    Get OpenAI configuration from environment variables.
    
    Returns:
        Dictionary with OpenAI configuration.
    """
    load_dotenv()
    
    return {
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        "llm_model": os.getenv("OPENAI_LLM_MODEL", "gpt-4-vision-preview")
    }

def get_supabase_config() -> Dict[str, str]:
    """
    Get Supabase configuration from environment variables.
    
    Returns:
        Dictionary with Supabase configuration.
    """
    load_dotenv()
    
    return {
        "url": os.getenv("SUPABASE_URL", ""),
        "key": os.getenv("SUPABASE_KEY", ""),
        "table": os.getenv("SUPABASE_PAGE_TABLE", "")
    }
