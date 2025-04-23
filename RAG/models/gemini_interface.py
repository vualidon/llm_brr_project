"""
Multimodal LLM interface for RAG system using Google's Gemini model.
Handles interactions with Google's multimodal models.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
import requests
import mimetypes
import time
from google import genai
from google.genai import types

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure the mimetypes module is initialized
mimetypes.init()

class GeminiLLM:
    """
    Interface for Google's Gemini multimodal language models that can process text and images.
    """
    
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        """
        Initialize the Gemini LLM interface.
        
        Args:
            model_name: Name of the Gemini model to use.
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

        logging.info(f"Loaded {len(self.api_keys)} API keys for Gemini.")

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
            logging.info(f"Google Generative AI client initialized successfully with key index {key_index} for model '{self.model_name}'.")
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
    
    def _prepare_image_part(self, image_path_or_url: str) -> Optional[types.Part]:
        """
        Download an image and prepare it as a types.Part for the Gemini API.
        
        Args:
            image_path_or_url: Path to local image or URL of remote image.
            
        Returns:
            Image part for Gemini API or None if preparation fails.
        """
        try:
            # Check if it's a URL
            if image_path_or_url.startswith(('http://', 'https://')):
                logging.info(f"Attempting to download image from: {image_path_or_url}")
                response_img = requests.get(image_path_or_url, timeout=30)
                response_img.raise_for_status()
                
                image_bytes = response_img.content
                if not image_bytes:
                    logging.warning(f"Downloaded image from {image_path_or_url} appears empty.")
                    return None
                
                mime_type, _ = mimetypes.guess_type(image_path_or_url)
                if not mime_type or not mime_type.startswith('image/'):
                    content_type_header = response_img.headers.get('Content-Type')
                    if content_type_header and content_type_header.split(';')[0].strip().startswith('image/'):
                        mime_type = content_type_header.split(';')[0].strip()
                    else:
                        default_mime = 'image/jpeg'
                        logging.warning(f"Could not reliably determine image MIME type for {image_path_or_url}. Defaulting to '{default_mime}'.")
                        mime_type = default_mime
            
            # Local file
            elif os.path.exists(image_path_or_url):
                with open(image_path_or_url, "rb") as f:
                    image_bytes = f.read()
                
                mime_type, _ = mimetypes.guess_type(image_path_or_url)
                if not mime_type or not mime_type.startswith('image/'):
                    default_mime = 'image/jpeg'
                    logging.warning(f"Could not determine MIME type for {image_path_or_url}. Defaulting to '{default_mime}'.")
                    mime_type = default_mime
            else:
                logging.error(f"Image not found: {image_path_or_url}")
                return None
            
            # Create image part
            image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            logging.info(f"Successfully prepared image part (MIME: {mime_type}).")
            return image_part
            
        except requests.exceptions.RequestException as img_req_err:
            logging.error(f"Failed to download image from {image_path_or_url}: {img_req_err}")
            return None
        except Exception as img_err:
            logging.error(f"Error processing image from {image_path_or_url}: {img_err}")
            return None
    
    def generate_response(
        self, 
        prompt: str, 
        image_path_or_url: Optional[str] = None,
        context: Optional[str] = None,
        max_tokens: int = 1000
    ) -> str:
        """
        Generate a response from the Gemini model.
        
        Args:
            prompt: Text prompt for the model.
            image_path_or_url: Optional path or URL to an image.
            context: Optional context from retrieved documents.
            max_tokens: Maximum number of tokens in the response.
            
        Returns:
            Generated text response.
        """
        # Prepare the prompt with context if provided
        if context:
            full_prompt = f"""You are a helpful assistant that answers questions based on the provided context.
If the context doesn't contain relevant information, use your general knowledge but make it clear.

Context:
{context}

Question: {prompt}

Answer:"""
        else:
            full_prompt = prompt
        
        # Prepare contents for the API call
        contents = []
        
        # Add text prompt
        contents.append(full_prompt)
        
        # Add image if provided
        if image_path_or_url:
            image_part = self._prepare_image_part(image_path_or_url)
            if image_part:
                contents.append(image_part)
            else:
                # If image preparation fails, add a note to the prompt
                contents = [f"{full_prompt}\n\n(Note: I tried to include an image but it couldn't be processed.)"]
        
        # Retry loop - try each key once
        start_key_index = self.current_key_index
        for attempt in range(len(self.api_keys)):
            current_attempt_index = (start_key_index + attempt) % len(self.api_keys)
            
            # Ensure the client is initialized with the correct key for this attempt
            if attempt > 0 or self.client is None:
                if not self._switch_to_next_key():
                    logging.error("Failed to initialize next key. Cannot proceed.")
                    return "Error: Failed to generate response due to API key issues."
            
            try:
                logging.info(f"Generating response using model '{self.model_name}' with API key index {self.current_key_index} (Attempt {attempt + 1}/{len(self.api_keys)})")
                
                # Call the Gemini API
                response = self.client.models.generate_content(
                    model=f"models/{self.model_name}",
                    contents=contents,
                    generation_config={"max_output_tokens": max_tokens}
                )
                
                # Extract and return the response text
                return response.text.strip()
                
            except Exception as e:
                if "rate limit" in str(e).lower() or "quota exceeded" in str(e).lower():
                    logging.warning(f"Rate limit or quota exceeded for key index {self.current_key_index}: {e}. Attempting to switch key.")
                    time.sleep(1)  # Brief pause before trying the next key
                    continue
                elif attempt == len(self.api_keys) - 1:
                    # Last attempt failed
                    logging.error(f"All {len(self.api_keys)} API keys failed. Error: {e}")
                    return f"Error: Failed to generate response. {str(e)}"
                else:
                    # Try the next key
                    logging.warning(f"Error with key index {self.current_key_index}: {e}. Trying next key.")
                    time.sleep(1)
                    continue
        
        # If we get here, all keys failed
        return "Error: Failed to generate response after trying all available API keys."


if __name__ == "__main__":
    # Simple test
    try:
        llm = GeminiLLM()
        
        # Test with text only
        text_response = llm.generate_response(
            prompt="What is retrieval-augmented generation?",
            max_tokens=100
        )
        print(f"Text-only response:\n{text_response}\n")
        
        # Test with image (replace with a valid URL or local path)
        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/ChatGPT_logo.svg/1024px-ChatGPT_logo.svg.png"
        image_response = llm.generate_response(
            prompt="What is shown in this image?",
            image_path_or_url=image_url,
            max_tokens=100
        )
        print(f"Image response:\n{image_response}")
        
    except Exception as e:
        print(f"Error in Gemini LLM test: {e}")
