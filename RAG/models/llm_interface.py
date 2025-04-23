"""
Multimodal LLM interface for RAG system.
Handles interactions with OpenAI's multimodal models.
"""

import os
import base64
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
import requests
from io import BytesIO
from PIL import Image
import openai

class MultimodalLLM:
    """
    Interface for multimodal language models that can process text and images.
    """
    
    def __init__(self, model_name: str = "gpt-4-vision-preview"):
        """
        Initialize the multimodal LLM interface.
        
        Args:
            model_name: Name of the OpenAI model to use.
        """
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        if not openai.api_key:
            raise ValueError("Missing OpenAI API key. Please check your .env file.")
        
        self.model_name = model_name
        print(f"Multimodal LLM initialized with model: {model_name}")
    
    def _encode_image(self, image_path_or_url: str) -> Optional[str]:
        """
        Encode an image as base64 for API requests.
        
        Args:
            image_path_or_url: Path to local image or URL of remote image.
            
        Returns:
            Base64-encoded image string or None if encoding fails.
        """
        try:
            # Check if it's a URL
            if image_path_or_url.startswith(('http://', 'https://')):
                response = requests.get(image_path_or_url)
                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content))
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    return base64.b64encode(buffered.getvalue()).decode('utf-8')
                else:
                    print(f"Failed to download image from URL: {image_path_or_url}")
                    return None
            # Local file
            elif os.path.exists(image_path_or_url):
                with open(image_path_or_url, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')
            else:
                print(f"Image not found: {image_path_or_url}")
                return None
        except Exception as e:
            print(f"Error encoding image: {e}")
            return None
    
    def generate_response(
        self, 
        prompt: str, 
        image_path_or_url: Optional[str] = None,
        context: Optional[str] = None,
        max_tokens: int = 1000
    ) -> str:
        """
        Generate a response from the multimodal LLM.
        
        Args:
            prompt: Text prompt for the model.
            image_path_or_url: Optional path or URL to an image.
            context: Optional context from retrieved documents.
            max_tokens: Maximum number of tokens in the response.
            
        Returns:
            Generated text response.
        """
        messages = []
        
        # Add system message with context if provided
        if context:
            system_content = "You are a helpful assistant that answers questions based on the provided context."
            system_content += " If the context doesn't contain relevant information, use your general knowledge but make it clear."
            system_content += f"\n\nContext:\n{context}"
            
            messages.append({
                "role": "system",
                "content": system_content
            })
        else:
            messages.append({
                "role": "system",
                "content": "You are a helpful assistant that can understand images and text."
            })
        
        # Prepare user message
        if image_path_or_url:
            # Encode image
            encoded_image = self._encode_image(image_path_or_url)
            
            if encoded_image:
                # Create message with text and image
                message_content = [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    }
                ]
                
                messages.append({
                    "role": "user",
                    "content": message_content
                })
            else:
                # Fallback to text-only if image encoding fails
                messages.append({
                    "role": "user",
                    "content": f"{prompt}\n\n(Note: I tried to include an image but it couldn't be processed.)"
                })
        else:
            # Text-only message
            messages.append({
                "role": "user",
                "content": prompt
            })
        
        try:
            # Call the OpenAI API
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens
            )
            
            # Extract and return the response text
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Error: Failed to generate response. {str(e)}"


if __name__ == "__main__":
    # Simple test
    try:
        llm = MultimodalLLM()
        
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
        print(f"Error in multimodal LLM test: {e}")
