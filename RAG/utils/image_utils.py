"""
Utility functions for image processing in the RAG system.
"""

import os
import requests
from io import BytesIO
from typing import Optional, Tuple, Union
from PIL import Image
import base64
import uuid

def download_image(image_url: str) -> Optional[bytes]:
    """
    Download an image from a URL.
    
    Args:
        image_url: URL of the image to download.
        
    Returns:
        Image bytes or None if download fails.
    """
    try:
        response = requests.get(image_url, timeout=10)
        if response.status_code == 200:
            return response.content
        else:
            print(f"Failed to download image. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

def save_image(image_data: bytes, directory: str = "uploads") -> Optional[str]:
    """
    Save image data to a file.
    
    Args:
        image_data: Image data as bytes.
        directory: Directory to save the image in.
        
    Returns:
        Path to the saved image or None if saving fails.
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Generate a unique filename
        filename = f"{uuid.uuid4()}.jpg"
        filepath = os.path.join(directory, filename)
        
        # Save the image
        with open(filepath, "wb") as f:
            f.write(image_data)
        
        return filepath
    except Exception as e:
        print(f"Error saving image: {e}")
        return None

def resize_image(image_path: str, max_size: Tuple[int, int] = (800, 800)) -> Optional[str]:
    """
    Resize an image to fit within the specified dimensions while maintaining aspect ratio.
    
    Args:
        image_path: Path to the image file.
        max_size: Maximum width and height.
        
    Returns:
        Path to the resized image or None if resizing fails.
    """
    try:
        # Open the image
        img = Image.open(image_path)
        
        # Calculate new dimensions
        width, height = img.size
        ratio = min(max_size[0] / width, max_size[1] / height)
        
        # Only resize if the image is larger than max_size
        if ratio < 1:
            new_size = (int(width * ratio), int(height * ratio))
            img = img.resize(new_size, Image.LANCZOS)
            
            # Save the resized image
            img.save(image_path)
        
        return image_path
    except Exception as e:
        print(f"Error resizing image: {e}")
        return None

def encode_image_base64(image_path: str) -> Optional[str]:
    """
    Encode an image as base64.
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        Base64-encoded image string or None if encoding fails.
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def is_valid_image(file_content: bytes) -> bool:
    """
    Check if the file content is a valid image.
    
    Args:
        file_content: File content as bytes.
        
    Returns:
        True if the file is a valid image, False otherwise.
    """
    try:
        Image.open(BytesIO(file_content))
        return True
    except Exception:
        return False

def get_image_format(image_path: str) -> Optional[str]:
    """
    Get the format of an image file.
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        Image format (e.g., 'JPEG', 'PNG') or None if the format cannot be determined.
    """
    try:
        with Image.open(image_path) as img:
            return img.format
    except Exception as e:
        print(f"Error getting image format: {e}")
        return None
