# QAGenerater.py

# --- Imports ---
from google import genai
from google.genai import types
# --- End Imports ---

import os
import logging
from dotenv import load_dotenv
import json
from typing import List, Dict, Any, Optional
import requests
import mimetypes
import time # Import time for potential backoff (optional but good practice)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure the mimetypes module is initialized
mimetypes.init()

class QAGenerator:
    """
    Generates Question/Answer pairs from text content and an associated
    image using the Google Gemini API (Client approach). Assumes image_link is always provided.
    Includes API key rotation logic.
    """
    def __init__(self):
        """
        Initializes the QAGenerator, loading Google API keys,
        configuring the initial Gemini client, and setting up key rotation.
        """
        load_dotenv()
        # Load comma-separated keys
        keys_csv = os.getenv("GOOGLE_API_KEYS_CSV")
        if not keys_csv:
            raise ValueError("GOOGLE_API_KEYS_CSV environment variable not set or empty.")

        self.api_keys = [key.strip() for key in keys_csv.split(',') if key.strip()]
        if not self.api_keys:
            raise ValueError("No valid API keys found in GOOGLE_API_KEYS_CSV.")

        self.current_key_index = 0
        self.client = None # Will be initialized by _initialize_client
        self.model_name = 'gemini-2.0-flash' # Using 1.5 flash as it's stable and multimodal

        logging.info(f"Loaded {len(self.api_keys)} API keys.")

        # Initialize the client with the first key
        self._initialize_client(self.current_key_index)

    def _initialize_client(self, key_index: int):
        """Initializes or re-initializes the Gemini client with the specified key."""
        if not 0 <= key_index < len(self.api_keys):
            raise IndexError("Invalid API key index provided.")

        api_key = self.api_keys[key_index]
        try:
            # Create a new client instance
            self.client = genai.Client(api_key=api_key)
            # Perform a simple test call (optional but recommended)
            # self.client.models.list() # Example test
            logging.info(f"Google Generative AI client initialized successfully with key index {key_index} for model '{self.model_name}'.")
            self.current_key_index = key_index # Update the current index tracker
        except Exception as e:
            logging.error(f"Failed to initialize Google Generative AI client with key index {key_index}: {e}")
            # Decide how to handle this: maybe try next key immediately or raise?
            # For now, we'll raise to indicate a setup problem.
            raise RuntimeError(f"Failed to initialize client with key index {key_index}") from e

    def _switch_to_next_key(self) -> bool:
        """Switches to the next available API key and re-initializes the client."""
        initial_index = self.current_key_index
        next_index = (self.current_key_index + 1) % len(self.api_keys)

        logging.warning(f"Attempting to switch API key from index {initial_index} to {next_index}.")

        # Try initializing with the next key
        try:
            self._initialize_client(next_index)
            logging.info(f"Successfully switched API key to index {next_index}.")
            return True # Successfully switched
        except Exception as e:
            logging.error(f"Failed to switch to API key index {next_index}: {e}. Staying with index {initial_index} (which might be problematic).")
            # If switching fails, we might be stuck. Depending on the error,
            # retrying the *same* key might still be useful if it was a temporary network issue.
            # However, if the key itself is invalid, we might loop indefinitely without this check.
            # For simplicity now, we just report the failure to switch.
            return False # Failed to switch

    def _create_text_prompt(self, text_content: str) -> str:
        """Creates the text part of the prompt for the Gemini model."""
        # (Prompt remains the same as in your provided code)
        prompt = f"""Imagine I just sent you this meme. I've seen it around, or maybe it's totally new to me, but I'm really curious and don't fully get it, or I just want to know the story behind it, especially *who* the person/character is.

Could you act like a friendly, knowledgeable storyteller and help me out?

Please generate 5 distinct Question/Answer pairs about this specific meme. **For *each* pair, you must provide *both* an English version and a Vietnamese version.**

* **The Questions:** These should sound like they're coming from me, genuinely curious, wanting to understand. Generate both an English (`en`) and Vietnamese (`vi`) version for each question. Think along these lines:
    * **(EN) Who *is* this person/character? / (VI) Người/nhân vật *này* là ai vậy?**
    * (EN) Okay, what's the actual joke here? / (VI) Rồi, cái trò đùa ở đây thực sự là gì vậy?
    * (You should also ask about the image itself)

* **The Answers:** Now, put on your storyteller hat! Don't just give dry facts. Weave a little narrative. Explain the origin (including who the person/character is, if known), the typical usage, the specific humor, or the feeling it evokes in an engaging, story-like way. Provide *both* an English (`en`) and a Vietnamese (`vi`) version for each answer. Ensure the storyteller tone comes through in both languages.

**Crucially:**
1.  Base your answers *only* on the provided meme image and any context given. Identify the character *if possible based on the image/context*, but don't invent information if they aren't identifiable.
2.  Ensure the questions and answers are distinct from each other.
3.  Maintain the appropriate tone (curious asker, engaging storyteller) in *both* languages.

Format your output as a valid JSON list of objects. Each object represents one Q&A pair and must contain two keys: "question" and "answer".

**Example Output Format:**
```json
        [

        {{"question": "en: Who is this person in the meme?\nvi: Người trong meme này là ai vậy?","answer": "en: Ah, so this is [Name/Description of character]. They became famous because [brief origin story related to the meme image]...\nvi: À, vậy đây là [Tên/Mô tả nhân vật]. Họ trở nên nổi tiếng vì [câu chuyện gốc ngắn gọn liên quan đến hình ảnh meme]..."}},

        {{"question": "en: Why is *that* caption funny with *this* character's expression? vi: Tại sao cái chú thích *đó* lại hài hước khi đi với biểu cảm của nhân vật *này*?","answer": "en: Well, the humor comes from the contrast! This character is known for [typical trait/situation], shown by their expression, but the caption talks about [unrelated topic B]. Connecting them creates the joke...\nvi: Chà, sự hài hước đến từ sự tương phản! Nhân vật này được biết đến với [đặc điểm/tình huống điển hình], thể hiện qua biểu cảm của họ, nhưng chú thích lại nói về [chủ đề không liên quan B]. Việc kết nối chúng tạo ra tiếng cười..."}},

        ]

        Text Content:
        ---
        {text_content}
        ---

        JSON Output:"""

        return prompt

    def _prepare_image_part(self, image_url: str, title: str) -> Optional[types.Part]:
        """
        Downloads an image and prepares it as a types.Part for the Gemini API.
        Assumes the URL is valid and downloadable.
        """
        # (This method remains the same as in your provided code)
        try:
            logging.info(f"Attempting to download image for '{title}' from: {image_url}")
            # Increased timeout slightly
            response_img = requests.get(image_url, timeout=30)
            response_img.raise_for_status()

            image_bytes = response_img.content
            if not image_bytes:
                 logging.warning(f"Downloaded image for '{title}' from {image_url} appears empty.")
                 return None

            mime_type, _ = mimetypes.guess_type(image_url)
            if not mime_type or not mime_type.startswith('image/'):
                content_type_header = response_img.headers.get('Content-Type')
                if content_type_header and content_type_header.split(';')[0].strip().startswith('image/'):
                    mime_type = content_type_header.split(';')[0].strip()
                else:
                    default_mime = 'image/jpeg'
                    logging.warning(f"Could not reliably determine image MIME type for {image_url}. Defaulting to '{default_mime}'.")
                    mime_type = default_mime

            image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            logging.info(f"Successfully prepared image part for '{title}' (MIME: {mime_type}).")
            return image_part

        except requests.exceptions.RequestException as img_req_err:
            logging.error(f"Failed to download image for '{title}' from {image_url}: {img_req_err}")
            return None
        except Exception as img_err:
            logging.error(f"Error processing image for '{title}' from {image_url}: {img_err}")
            return None

    def generate_qas_for_page(self, content: str, title: str, image_url: str) -> Optional[List[Dict[str, str]]]:
        """
        Generates QA pairs for a page's content and mandatory image using the Gemini model client.
        Includes retry logic with API key rotation.
        """
        if not content or not content.strip():
            logging.warning(f"Skipping QA generation for '{title}' due to empty text content.")
            return None

        image_part = self._prepare_image_part(image_url, title)
        if not image_part:
            logging.error(f"Skipping QA generation for '{title}' because image preparation failed.")
            return None

        text_prompt = self._create_text_prompt(content)
        gemini_contents = [text_prompt, image_part]
        generation_mode = "(with image)"

        # Retry loop - try each key once for this specific request
        start_key_index = self.current_key_index
        for attempt in range(len(self.api_keys)):
            current_attempt_index = (start_key_index + attempt) % len(self.api_keys)

            # Ensure the client is initialized with the correct key for this attempt
            # This handles the first attempt and subsequent retries after switching.
            if attempt > 0 or self.client is None: # If retrying or client not set
                 # Attempt to switch/initialize client. If it fails, we likely can't proceed.
                 if not self._switch_to_next_key():
                      logging.error(f"Halting attempts for '{title}' as failed to initialize next key.")
                      return None # Cannot proceed if client init fails

            # Double check the client is using the intended key index for this attempt
            # (This might be overly cautious but helps ensure state consistency)
            if self.current_key_index != current_attempt_index:
                 logging.warning(f"Client key index ({self.current_key_index}) differs from attempt index ({current_attempt_index}). Attempting re-sync.")
                 try:
                     self._initialize_client(current_attempt_index)
                 except Exception as sync_err:
                     logging.error(f"Failed to re-sync client to index {current_attempt_index} for '{title}': {sync_err}. Aborting.")
                     return None


            try:
                logging.info(f"Generating QAs for page: '{title}' {generation_mode} using model '{self.model_name}' with API key index {self.current_key_index} (Attempt {attempt + 1}/{len(self.api_keys)})")

                response = self.client.models.generate_content(
                    model=f"models/{self.model_name}",
                    contents=gemini_contents,
                    # Consider adding safety settings if needed
                    # safety_settings=...
                    # Consider generation config if needed (temp, top_p, etc.)
                    # generation_config=...
                )

                # --- Successful Response Parsing ---
                raw_response_text = response.text.strip()
                logging.debug(f"Raw Gemini response for '{title}' (Key Index {self.current_key_index}):\n{raw_response_text}")

                json_start = raw_response_text.find('[')
                json_end = raw_response_text.rfind(']') + 1

                if json_start != -1 and json_end != -1:
                    json_string = raw_response_text[json_start:json_end]
                    try:
                        qa_pairs = json.loads(json_string)
                        if isinstance(qa_pairs, list) and all(isinstance(item, dict) and 'question' in item and 'answer' in item for item in qa_pairs):
                            qa_pairs = self.process_final_output(qa_pairs)
                            logging.info(f"Successfully generated {len(qa_pairs)} QA pairs for '{title}' using key index {self.current_key_index}.")
                            return qa_pairs # SUCCESS! Exit the loop and function.
                        else:
                            logging.error(f"Parsed JSON for '{title}' (Key Index {self.current_key_index}) is not in the expected format: {json_string}")
                            # Treat unexpected format as an error for this key, maybe retry? Or fail? For now, let's fail this attempt.
                            # Consider if this error type should trigger a key switch or just fail the page.
                            # Let's try switching key for this.
                            raise ValueError("Parsed JSON not in expected format") # Raise to trigger key switch logic below

                    except json.JSONDecodeError as json_err:
                        logging.error(f"Failed to decode JSON response for '{title}' (Key Index {self.current_key_index}): {json_err}\nRaw JSON string attempted: {json_string}")
                        # This might be a model issue or transient error. Let's try switching key.
                        raise ValueError("JSON Decode Error") from json_err # Raise to trigger key switch
                else:
                    logging.error(f"Could not find valid JSON list structure in Gemini response for '{title}' (Key Index {self.current_key_index}). Response: {raw_response_text}")
                    # This might be a model issue (e.g., safety block). Let's try switching key.
                    # Log feedback if available
                    if hasattr(response, 'prompt_feedback'):
                         logging.error(f"Gemini prompt feedback: {response.prompt_feedback}")
                    if hasattr(response, 'candidates') and response.candidates:
                         logging.error(f"Gemini candidates finish reason: {response.candidates[0].finish_reason}")
                         logging.error(f"Gemini candidates safety ratings: {response.candidates[0].safety_ratings}")
                    raise ValueError("No valid JSON structure found in response") # Raise to trigger key switch

                # --- End Response Parsing ---

            # --- Specific API Error Handling for Retry/Key Switch ---
            except Exception as rate_limit_err:
                if "rate limit" in str(rate_limit_err).lower() or "quota exceeded" in str(rate_limit_err).lower():
                    # Handle rate limit or quota exceeded errors
                    logging.warning(f"Rate limit or quota exceeded for key index {self.current_key_index} for '{title}': {rate_limit_err}. Attempting to switch key.")
                    # Simple backoff before switching (optional)
                    time.sleep(20)
                # The loop structure handles switching via _switch_to_next_key on the next iteration
                continue # Continue to the next iteration of the loop to try the next key

            except ValueError as parsing_err: # Catch our custom ValueErrors raised above
                logging.warning(f"Response parsing issue for '{title}' (Key Index {self.current_key_index}): {parsing_err}. Attempting to switch key.")
                time.sleep(1)
                # Continue to the next iteration of the loop to try the next key
                continue

            # --- General Exception Handling (Non-retryable for this page) ---
            except Exception as e:
                logging.error(f"Unhandled error generating QAs for '{title}' using key index {self.current_key_index}: {e}")
                # Log details if available (same as before)
                if 'response' in locals() and hasattr(response, 'prompt_feedback'):
                     logging.error(f"Gemini prompt feedback on error: {response.prompt_feedback}")
                if 'response' in locals() and hasattr(response, 'candidates') and response.candidates:
                     logging.error(f"Gemini candidates on error: {response.candidates}")
                elif 'response' in locals() and hasattr(response, 'text'):
                     logging.error(f"Gemini raw response text on error: {response.text}")

                # Don't retry for unknown errors, fail this page generation
                return None # Exit the loop and function.

        # If the loop completes without returning success, all keys failed for this request
        logging.error(f"All {len(self.api_keys)} API keys failed for '{title}'. Skipping QA generation for this page.")
        return None

    def process_scraped_data(self, scraped_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Processes a list of scraped page data, generating QA pairs for each page,
        using the mandatory associated image and handling API key rotation.
        """
        # (This method remains largely the same, it just calls the updated generate_qas_for_page)
        results_with_qas = []
        if not scraped_data:
            logging.warning("Received empty list of scraped data. No QAs to generate.")
            return results_with_qas

        for page_data in scraped_data:
            title = page_data.get('title', 'Unknown Title')
            content = page_data.get('content')
            page_url = page_data.get('url', 'Unknown URL')
            image_link = page_data.get('image_link')

            generated_qas = None
            
            if content:
                if len(content.split()) < 100:
                    logging.warning(f"Skipping page '{title}' ({page_url}) due to insufficient content length.")
                    continue # Skip this page if content is too short
            
            if not content:
                logging.warning(f"Skipping page '{title}' ({page_url}) due to missing 'content' field.")
            elif not image_link:
                 logging.warning(f"Skipping page '{title}' ({page_url}) due to missing 'image_link' field (violates assumption).")
            else:
                # This now uses the method with retry/rotation logic
                generated_qas = self.generate_qas_for_page(content, title, image_url=image_link)

            processed_page_data = page_data.copy()
            processed_page_data['qa_pairs'] = generated_qas if generated_qas else [] # Ensure qa_pairs is a list even on failure
            results_with_qas.append(processed_page_data)

        logging.info(f"Finished processing {len(scraped_data)} pages for QA generation.")
        return results_with_qas

    def process_final_output(self, qa_pairs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Processes the raw QA pairs list to split en/vi versions."""
        # (This method remains the same as in your provided code)
        processed_list = []
        missing_vi_count = 0
        for item in qa_pairs:
            try:
                eng_item = {}
                vi_item = {}

                # Robust splitting
                q_parts = item.get('question', '').split("vi:")
                a_parts = item.get('answer', '').split("vi:")

                eng_item['question'] = q_parts[0].replace("en:", "").strip() if len(q_parts) > 0 else ""
                eng_item['answer'] = a_parts[0].replace("en:", "").strip() if len(a_parts) > 0 else ""

                if len(q_parts) > 1 and len(a_parts) > 1:
                    vi_item['question'] = q_parts[1].strip()
                    vi_item['answer'] = a_parts[1].strip()
                else:
                    # Handle cases where 'vi:' might be missing or format is unexpected
                    vi_item['question'] = "" # Or potentially copy english? Or log warning?
                    vi_item['answer'] = ""
                    missing_vi_count += 1

                # Only add if the core parts exist
                if eng_item.get('question') and eng_item.get('answer'):
                     processed_list.append(eng_item)
                if vi_item.get('question') and vi_item.get('answer'):
                     processed_list.append(vi_item)
                elif not vi_item.get('question') and not vi_item.get('answer') and (eng_item.get('question') or eng_item.get('answer')):
                     # Log if only english part was processed due to missing vi part
                     logging.debug(f"Processed item only had English content: {item}")


            except Exception as e:
                logging.error(f"Error processing QA pair item: {item}. Error: {e}")
                continue # Skip this malformed item

        if missing_vi_count > 0:
             logging.warning(f"Processed {len(qa_pairs)} raw pairs, but {missing_vi_count} were missing the 'vi:' delimiter or content.")

        return processed_list


# --- Example Usage ---
if __name__ == "__main__":
    # IMPORTANT: Set the GOOGLE_API_KEYS_CSV environment variable!
    # Example: export GOOGLE_API_KEYS_CSV="YOUR_KEY_1,YOUR_KEY_2,YOUR_KEY_3"
    # Or add it to your .env file:
    # GOOGLE_API_KEYS_CSV=YOUR_KEY_1,YOUR_KEY_2,YOUR_KEY_3

    # Mock data remains the same
    mock_scraped_data = [{'title': '"MGE Paravozik"', 'url': '[https://brainrot.fandom.com/wiki/%22MGE_Paravozik%22](https://brainrot.fandom.com/wiki/%22MGE_Paravozik%22)', 'content': 'This is ABSOLUTELY brainrot russian 18+ TF2 meme. I can\'t place a pic here because it\'s NSFW. If you REALLY want to know what is it, google "МГЕ паравозик" or "Ух Артём". Totally brainrot.', 'image_link': 'https://static.wikia.nocookie.net/brainrotnew/images/3/39/Site-community-image/revision/latest?cb=20250104173614', 'crawled_at': '2025-04-22T23:09:34.957619'}] # Note: This image link might be broken/outdated

    # Example with a potentially working image link (replace if needed)
    


    try:
        qa_generator = QAGenerator()
        # Use the mock data with the working image link
        results = qa_generator.process_scraped_data(mock_scraped_data)

        print("\n--- QA Generation Results ---")
        print(json.dumps(results, indent=2, ensure_ascii=False)) # Pretty print JSON

    except ValueError as ve:
        logging.error(f"Initialization failed: {ve}")
    except RuntimeError as rte:
         logging.error(f"Client initialization runtime error: {rte}")
    except Exception as main_err:
        logging.error(f"An error occurred during QA generation: {main_err}")