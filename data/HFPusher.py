# HFPusher.py

import os
import logging
from dotenv import load_dotenv
# Import concatenate_datasets
from datasets import Dataset, load_dataset, concatenate_datasets
# Import specific exception for dataset loading
from datasets.exceptions import DatasetNotFoundError
from huggingface_hub import HfApi, HfFolder
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HFPusher:
    """
    Processes data containing scraped page info and QA pairs, then pushes
    it to a specified Hugging Face Dataset repository, appending to existing data.
    """
    def __init__(self):
        """
        Initializes the HFPusher, loading necessary environment variables
        for Hugging Face authentication and repository configuration.
        """
        load_dotenv()
        self.hf_token = os.getenv("HF_TOKEN")
        self.repo_id = os.getenv("HF_DATASET_REPO_ID") # e.g., "your_username/your_dataset_name"

        if not self.hf_token:
            raise ValueError("HF_TOKEN environment variable not set. Please provide your Hugging Face API token.")
        if not self.repo_id:
            raise ValueError("HF_DATASET_REPO_ID environment variable not set. Please specify the target Hugging Face dataset repository ID.")

        try:
            HfFolder.save_token(self.hf_token)
            logging.info("Hugging Face token saved successfully for library use.")
        except Exception as e:
            logging.warning(f"Could not save Hugging Face token automatically: {e}. Ensure token is available environment.")

    def _process_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transforms the raw input data (list of page dicts with nested qa_pairs)
        into a flattened list suitable for a Hugging Face Dataset.

        (This method remains the same as before)
        """
        processed_list = []
        if not raw_data:
            logging.warning("Received empty raw data list for processing.")
            return processed_list

        for page_data in raw_data:
            title = page_data.get('title')
            image_link = page_data.get('image_link')
            qa_pairs = page_data.get('qa_pairs')

            if not title:
                logging.warning(f"Skipping page data due to missing 'title': {page_data.get('url', 'N/A')}")
                continue

            if not qa_pairs or not isinstance(qa_pairs, list):
                # Ensure qa_pairs is treated as empty if missing/invalid, preventing errors later
                # logging.warning(f"Page '{title}' has missing or invalid 'qa_pairs'. No QA items to process.")
                qa_pairs = [] # Process the page but add no QA items

            for qa_pair in qa_pairs:
                question = qa_pair.get('question')
                answer = qa_pair.get('answer')

                if question and answer:
                    processed_item = {
                        'question': question,
                        'answer': answer,
                        'image_link': image_link,
                        'title': title
                    }
                    processed_list.append(processed_item)
                else:
                    logging.warning(f"Skipping QA pair in '{title}' due to missing question or answer: {qa_pair}")

        logging.info(f"Processed {len(raw_data)} raw items into {len(processed_list)} flattened QA items.")
        return processed_list

    def push_data(self, data_to_push: List[Dict[str, Any]], private: bool = False):
        """
        Processes the input data, loads existing data from the Hub (if any),
        concatenates, and pushes the updated dataset to the configured repository.

        Args:
            data_to_push: The raw list of dictionaries (output from QAGenerator).
            private: Set to True if the repository is private. Defaults to False.
        """
        processed_new_data = self._process_data(data_to_push)

        if not processed_new_data:
            logging.warning("No new processed data available to push to Hugging Face Hub.")
            return

        try:
            logging.info(f"Preparing to update dataset: {self.repo_id}")
            # Create a Dataset object for the new data
            new_dataset = Dataset.from_list(processed_new_data)
            logging.info(f"Created dataset object from {len(new_dataset)} new items.")

            # --- Load existing dataset and concatenate ---
            try:
                # Attempt to load the existing dataset from the hub
                logging.info(f"Attempting to load existing dataset from {self.repo_id}...")
                # Use token for private repos or if needed for authentication context
                existing_dataset = load_dataset(self.repo_id, token=self.hf_token, split='train') # Assuming default split is 'train'
                logging.info(f"Successfully loaded existing dataset with {len(existing_dataset)} items.")

                # Concatenate the existing and new datasets
                combined_dataset = concatenate_datasets([existing_dataset, new_dataset])
                logging.info(f"Concatenated datasets. Total items: {len(combined_dataset)}")
                dataset_to_push = combined_dataset

            except DatasetNotFoundError:
                # Handle the case where the dataset doesn't exist yet
                logging.info(f"Dataset {self.repo_id} not found on the Hub. Creating new dataset.")
                dataset_to_push = new_dataset
            except Exception as load_err:
                # Handle other potential errors during loading
                logging.error(f"Error loading existing dataset {self.repo_id}: {load_err}. Will attempt to push only new data.")
                # Fallback: push only the new data (might overwrite if repo exists but loading failed)
                # Or you could choose to abort here depending on desired behavior
                dataset_to_push = new_dataset
            # --- End loading and concatenation ---


            # --- Push the final dataset ---
            if dataset_to_push:
                logging.info(f"Pushing {len(dataset_to_push)} items to Hugging Face Hub repository: {self.repo_id}")
                dataset_to_push.push_to_hub(
                    repo_id=self.repo_id,
                    token=self.hf_token,
                    private=private
                    # Default behavior should handle splits correctly, often creating/updating 'train'
                )
                logging.info(f"Successfully pushed dataset to {self.repo_id}")
            else:
                 logging.warning("No dataset object available to push after processing and loading.")


        except Exception as e:
            logging.error(f"Failed during the dataset push process for {self.repo_id}: {e}")
            # Consider more specific error handling


# --- Example Usage (remains similar) ---
if __name__ == "__main__":
    # IMPORTANT: Set environment variables HF_TOKEN and HF_DATASET_REPO_ID

    sample_input_data = [
      {
        "title": "\"MGE Paravozik\" - Run 2", # Added identifier for testing append
        "url": "https://brainrot.fandom.com/wiki/%22MGE_Paravozik%22",
        "content": "This is ABSOLUTELY brainrot russian 18+ TF2 meme...",
        "image_link": "https://static.wikia.nocookie.net/brainrotnew/images/3/39/Site-community-image/revision/latest?cb=20250104173614",
        "crawled_at": "2025-04-23T10:00:00.000000", # Different timestamp
        "qa_pairs": [
          {
            "question": "What is this meme about? (Update)",
            "answer": "This is an updated answer for the second run..."
          },
          {
            "question": "Meme này nói về cái gì vậy? (Cập nhật)",
            "answer": "Đây là câu trả lời cập nhật cho lần chạy thứ hai..."
          },
        ]
      },
    ]

    try:
        pusher = HFPusher()
        # Push the data (this will now append if the dataset exists)
        pusher.push_data(sample_input_data)
        logging.info("Example script finished.")

    except ValueError as ve:
        logging.error(f"Initialization failed: {ve}")
    except Exception as main_err:
        logging.error(f"An error occurred during the push process: {main_err}")

