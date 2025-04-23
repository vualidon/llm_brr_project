# -*- coding: utf-8 -*-
"""
Encapsulated Qwen2.5_VL_(7B)-Vision Training & Inference Pipeline
(Revised for question/answer/image_link dataset format)
"""

# %%capture
# !pip install "unsloth[colab-newest] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps xformers==0.0.29.post3 trl==0.15.2 peft accelerate bitsandbytes sentencepiece protobuf datasets huggingface_hub hf_transfer pillow requests # Added pillow & requests
# !pip install --no-deps unsloth_zoo cut_cross_entropy triton # Optional extra dependencies

import os
import torch
import requests # For loading images from URLs
from io import BytesIO # For handling image data from requests
from PIL import Image, UnidentifiedImageError # For loading and handling images
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers import TextStreamer
from datasets import load_dataset, Dataset
from typing import List, Dict, Optional, Union, Any
from IPython.display import display, Math # Keep for displaying results

# Helper function to safely load images from path or URL
def load_image_from_source(source: str) -> Optional[Image.Image]:
    """Loads an image from a local path or a URL."""
    try:
        if source.startswith(('http://', 'https://')):
            response = requests.get(source, stream=True, timeout=10) # Added timeout
            response.raise_for_status() # Raise an exception for bad status codes
            img = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            # Check if the file exists locally
            if not os.path.exists(source):
                print(f"Warning: Local image file not found: {source}")
                return None
            img = Image.open(source).convert("RGB")
        return img
    except (requests.exceptions.RequestException, UnidentifiedImageError, FileNotFoundError, IOError, Exception) as e:
        print(f"Warning: Could not load image from {source}. Error: {e}")
        return None

class UnslothVisionPipeline:
    """
    An encapsulated pipeline for Unsloth Vision Model finetuning and inference.

    Handles model loading, PEFT setup, data processing (from question/answer/image_link format),
    training, and inference.
    """
    def __init__(
        self,
        # --- Model Loading ---
        base_model_name: str = "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
        load_in_4bit: bool = True,
        use_gradient_checkpointing: Union[str, bool] = "unsloth",
        lora_adapter_path: Optional[str] = None, # Path to load pre-trained adapters

        # --- PEFT Configuration (only used if lora_adapter_path is None) ---
        finetune_vision_layers: bool = True, finetune_language_layers: bool = True,
        finetune_attention_modules: bool = True, finetune_mlp_modules: bool = True,
        r: int = 16, lora_alpha: int = 16, lora_dropout: float = 0, bias: str = "none",
        random_state: int = 3407, use_rslora: bool = False, loftq_config: Optional[dict] = None,

        # --- Data Configuration ---
        # Removed fixed instruction. Expects 'question', 'answer', 'image_link' columns.
        dataset_name: Optional[str] = None, # Can load from HF Hub
        dataset_split: Optional[str] = None, # Split to use if loading from HF Hub
        train_dataset: Optional[Dataset] = None, # Or provide a pre-loaded HF Dataset

        # --- Training Configuration (only used if calling train()) ---
        output_dir: str = "outputs_pipeline",
        per_device_train_batch_size: int = 2, gradient_accumulation_steps: int = 4,
        warmup_steps: int = 5, max_steps: Optional[int] = 60, num_train_epochs: Optional[int] = None,
        learning_rate: float = 2e-4, optim: str = "adamw_8bit", weight_decay: float = 0.01,
        lr_scheduler_type: str = "linear", seed: int = 3407, logging_steps: int = 1,
        report_to: str = "none", max_seq_length: int = 2048, dataset_num_proc: int = 4,
    ):
        """
        Initializes the Unsloth Vision Pipeline.

        Args:
            base_model_name: HF model identifier.
            load_in_4bit: Load in 4-bit.
            use_gradient_checkpointing: Gradient checkpointing setting.
            lora_adapter_path: Path to pre-trained LoRA adapters. If None, sets up for training.
            (PEFT Config Args): Parameters for PEFT adaptation if training.
            dataset_name: HF dataset identifier (used if train_dataset is None).
            dataset_split: Split to load (used if train_dataset is None).
            train_dataset: A pre-loaded Hugging Face Dataset object with 'question',
                           'answer', and 'image_link' columns. Takes precedence over
                           dataset_name/split.
            (Training Config Args): Parameters for SFTConfig.
        """
        print("Initializing UnslothVisionPipeline...")
        self.base_model_name = base_model_name
        self.lora_adapter_path = lora_adapter_path
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.input_train_dataset = train_dataset # Store the provided dataset
        self.output_dir = output_dir
        self.load_in_4bit = load_in_4bit
        self.bf16_supported = is_bf16_supported()
        self.push_to_hub = True # Default to True for saving adapters

        # Store PEFT and Training configurations
        self._peft_config = {
            "finetune_vision_layers": finetune_vision_layers, "finetune_language_layers": finetune_language_layers,
            "finetune_attention_modules": finetune_attention_modules, "finetune_mlp_modules": finetune_mlp_modules,
            "r": r, "lora_alpha": lora_alpha, "lora_dropout": lora_dropout, "bias": bias,
            "random_state": random_state, "use_rslora": use_rslora, "loftq_config": loftq_config,
        }
        self._training_config_input = {
            "per_device_train_batch_size": per_device_train_batch_size, "gradient_accumulation_steps": gradient_accumulation_steps,
            "warmup_steps": warmup_steps, "max_steps": max_steps, "num_train_epochs": num_train_epochs,
            "learning_rate": learning_rate, "optim": optim, "weight_decay": weight_decay,
            "lr_scheduler_type": lr_scheduler_type, "seed": seed, "logging_steps": logging_steps,
            "report_to": report_to, "max_seq_length": max_seq_length, "dataset_num_proc": dataset_num_proc,
            "output_dir": output_dir,
            "push_to_hub": self.push_to_hub
        }

        # --- Load Model and Tokenizer ---
        self._load_model_and_tokenizer(use_gradient_checkpointing)

        # --- Initialize other components ---
        self.processed_train_data: Optional[List[Dict]] = None # Will hold formatted data list
        self.trainer: Optional[SFTTrainer] = None
        # Initialize collator early, it doesn't depend on the dataset content
        self.data_collator = UnslothVisionDataCollator(self.model, self.tokenizer)
        print("Pipeline initialized.")

    def _load_model_and_tokenizer(self, use_gradient_checkpointing):
        """Loads the model and tokenizer based on initialization config."""
        print(f"Loading model: {self.base_model_name} {'with adapters from '+self.lora_adapter_path if self.lora_adapter_path else ''}")
        if self.lora_adapter_path:
            self.model, self.tokenizer = FastVisionModel.from_pretrained(
                model_name=self.lora_adapter_path, load_in_4bit=self.load_in_4bit,
                use_gradient_checkpointing=use_gradient_checkpointing,
            )
            print("Loaded model with pre-existing LoRA adapters.")
            FastVisionModel.for_inference(self.model)
        else:
            self.model, self.tokenizer = FastVisionModel.from_pretrained(
                model_name=self.base_model_name, load_in_4bit=self.load_in_4bit,
                use_gradient_checkpointing=use_gradient_checkpointing,
            )
            print("Loaded base model. Applying new PEFT adaptations...")
            self.model = FastVisionModel.get_peft_model(self.model, **self._peft_config)
            print("PEFT adaptations applied for potential training.")

    def _format_conversation_from_sample(self, sample: Dict) -> Optional[Dict[str, List[Dict]]]:
        """
        Loads image and formats a single data sample (dict) into the required
        conversational format for training.

        Args:
            sample: A dictionary containing 'question', 'answer', and 'image_link'.

        Returns:
            A dictionary with the 'messages' key, or None if image loading fails.
        """
        question = sample.get("question")
        answer = sample.get("answer")
        image_link = sample.get("image_link")

        if not all([question, answer, image_link]):
            print(f"Warning: Skipping sample due to missing 'question', 'answer', or 'image_link'. Sample: {sample}")
            return None

        # Load image using the helper function
        image = load_image_from_source(image_link)
        if image is None:
            # Warning is printed inside load_image_from_source
            return None # Skip this sample if image loading failed

        # Format according to the required structure
        conversation = [
            {"role": "user",
             "content": [
                 {"type": "text", "text": question}, # Use question from dataset
                 {"type": "image", "image": image}    # Use the loaded PIL image
             ]},
            {"role": "assistant",
             "content": [
                 {"type": "text", "text": answer}    # Use answer from dataset
             ]},
        ]
        return {"messages": conversation}

    def prepare_data(self) -> None:
        """Loads and preprocesses the training dataset."""
        if self.processed_train_data:
            print("Training data already prepared.")
            return

        dataset_to_process = None
        if self.input_train_dataset:
            print("Using provided pre-loaded dataset.")
            dataset_to_process = self.input_train_dataset
        elif self.dataset_name and self.dataset_split:
            print(f"Loading dataset '{self.dataset_name}' split '{self.dataset_split}' from Hugging Face Hub...")
            try:
                dataset_to_process = load_dataset(self.dataset_name, split=self.dataset_split)
            except Exception as e:
                print(f"Error loading dataset from Hub: {e}")
                raise # Re-raise the exception
        else:
            raise ValueError("Must provide either 'train_dataset' or ('dataset_name' and 'dataset_split') during initialization to prepare data.")

        if not isinstance(dataset_to_process, Dataset):
             raise TypeError(f"Expected dataset_to_process to be a datasets.Dataset, but got {type(dataset_to_process)}")

        # Validate required columns
        required_columns = ["question", "answer", "image_link"]
        if not all(col in dataset_to_process.column_names for col in required_columns):
             raise ValueError(f"Dataset must contain the columns: {required_columns}. Found: {dataset_to_process.column_names}")

        print("Formatting dataset and loading images...")
        # Iterate and format. Using a list comprehension for simplicity here.
        # For very large datasets, consider using dataset.map with careful error handling.
        formatted_data = []
        num_processed = 0
        num_skipped = 0
        for sample in dataset_to_process:
             formatted = self._format_conversation_from_sample(sample)
             if formatted:
                 formatted_data.append(formatted)
                 num_processed += 1
             else:
                 num_skipped += 1
             # Optional: Add progress indicator for large datasets
             # if (num_processed + num_skipped) % 100 == 0:
             #     print(f"Processed {num_processed+num_skipped}/{len(dataset_to_process)} samples...")

        self.processed_train_data = formatted_data
        print(f"Dataset preparation complete. Processed: {num_processed}, Skipped (due to image errors): {num_skipped}")
        if not self.processed_train_data:
             print("Warning: No valid data points were processed. Training cannot proceed.")


    def train(self) -> Optional[Dict[str, Any]]:
        """
        Configures and runs the training process using SFTTrainer.
        """
        if self.lora_adapter_path:
            print("Model was loaded with pre-existing adapters. Training is disabled.")
            return None

        if not self.processed_train_data:
            print("Preparing data before training...")
            self.prepare_data()
            if not self.processed_train_data: # Check if preparation failed or yielded no data
                 print("Data preparation failed or resulted in no usable data. Cannot start training.")
                 return None

        print("\n--- Configuring Training ---")
        sft_config_args = self._training_config_input.copy()
        sft_config_args.update({
            "fp16": not self.bf16_supported, "bf16": self.bf16_supported,
            "remove_unused_columns": False, "dataset_text_field": "",
            "dataset_kwargs": {"skip_prepare_dataset": True},
        })

        # Validate max_steps vs num_train_epochs
        if sft_config_args["max_steps"] is None and sft_config_args["num_train_epochs"] is None:
             raise ValueError("Must provide either 'max_steps' or 'num_train_epochs' in training config.")
        if sft_config_args["max_steps"] is not None and sft_config_args["num_train_epochs"] is not None:
             print("Warning: Both 'max_steps' and 'num_train_epochs' provided. 'max_steps' will take precedence.")
             if "num_train_epochs" in sft_config_args: del sft_config_args["num_train_epochs"]
        elif sft_config_args["num_train_epochs"] is not None:
             if "max_steps" in sft_config_args: del sft_config_args["max_steps"]
        else: # max_steps is provided, num_train_epochs is None
             if "num_train_epochs" in sft_config_args: del sft_config_args["num_train_epochs"]

        sft_config = SFTConfig(**sft_config_args)

        self.trainer = SFTTrainer(
            model=self.model, tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            train_dataset=self.processed_train_data, # Use the list of formatted dicts
            args=sft_config,
        )

        print("\n--- Starting Training ---")
        FastVisionModel.for_training(self.model)
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved before training starts.")

        try:
            trainer_stats = self.trainer.train()
        except Exception as e:
            print(f"An error occurred during training: {e}")
            raise

        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

        print("\n--- Training Finished ---")
        train_runtime = trainer_stats.metrics.get('train_runtime', None)
        if train_runtime:
            print(f"{train_runtime:.2f} seconds used for training.")
            print(f"{train_runtime/60:.2f} minutes used for training.")
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.\n")

        return trainer_stats.metrics

    def infer(
        self,
        image: Union[Image.Image, str], # Accept PIL image or path/URL string
        question: str,                  # User's question/instruction is now mandatory
        max_new_tokens: int = 128,
        temperature: float = 0.1,       # Lowered default temp for potentially more focused answers
        min_p: float = 0.05,
        stream: bool = True,
        **generation_kwargs
        ) -> Optional[str]:
        """
        Performs inference on a single image with a specific question.

        Args:
            image: A PIL Image object OR a string path/URL to an image.
            question: The text question/instruction for the model.
            max_new_tokens: Max tokens to generate.
            temperature: Sampling temperature.
            min_p: Minimum probability for nucleus sampling.
            stream: Whether to stream the output to stdout.
            **generation_kwargs: Additional arguments for model.generate().

        Returns:
            The generated text string if stream is False, otherwise None.
        """
        if not self.model or not self.tokenizer:
            print("Error: Model and tokenizer not loaded.")
            return None

        # Load image if a path/URL string is provided
        if isinstance(image, str):
            print(f"Loading image for inference from: {image}")
            img_input = load_image_from_source(image)
            if img_input is None:
                print("Error: Could not load image for inference.")
                return None
        elif isinstance(image, Image.Image):
            img_input = image
        else:
            raise TypeError("Input 'image' must be a PIL Image object or a string (path/URL).")


        FastVisionModel.for_inference(self.model)

        messages = [
            {"role": "user", "content": [
                {"type": "image"}, # Placeholder for the image
                {"type": "text", "text": question} # Use the provided question
            ]}
        ]
        # Do NOT add generation prompt here
        input_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False)

        inputs = self.tokenizer(
            img_input, # Use the loaded PIL image
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.model.device)

        generate_args = {
            "max_new_tokens": max_new_tokens, "use_cache": True,
            "temperature": temperature, "min_p": min_p, **generation_kwargs
        }

        print("\n--- Running Inference ---")
        if stream:
            print(f"Question: {question}")
            print("Model Generation (Streaming):")
            text_streamer = TextStreamer(self.tokenizer, skip_prompt=True)
            generate_args["streamer"] = text_streamer
            _ = self.model.generate(**inputs, **generate_args)
            print("\n--- Stream Finished ---")
            return None
        else:
            print(f"Question: {question}")
            print("Model Generation (Non-Streaming):")
            with torch.no_grad():
                 outputs = self.model.generate(**inputs, **generate_args)
            input_length = inputs["input_ids"].shape[1]
            generated_ids = outputs[0, input_length:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            print(generated_text)
            print("\n--- Inference Complete ---")
            return generated_text

    def save_adapters(self, save_directory: Optional[str] = None) -> None:
        """Saves the LoRA adapters and tokenizer locally."""
        if self.lora_adapter_path:
             print("Model was loaded with existing adapters. Cannot save new ones unless retrained.")
             return
        if not hasattr(self.model, "save_pretrained"):
             print("Model does not seem to have PEFT adapters loaded or is not a PEFT model.")
             return

        target_dir = save_directory if save_directory is not None else self.output_dir
        if not target_dir:
             print("Error: No save directory specified.")
             return

        print(f"Saving LoRA adapters and tokenizer to {target_dir}...")
        os.makedirs(target_dir, exist_ok=True)
        self.model.save_pretrained(target_dir)
        self.tokenizer.save_pretrained(target_dir)
        print("Adapters and tokenizer saved.")

# --------------------------------------------------------------
# Example Usage (Requires a dataset with question, answer, image_link)
# --------------------------------------------------------------

# Let's create a dummy dataset for demonstration
# In a real scenario, you would load this from HF Hub or a local file/object
dummy_data = [
    {"question": "What formula is shown in the image?",
     "answer": r"\frac{a}{b}", # Raw string for LaTeX
     "image_link": "https://placehold.co/600x100/FFF/000/?text=\frac{a}{b}"}, # Placeholder image URL
    {"question": "Transcribe the equation.",
     "answer": r"E = mc^2",
     "image_link": "https://placehold.co/600x100/EEE/000/?text=E = mc^2"},
    {"question": "Write the LaTeX for this integral.",
     "answer": r"\int_{0}^{1} x^2 dx",
     "image_link": "https://placehold.co/600x100/DDD/000/?text=\int_{0}^{1} x^2 dx"},
    {"question": "What is this chemical formula?",
     "answer": r"H_2O",
     "image_link": "https://placehold.co/600x100/CCC/000/?text=H_2O"},
    # Add a sample with a potentially bad link for testing robustness
    {"question": "What is this?",
     "answer": "Should be skipped",
     "image_link": "https://invalid.url/image.jpg"}
]
dummy_dataset = Dataset.from_list(dummy_data)

# --- Option 1: Train a new model using the dummy dataset ---
print("\n" + "="*50)
print("Example 1: Initializing Pipeline for Training (Dummy Data)")
print("="*50)
try:
    training_pipeline = UnslothVisionPipeline(
        base_model_name="unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit", # Or choose a smaller one if needed
        load_in_4bit=True, # Set to False if you have more VRAM
        train_dataset=dummy_dataset, # Pass the pre-loaded dummy dataset
        output_dir="dummy_finetune_adapters",
        max_steps=10, # Very few steps for quick demo
        logging_steps=1,
        per_device_train_batch_size=1, # Small batch size for dummy data / low VRAM
        gradient_accumulation_steps=2,
    )

    # Training automatically calls prepare_data()
    training_metrics = training_pipeline.train()
    if training_metrics:
        print("\nTraining completed. Metrics:", training_metrics)
        training_pipeline.save_adapters() # Saves to 'dummy_finetune_adapters'

        # --- Inference using the TRAINED pipeline ---
        print("\n--- Inference after Training (Dummy Data) ---")
        # Use one of the dummy image links for inference
        test_image_link = dummy_data[1]["image_link"]
        test_question = "Transcribe the math expression shown." # Use a relevant question
        print(f"Inferring with Image: {test_image_link}")
        print(f"Question: {test_question}")
        # Image is loaded from the link inside infer()
        training_pipeline.infer(image=test_image_link, question=test_question, stream=True)

except Exception as e:
    print(f"\nError during Training Pipeline execution: {e}")
    # Print traceback for detailed debugging if needed
    # import traceback
    # traceback.print_exc()


# --- Option 2: Load the dummy finetuned model for inference ---
print("\n" + "="*50)
print("Example 2: Initializing Pipeline for Inference using Saved Dummy Adapters")
print("="*50)

adapter_save_path = "dummy_finetune_adapters"

if os.path.exists(adapter_save_path):
    try:
        inference_pipeline = UnslothVisionPipeline(
            base_model_name="unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
            lora_adapter_path=adapter_save_path, # Load the dummy adapters
        )

        print("\n--- Inference using Loaded Dummy Adapters ---")
        test_image_link_2 = dummy_data[2]["image_link"]
        test_question_2 = "What is the integral expression?"

        print(f"Inferring with Image: {test_image_link_2}")
        # Run inference (non-streaming)
        generated_text = inference_pipeline.infer(
            image=test_image_link_2,
            question=test_question_2,
            stream=False,
            temperature=0.01 # Very low temp for deterministic output
        )
        print("\nNon-Streamed Generated Text:")
        if generated_text:
             print(generated_text)
             # Try rendering if it looks like LaTeX
             if '\\' in generated_text or '^' in generated_text or '_' in generated_text:
                 try:
                     display(Math(generated_text))
                 except Exception as render_e:
                     print(f"(Could not render generated LaTeX: {render_e})")

    except Exception as e:
        print(f"\nError during Inference Pipeline execution: {e}")
        # import traceback
        # traceback.print_exc()
else:
    print(f"Skipping Inference Pipeline example because adapter directory '{adapter_save_path}' not found.")
    print("(Run the training example first to generate adapters).")