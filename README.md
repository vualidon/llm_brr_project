# LLM Brainrot Project

A comprehensive data pipeline for scraping, processing, and training vision-language models on Fandom wiki content.

## 🚀 Overview

This project automates the process of:

1. Crawling and scraping content from Fandom wikis
2. Generating question-answer pairs from the content using Google's Gemini API
3. Pushing the processed data to Hugging Face datasets
4. Training vision-language models (Qwen2.5-VL) using Unsloth

The entire pipeline is orchestrated using Dagster, providing reliable scheduling and monitoring.

## 📋 Features

- **Automated Web Scraping**: Crawls Fandom wikis and extracts content and images
- **Database Integration**: Stores scraped content in Supabase
- **QA Generation**: Creates question-answer pairs from content using Google's Gemini API
- **Hugging Face Integration**: Pushes processed data to Hugging Face datasets
- **Vision-Language Model Training**: Fine-tunes Qwen2.5-VL models using Unsloth
- **Orchestration**: Manages the entire pipeline with Dagster

## 🛠️ Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/llm_brr_project.git
cd llm_brr_project
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables by creating a `.env` file:

```
# Supabase Configuration
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
SUPABASE_PAGE_TABLE=your_table_name

# Firecrawl API
FIRECRAWL_API_KEY=your_firecrawl_api_key

# Google Gemini API
GOOGLE_API_KEYS_CSV=key1,key2,key3

# Hugging Face
HF_TOKEN=your_huggingface_token
HF_DATASET_REPO_ID=your_username/your_dataset_name
```

## 🏃‍♂️ Usage

### Running the Pipeline

To run the complete pipeline using Dagster:

```bash
dagster dev -f MyDataPipeline.py
```

This will start the Dagster UI (usually at <http://127.0.0.1:3000>), where you can:

- Manually trigger the pipeline
- Set up schedules
- Monitor pipeline execution
- View logs and errors

### Individual Components

You can also run individual components of the pipeline:

#### Fandom Crawler

```python
from data.crawler import FandomCrawler

crawler = FandomCrawler()
scraped_data = crawler.crawl_all_pages("https://brainrot.fandom.com/wiki/Special:AllPages")
crawler.save_to_supabase(scraped_data)
```

#### QA Generator

```python
from data.QAGenerater import QAGenerator

qa_generator = QAGenerator()
qa_pairs = qa_generator.process_scraped_data(scraped_data)
```

#### Hugging Face Pusher

```python
from data.HFPusher import HFPusher

pusher = HFPusher()
pusher.push_data(items_with_qa_pairs)
```

#### Vision Model Training

```python
from training.MyUnslothTrainer import UnslothVisionPipeline

pipeline = UnslothVisionPipeline(
    base_model_name="unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
    dataset_name="your_username/your_dataset_name",
    output_dir="trained_model"
)
pipeline.train()
pipeline.save_adapters()
```

## 📊 Project Structure

The project follows a modular structure:

```
llm_brr_project/
├── data/                      # Data processing modules
│   ├── crawler.py             # Fandom wiki crawler
│   ├── QAGenerater.py         # QA pair generator using Gemini
│   └── HFPusher.py            # Hugging Face dataset pusher
├── deploy/                    # Deployment modules
│   └── MyDeployer.py          # Model deployment utilities
├── training/                  # Training modules
│   └── MyUnslothTrainer.py    # Vision model training
├── MyDataPipeline.py          # Main Dagster pipeline
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## 🔄 Pipeline Flow

1. **Crawl Fandom Pages**: Scrape content from Fandom wikis
2. **Generate QA Pairs**: Create question-answer pairs from the content
3. **Push to Hugging Face**: Store the processed data in a Hugging Face dataset
4. **Train Vision Model**: Fine-tune a vision-language model on the data

## 🎬 Demo

Check out the demo video of the project in action:

[Demo Video](./demo_video.mp4)

## Huggingface link

**You can check out the dataset**: [here](https://huggingface.co/datasets/thangvip/brr_training_dataset)

## 📝 License

[Your License Here]

## 🙏 Acknowledgements

- [Dagster](https://dagster.io/) for pipeline orchestration
- [Unsloth](https://github.com/unslothai/unsloth) for efficient model training
- [Hugging Face](https://huggingface.co/) for model hosting and datasets
- [Google Gemini](https://ai.google.dev/) for QA generation
