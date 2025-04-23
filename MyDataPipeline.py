# MyDagsterPipeline.py

import os
import logging
from dagster import (
    op,
    job,
    get_dagster_logger,
    OpExecutionContext,
    ScheduleDefinition, # Import ScheduleDefinition
    Definitions         # Import Definitions
)
from typing import Any # Import Any for type hinting

from data.crawler import FandomCrawler
from data.QAGenerater import QAGenerator
from data.HFPusher import HFPusher


@op
def crawl_fandom_pages(context: OpExecutionContext) -> list[dict]:
    """
    Dagster op to crawl Fandom 'Special:AllPages', check against Supabase,
    and scrape new pages using Firecrawl.
    """
    logger = get_dagster_logger()
    logger.info("Starting Fandom crawl...")
    target_url = "https://brainrot.fandom.com/wiki/Special:AllPages" # Or get from config

    try:
        crawler = FandomCrawler()
        newly_scraped_data = crawler.crawl_all_pages(target_url)

        if newly_scraped_data:
            logger.info(f"Successfully scraped {len(newly_scraped_data)} new page(s).")
            logger.info("Saving scraped data to Supabase...")
            crawler.save_to_supabase(newly_scraped_data)
            logger.info("Data saved to Supabase.")
        else:
            logger.info("No new pages were scraped in this run.")

        return newly_scraped_data
    except Exception as e:
        logger.error(f"Error during Fandom crawl or save: {e}")
        raise

@op
def generate_qa_pairs(context: OpExecutionContext, scraped_data: list[dict]) -> list[dict]:
    """
    Dagster op to generate Question/Answer pairs for the scraped data
    using Google Gemini with API key rotation.
    """
    logger = get_dagster_logger()
    logger.info(f"Starting QA generation for {len(scraped_data)} scraped item(s)...")

    if not scraped_data:
        logger.warning("No scraped data received. Skipping QA generation.")
        return []

    try:
        qa_generator = QAGenerator()
        results_with_qas = qa_generator.process_scraped_data(scraped_data)
        logger.info("Finished QA generation.")
        successful_qas = sum(1 for item in results_with_qas if item.get('qa_pairs'))
        logger.info(f"Successfully generated QA pairs for {successful_qas} out of {len(results_with_qas)} items.")
        return results_with_qas
    except Exception as e:
        logger.error(f"Error during QA generation: {e}")
        raise

@op
def push_to_huggingface(context: OpExecutionContext, data_with_qas: list[dict]):
    """
    Dagster op to process the data with QAs and push/append it to a
    Hugging Face Dataset repository.
    """
    logger = get_dagster_logger()
    logger.info(f"Starting push to Hugging Face for {len(data_with_qas)} item(s)...")

    if not data_with_qas:
        logger.warning("No data with QAs received. Skipping Hugging Face push.")
        return

    items_to_push = [item for item in data_with_qas if item.get('qa_pairs')]
    if not items_to_push:
         logger.warning("Received data, but none contained generated QA pairs. Skipping Hugging Face push.")
         return

    logger.info(f"Found {len(items_to_push)} items with QA pairs to potentially push.")

    try:
        pusher = HFPusher()
        pusher.push_data(items_to_push)
        logger.info("Finished pushing data to Hugging Face Hub.")
        # Optionally return the repo ID or status
        # return pusher.repo_id
    except Exception as e:
        logger.error(f"Error during Hugging Face push: {e}")
        raise



@job
def data_to_deployment_pipeline():
    """
    A Dagster job orchestrating the crawling, QA generation,
    Hugging Face push, and VLLM deployment initialization.
    """
    scraped_results = crawl_fandom_pages()
    qa_results = generate_qa_pairs(scraped_results)
    push_results = push_to_huggingface(qa_results)
    # deploy_vllm_lora(push_results)

# --- Schedule Definition ---

# Define a schedule to run the job once a month
# Cron string "0 0 1 * *" means: at 00:00 (midnight) on day-of-month 1.
monthly_schedule = ScheduleDefinition(
    job=data_to_deployment_pipeline,
    cron_schedule="0 0 1 * *", # Run at midnight on the 1st of every month
    execution_timezone="UTC", # Specify timezone, e.g., UTC or "America/New_York"
)

# --- Definitions object for Dagster UI/Daemon ---
# This makes the job and schedule visible to Dagster tools
defs = Definitions(
    jobs=[data_to_deployment_pipeline],
    schedules=[monthly_schedule]
    # Add sensors here if needed
    # sensors=[...]
    # Add resources here if you refactor config
    # resources={...}
)

# --- Running Instructions ---
# To run this locally using Dagster UI:
# 1. Ensure dagster, dagster-webserver, and all pipeline dependencies are installed.
# 2. Ensure your .env file with all necessary API keys and config is accessible.
# 3. Run the Dagster UI/development server:
#    dagster dev -f MyDagsterPipeline.py
# 4. Open the Dagster UI (usually http://127.0.0.1:3000).
# 5. You can manually launch the 'data_to_deployment_pipeline' job from the UI.
# 6. To enable the schedule:
#    - Go to the 'Deployment' or 'Overview' section in the UI.
#    - Find the 'monthly_schedule' under 'Schedules'.
#    - Toggle the schedule 'On'.
#    - NOTE: The Dagster daemon process must be running for schedules to trigger automatically.
#      In production, you'd typically run `dagster-daemon run` as a separate service.
#      When using `dagster dev`, a daemon might run as part of that process, but check Dagster's documentation for robust scheduling setups.
