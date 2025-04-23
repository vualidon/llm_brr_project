import requests
from bs4 import BeautifulSoup
import logging
from firecrawl import FirecrawlApp
from datetime import datetime
from supabase import create_client, Client, PostgrestAPIResponse, ClientOptions
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FandomCrawler:
    """
    A class to crawl Fandom 'Special:AllPages', check against Supabase,
    and scrape new pages using Firecrawl.
    """
    def __init__(self):
        """
        Initializes the FandomCrawler, loading environment variables
        and setting up Supabase and Firecrawl clients.
        """
        load_dotenv()
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        self.supabase_table = os.getenv("SUPABASE_PAGE_TABLE")
        self.firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
        self.base_url_prefix = "https://brainrot.fandom.com" # Define base prefix here

        if not all([self.supabase_url, self.supabase_key, self.supabase_table, self.firecrawl_api_key]):
            raise ValueError("One or more environment variables (SUPABASE_URL, SUPABASE_KEY, SUPABASE_PAGE_TABLE, FIRECRAWL_API_KEY) are not set.")

        try:
            options = ClientOptions(schema="public")
            self.supabase_client: Client = create_client(self.supabase_url, self.supabase_key, options=options)
            logging.info("Supabase client initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize Supabase client: {e}")
            raise # Re-raise exception as client is essential

        try:
            self.firecrawl_app = FirecrawlApp(api_key=self.firecrawl_api_key)
            logging.info("FirecrawlApp initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize FirecrawlApp: {e}")
            raise # Re-raise exception as app is essential

    def check_title_exists(self, title: str) -> bool:
        """
        Checks if a page with the given title already exists in the Supabase table.

        Args:
            title: The page title to check.

        Returns:
            True if the title exists, False otherwise.
        """
        try:
            response: PostgrestAPIResponse = self.supabase_client.table(self.supabase_table)\
                .select('id', count='exact')\
                .eq('title', title)\
                .execute()

            if response.count is not None and response.count > 0:
                logging.info(f"Title '{title}' already exists in the database.")
                return True

        except Exception as e:
            logging.error(f"Error checking title '{title}' in Supabase: {e}")
            # Treat error as "possibly exists" to be safe and avoid duplicate processing attempts
            return True

        return False # Title does not exist or check resulted in non-error empty response

    def crawl_all_pages(self, url: str) -> list[dict]:
        """
        Crawls the 'Special:AllPages' page, identifies new pages not in Supabase,
        scrapes them using Firecrawl, and returns the data for the new pages found
        (up to the first successful scrape due to the original 'break' statement).

        Args:
            url: The URL of the 'Special:AllPages' page.

        Returns:
            A list containing dictionaries of the scraped data for new pages found.
            Returns an empty list if no new pages are found or an error occurs.
            Note: Due to the 'break' in the original logic, this might only return
            data for the *first* new page successfully scraped on that run.
        """
        scraped_pages_data = [] # Initialize list to store data for *all* scraped pages in this run

        try:
            logging.info(f"Attempting to fetch URL: {url}")
            response = requests.get(url, timeout=15) # Increased timeout slightly
            response.raise_for_status()
            logging.info(f"Successfully fetched URL: {url}")

            soup = BeautifulSoup(response.content, 'html.parser')
            content_area = soup.find('div', class_='mw-allpages-body')
            if not content_area:
                content_area = soup.find('div',class_="mw-body-content")
                list_items = content_area.find_all("li", class_="category-page__member") # Specific selector
                
                for link in list_items:
                    
                    link = link.find('a') # Find the anchor tag within the list item
                    href = link.get('href')
                    title = link.get('title')
                    if title and href:
                        logging.debug(f"Processing link: Title='{title}', Href='{href}'")
                        if self.check_title_exists(title):
                            logging.info(f"Skipping existing title: {title}")
                            continue
                        # --- Scrape new page ---
                        page_url = f"{self.base_url_prefix}{href}" # Construct full URL
                        logging.info(f"Found new title: '{title}'. Attempting scrape from: {page_url}")
                        try:
                            # --- Using Firecrawl as requested ---
                            response_fc = self.firecrawl_app.scrape_url(
                                url=page_url,
                                # params={'pageOptions': {'onlyMainContent': True}}, # Example of potential param if needed later
                                formats=[ 'markdown', 'html' ], # As per original code
                                include_tags=[ '#mw-content-text' ], # As per original code
                            )
                            # --- End Firecrawl usage ---

                            logging.info(f"Firecrawl response for {title}: {response_fc}") # Log the raw response for debugging if needed

                            if response_fc and response_fc.markdown: # Check if markdown content exists
                                # Extract metadata safely
                                og_image = response_fc.metadata.get('og:image', None) if response_fc.metadata else None

                                page_data = {
                                    'title': title,
                                    'url': page_url,
                                    'content': response_fc.markdown,
                                    'image_link': og_image,
                                    'crawled_at': datetime.now().isoformat(),
                                    # Consider adding 'html_content': response_fc.html if needed
                                }
                                scraped_pages_data.append(page_data)
                                logging.info(f"Successfully scraped and prepared data for '{title}'.")

                                # --- Original break statement ---
                                # This break will cause the loop to exit after the *first*
                                # successful scrape. If you want to scrape all new pages

                                # found on the 'AllPages' list during one run, remove this break.
                                # break
                                # --- End original break ---
                            else:

                                logging.warning(f"Firecrawl scrape for '{title}' completed but returned no markdown content. Skipping.")
                        except Exception as fc_error:
                            logging.error(f"Error during Firecrawl scrape for '{title}' at {page_url}: {fc_error}")
                            # Decide whether to continue to the next link or stop
                            continue
                    else:
                        logging.warning(f"Skipping link with missing title or href: {link}")
                
                
                return scraped_pages_data # Return the list of successfully scraped pages (possibly only one due to break)
            # --- Original logic continued ---
            # If the content area was not found, try a different selector   

            if content_area:
                list_items = content_area.select('ul.mw-allpages-chunk li a') # Specific selector
                logging.info(f"Found {len(list_items)} potential page links.")

                if not list_items:
                    logging.warning(f"No page links found within the content area using selector 'ul.mw-allpages-chunk li a' on {url}")
                    return []

                for link in list_items:
                    # Ensure it's not a redirect link
                    link_classes = link.get('class', []) # Use .get with default
                    if "mw-redirect" not in link_classes:
                        title = link.get_text(strip=True)
                        href = link.get('href')

                        if title and href:
                            logging.debug(f"Processing link: Title='{title}', Href='{href}'")

                            if self.check_title_exists(title):
                                logging.info(f"Skipping existing title: {title}")
                                continue # Skip to the next link

                            # --- Scrape new page ---
                            page_url = f"{self.base_url_prefix}{href}" # Construct full URL
                            logging.info(f"Found new title: '{title}'. Attempting scrape from: {page_url}")

                            try:
                                # --- Using Firecrawl as requested ---
                                response_fc = self.firecrawl_app.scrape_url(
                                    url=page_url,
                                    # params={'pageOptions': {'onlyMainContent': True}}, # Example of potential param if needed later
                                    formats=[ 'markdown', 'html' ], # As per original code
                                    include_tags=[ '#mw-content-text' ], # As per original code
                                )
                                # --- End Firecrawl usage ---

                                logging.debug(f"Firecrawl response for {title}: {response_fc}") # Log the raw response for debugging if needed

                                if response_fc and response_fc.markdown: # Check if markdown content exists
                                    # Extract metadata safely
                                    og_image = response_fc.metadata.get('og:image', None) if response_fc.metadata else None

                                    page_data = {
                                        'title': title,
                                        'url': page_url,
                                        'content': response_fc.markdown,
                                        'image_link': og_image,
                                        'crawled_at': datetime.now().isoformat(),
                                        # Consider adding 'html_content': response_fc.html if needed
                                    }
                                    scraped_pages_data.append(page_data)
                                    logging.info(f"Successfully scraped and prepared data for '{title}'.")

                                    # --- Original break statement ---
                                    # This break will cause the loop to exit after the *first*
                                    # successful scrape. If you want to scrape all new pages
                                    # found on the 'AllPages' list during one run, remove this break.
                                    # break
                                    # --- End original break ---

                                else:
                                     logging.warning(f"Firecrawl scrape for '{title}' completed but returned no markdown content. Skipping.")

                            except Exception as fc_error:
                                logging.error(f"Error during Firecrawl scrape for '{title}' at {page_url}: {fc_error}")
                                # Decide whether to continue to the next link or stop
                                continue # Let's continue to try other links

                        else:
                             logging.warning(f"Skipping link with missing title or href: {link}")
                    else:
                        logging.debug(f"Skipping redirect link: {link.get_text(strip=True)}")

            else:
                logging.error(f"Could not find the main content area on {url}. Check selectors '.mw-allpages-body' or '#mw-content-text'.")

        except requests.exceptions.RequestException as req_err:
            logging.error(f"HTTP Request failed for {url}: {req_err}")
        except Exception as e:
            logging.error(f"An unexpected error occurred in crawl_all_pages for {url}: {e}")

        return scraped_pages_data # Return the list of successfully scraped pages (possibly only one due to break)
    
    def save_to_supabase(self, data: list[dict]):
        """
        Saves the scraped data to Supabase.

        Args:
            data: A list of dictionaries containing the scraped data.
        """
        try:
            if data:
                response = self.supabase_client.table(self.supabase_table).insert(data).execute()
                if response.data:
                    logging.info(f"Successfully inserted {len(response.data)} records into Supabase.")
                else:
                    logging.error(f"Failed to insert data into Supabase. Response: {response}")
            else:
                logging.warning("No data to save to Supabase.")
        except Exception as db_error:
            logging.error(f"Error inserting data into Supabase: {db_error}")
    
    

# Example of how to use the class
if __name__ == "__main__":
    target_url = "https://brainrot.fandom.com/wiki/Category:AI_Brainrot"

    try:
        crawler = FandomCrawler() # Initialize the crawler
        newly_scraped_data = crawler.crawl_all_pages(target_url) # Run the crawl
        print(newly_scraped_data)
        # if newly_scraped_data:
        #     print("\n--- Successfully Scraped Pages ---")
        #     # If you were processing multiple pages, you would loop here
        #     for page in newly_scraped_data:
        #          print(f"Title: {page['title']}")
        #          print(f"URL: {page['url']}")
        #          print(f"Image Link: {page.get('image_link', 'N/A')}") # Use .get for safety
        #          print(f"Crawled At: {page['crawled_at']}")
        #          print(f"Content Preview (first 100 chars): {page['content'][:100]}...")
        #          print("---")
            # Save the scraped data to Supabase
            # crawler.save_to_supabase(newly_scraped_data)
            # Here you would typically insert the data into Supabase
            # Example (uncomment and adapt if needed):
            # try:
            #     insert_response = crawler.supabase_client.table(crawler.supabase_table).insert(newly_scraped_data).execute()
            #     if insert_response.data:
            #         logging.info(f"Successfully inserted {len(insert_response.data)} records into Supabase.")
            #     else:
            #         # Handle potential insertion errors (check insert_response for details)
            #         logging.error(f"Failed to insert data into Supabase. Response: {insert_response}")
            # except Exception as db_error:
            #     logging.error(f"Error inserting data into Supabase: {db_error}")

        # else:
        #     print("No new page titles were scraped in this run. Check logs for details.")

    except ValueError as ve:
        # Handles missing environment variables during initialization
        logging.error(f"Initialization failed: {ve}")
    except Exception as main_err:
        # Handles other potential errors during crawler setup or execution
        logging.error(f"An error occurred in the main execution block: {main_err}")