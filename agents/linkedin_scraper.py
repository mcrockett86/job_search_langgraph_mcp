from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import NoSuchElementException

import pandas as pd
import os
import time

# load environment variables from .env file (e.g. OpenAI API key, LinkedIn username, password ...)
from dotenv import load_dotenv
load_dotenv('dotenv.env')


def scrape_linkedin_job_urls(target_jobs_urls:list, max_pages=30) -> pd.DataFrame:
    """
    Scrapes recommended job URLs from LinkedIn.
    """
    # Get LinkedIn credentials from environment variables or user input
    email = os.environ.get("LINKEDIN_EMAIL") or input("Enter your LinkedIn email: ")
    password = os.environ.get("LINKEDIN_PASSWORD") or input("Enter your LinkedIn password: ")

    # Set up the Chrome driver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)

    # Navigate to the LinkedIn login page
    driver.get("https://www.linkedin.com/login")
    time.sleep(2)

    # Log in
    driver.find_element(By.ID, "username").send_keys(email)
    driver.find_element(By.ID, "password").send_keys(password)
    driver.find_element(By.XPATH, "//button[@type='submit']").click()
    time.sleep(5)  # Wait for the page to load

    job_urls = []
    for target_jobs_url in target_jobs_urls:

        # Navigate to the recommended jobs page
        driver.get(target_jobs_url)
        time.sleep(5)

        page_number = 1

        while page_number <= max_pages:  # Scrape the first 10 pages
            print(f"Scraping page {page_number}...")

            # Scroll to the bottom of the page to load all jobs
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)

            # Extract job links
            job_listings = driver.find_elements(By.XPATH, "//a[contains(@href, '/jobs/view/')]")
            for job in job_listings:
                job_url = job.get_attribute("href")
                if job_url not in job_urls:
                    job_urls.append(job_url)

            # Go to the next page
            try:
                next_button = driver.find_element(By.XPATH, f"//button[@aria-label='Page {page_number + 1}']")
                next_button.click()
                time.sleep(5)
                page_number += 1
                
            except Exception as error:
                print(error)
                break

    driver.quit()


    # strip down to the base view for the job link
    job_urls_cleaned = [link.split("/?")[0].replace('https', 'http') for link in job_urls]

    # create pandas dataframe from job_urls list
    df = pd.DataFrame({'job_url': job_urls_cleaned} )

    return df