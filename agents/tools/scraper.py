from langchain.tools import tool
from typing import Any
import requests
from bs4 import BeautifulSoup
import html2text
import os
import pdb


# Reference: https://stackademic.com/blog/web-scraping-with-llms-using-langchain
async def scraper_extract_html_from_url(url:str) -> str | None:    
    """Navigate to a URL, Fetch HTML content, and convert it to plain text, excluding certain tags."""

    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Ubuntu; Firefox=41)'
    }

    try:
        # Fetch HTML content from the URL using requests
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()  # Raise an exception for bad responses (4xx and 5xx)
    
        # Parse HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        excluded_tagNames = ['footer', 'nav', 'a', 'script']
        
        # Exclude elements with class names 'footer' and 'navbar'
        excluded_tags = excluded_tagNames or []  # Default to an empty list if not provided
        for tag_name in excluded_tags:
            for unwanted_tag in soup.find_all(tag_name):
                unwanted_tag.extract()


        # Convert HTML to plain text using html2text
        text_content = html2text.html2text(str(soup))
        #await ctx.debug(f"Data Fetched From {url}: {text_content}[0:100]")  # Log the first 100 characters of the text content
        return text_content

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from {url}: {e}")
        return ""