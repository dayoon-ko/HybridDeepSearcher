import os
import http
import json
import requests
import re
import string
import time
import pdfplumber
import ssl
import asyncio
from tqdm import tqdm
from io import BytesIO
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from datetime import datetime, timezone
from fake_useragent import UserAgent
from nltk.tokenize import sent_tokenize
from requests.exceptions import Timeout
from typing import Optional, Tuple, List, Dict, Union

import urllib3
from urllib.parse import quote_plus
import warnings

ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


# ----------------------- Custom Headers -----------------------
user_agent = UserAgent()
headers = {
    'User-Agent': user_agent.random,
    'Referer': 'https://www.google.com/',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}

# Initialize session with connection pooling
session = requests.Session()
session.headers.update(headers)
# Optimize connection pooling
adapter = requests.adapters.HTTPAdapter(
    pool_connections=100,  # Number of connection pools
    pool_maxsize=100,      # Maximum number of connections in pool
    max_retries=3,         # Retry failed requests
    pool_block=False       # Don't block when pool is full
)
session.mount('http://', adapter)
session.mount('https://', adapter)



def remove_punctuation(text: str) -> str:
    """Remove punctuation from the text."""
    return text.translate(str.maketrans("", "", string.punctuation))

def f1_score(true_set: set, pred_set: set) -> float:
    """Calculate the F1 score between two sets of words."""
    intersection = len(true_set.intersection(pred_set))
    if not intersection:
        return 0.0
    precision = intersection / float(len(pred_set))
    recall = intersection / float(len(true_set))
    return 2 * (precision * recall) / (precision + recall)

def extract_snippet_with_context(full_text: str, snippet: str, context_chars: int = 2500) -> Tuple[bool, str]:
    """
    Extract the sentence that best matches the snippet and its context from the full text.

    Args:
        full_text (str): The full text extracted from the webpage.
        snippet (str): The snippet to match.
        context_chars (int): Number of characters to include before and after the snippet.

    Returns:
        Tuple[bool, str]: The first element indicates whether extraction was successful, the second element is the extracted context.
    """
    try:
        full_text = full_text[:50000]

        snippet = snippet.lower()
        snippet = remove_punctuation(snippet)
        snippet_words = set(snippet.split())

        best_sentence = None
        best_f1 = 0.2

        # sentences = re.split(r'(?<=[.!?]) +', full_text)  # Split sentences using regex, supporting ., !, ? endings
        sentences = sent_tokenize(full_text)  # Split sentences using nltk's sent_tokenize

        for sentence in sentences:
            key_sentence = sentence.lower()
            key_sentence = remove_punctuation(key_sentence)
            sentence_words = set(key_sentence.split())
            f1 = f1_score(snippet_words, sentence_words)
            if f1 > best_f1:
                best_f1 = f1
                best_sentence = sentence

        if best_sentence:
            para_start = full_text.find(best_sentence)
            para_end = para_start + len(best_sentence)
            start_index = max(0, para_start - context_chars)
            end_index = min(len(full_text), para_end + context_chars)
            context = full_text[start_index:end_index]
            return True, context
        else:
            # If no matching sentence is found, return the first context_chars*2 characters of the full text
            return False, full_text[:context_chars]
    except Exception as e:
        return False, f"Failed to extract snippet context due to {str(e)}"

def extract_text_from_url(url, use_jina=False, jina_api_key=None, snippet: Optional[str] = None):
    """
    Extract text from a URL. If a snippet is provided, extract the context related to it.

    Args:
        url (str): URL of a webpage or PDF.
        use_jina (bool): Whether to use Jina for extraction.
        snippet (Optional[str]): The snippet to search for.

    Returns:
        str: Extracted text or context.
    """
    try:
        if use_jina:
            jina_headers = {
                'Authorization': f'Bearer {jina_api_key}',
                'X-Return-Format': 'markdown',
                # 'X-With-Links-Summary': 'true'
            }
            response = requests.get(f'https://r.jina.ai/{url}', headers=jina_headers, verify=False, timeout=10).text # verify=False
            # Remove URLs
            pattern = r"\(https?:.*?\)|\[https?:.*?\]"
            text = re.sub(pattern, "", response).replace('---','-').replace('===','=').replace('   ',' ').replace('   ',' ')
        else:
            response = session.get(url, timeout=10, verify=False)  # Set timeout to 10 seconds
            response.raise_for_status()  # Raise HTTPError if the request failed
            # Determine the content type
            content_type = response.headers.get('Content-Type', '')
            if 'pdf' in content_type:
                # If it's a PDF file, extract PDF text
                return extract_pdf_text(url)
            # Try using lxml parser, fallback to html.parser if unavailable
            try:
                soup = BeautifulSoup(response.text, 'lxml')
            except Exception:
                print("lxml parser not found or failed, falling back to html.parser")
                soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)

        if snippet:
            success, context = extract_snippet_with_context(text, snippet)
            if success:
                return context
            else:
                return text
        else:
            # If no snippet is provided, return directly
            return text[:1024]
    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}"
    except requests.exceptions.ConnectionError as e:
        return "Error: Connection error occurred"
    except requests.exceptions.Timeout:
        return "Error: Request timed out after 10 seconds"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

async def fetch_page_content(urls, use_jina=False, jina_api_key=None, snippets: Optional[dict] = None):
    """
    Concurrently fetch content from multiple URLs with optimized performance.

    Args:
        urls (list): List of URLs to scrape.
        max_workers (int): Maximum number of concurrent threads.
        use_jina (bool): Whether to use Jina for extraction.
        snippets (Optional[dict]): A dictionary mapping URLs to their respective snippets.

    Returns:
        dict: A dictionary mapping URLs to the extracted content or context.
    """
    results = {}
    
    async def run_one(url, use_jina, jina_api_key, snippet):
        return await asyncio.to_thread(extract_text_from_url, url, use_jina, jina_api_key, snippet)
    
    tasks = [run_one(url, use_jina, jina_api_key, snippets.get(url) if snippets else None) for url in urls]
    results = await asyncio.gather(*tasks)
    return {url: result for url, result in zip(urls, results)}


def sleep_until(timestamp_str: str):
    """
    Sleeps until the specified UTC timestamp.

    Args:
        timestamp_str: An ISO 8601 formatted timestamp string in UTC,
                       e.g., "2025-08-09T06:14:07.036Z".
    """
    try:
        # Python's fromisoformat before 3.11 doesn't handle 'Z' for UTC.
        if timestamp_str.endswith('Z'):
            timestamp_str = timestamp_str[:-1] + '+00:00'

        future_time = datetime.fromisoformat(timestamp_str)

        # Ensure we are comparing timezone-aware datetimes
        if future_time.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware.")

        now_utc = datetime.now(timezone.utc)

        sleep_duration_seconds = (future_time - now_utc).total_seconds() + 1

        if sleep_duration_seconds > 0:
            time.sleep(sleep_duration_seconds)

    except (ValueError, TypeError) as e:
        print(f"Error parsing timestamp: {e}")


async def jina_web_search_all_queries(queries, jina_api_key=None):
    results = {}
    async def run_one(query):
        return await asyncio.to_thread(jina_web_search, query, jina_api_key, depth=0)
    tasks = [run_one(query) for query in queries]
    results = await asyncio.gather(*tasks)
    return {query: result for query, result in zip(queries, results)}


def jina_web_search(query, subscription_key, depth=0):
    url = 'https://s.jina.ai/'
    params = {'q': query}
    search_headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {subscription_key}',
        'X-Respond-With': 'no-content'
    }
    results = []
    try:
        response = requests.get(url, headers=search_headers, params=params, verify=False)
        response = json.loads(response.text) 
        if response["data"] is None:
            if 'No search results available' in response["message"]:
                return []
            elif 'rate limit exceeded' in response["message"]:
                sleep_until(response["retryAfterDate"])
                raise Exception(f"Try again after {response['retryAfterDate']}")
            else:
                raise Exception(f"Jina search API error: {response['message']}")
        
        for e in response["data"]:
            results.append({
                "title": e['title'],
                "url": e['url'],
                "snippet": e['description'] 
            })
    except Exception as e:
        if depth < 128:
            time.sleep(0.1)
            return jina_web_search(query, subscription_key, depth+1) 
        else:
            return []
    return results



def azure_bing_search(query, subscription_key, mkt, top_k, depth=0):
    params = {'q': query, 'mkt': mkt, 'count': top_k}
    headers = {'Ocp-Apim-Subscription-Key': subscription_key}

    results = []

    try:
        response = requests.get("https://api.bing.microsoft.com/v7.0/search", headers=headers, params=params)
        json_response = response.json()
        for e in json_response['webPages']['value']:
            results.append({
                "title": e['name'],
                "link": e['url'],
                "snippet": e['description']
            })
    except Exception as e:
        print(f"Bing search API error: {e}")
        if depth < 8:
            time.sleep(0.1)
            return azure_bing_search(query, subscription_key, mkt, top_k, depth+1)
    return results


def serper_google_search(
        query, 
        serper_api_key,
        top_k,
        region="us",
        lang="en",
        depth=0
    ):
    try:
        conn = http.client.HTTPSConnection("google.serper.dev")
        payload = json.dumps({
                "q": query,
                "num": top_k,
                "gl": region,
                "hl": lang,
            })
        headers = {
            'X-API-KEY': serper_api_key,
            'Content-Type': 'application/json'
        }
        conn.request("POST", "/search", payload, headers)
        res = conn.getresponse()
        data = json.loads(res.read().decode("utf-8"))
        
        print(data)
        if not data:
            raise Exception("The google search API is temporarily unavailable, please try again later.")

        if "organic" not in data:
            raise Exception(f"No results found for query: '{query}'. Use a less specific query.")
        else:
            results = data["organic"]
            print("search success")
            return results
    except Exception as e:
        print(f"Serper search API error: {e}")
        if depth < 8:
            time.sleep(0.1)
            return serper_google_search(query, serper_api_key, top_k, region, lang, depth=depth+1)
    print("search failed")
    return []


def extract_pdf_text(url):
    """
    Extract text from a PDF.

    Args:
        url (str): URL of the PDF file.

    Returns:
        str: Extracted text content or error message.
    """
    try:
        response = session.get(url, timeout=20)  # Set timeout to 20 seconds
        if response.status_code != 200:
            return f"Error: Unable to retrieve the PDF (status code {response.status_code})"
        
        # Open the PDF file using pdfplumber
        with pdfplumber.open(BytesIO(response.content)) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text
        
        # Limit the text length
        cleaned_text = ' '.join(full_text.split()[:600])
        return cleaned_text
    except requests.exceptions.Timeout:
        return "Error: Request timed out after 20 seconds"
    except Exception as e:
        return f"Error: {str(e)}"



