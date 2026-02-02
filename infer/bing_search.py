"""
Web Search and Content Extraction Module

This module provides utilities for:
- Web search using Jina, Bing, and Serper APIs
- URL content extraction (HTML and PDF)
- Snippet-based context extraction
"""

import re
import http
import json
import requests
import string
import time
import pdfplumber
import ssl
import asyncio
import warnings
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from io import BytesIO
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from datetime import datetime, timezone
from fake_useragent import UserAgent
from nltk.tokenize import sent_tokenize
from typing import Optional, Tuple, Dict, List

import urllib3


# ============================================================================
# Constants
# ============================================================================

MAX_RESPONSE_SIZE = 500000  # 500KB limit to prevent hanging on huge pages
DEFAULT_TEXT_RETURN_LENGTH = 8000  # Default text length when no snippet provided (match Search-o1)
URL_FETCH_TIMEOUT = 60  # Per-URL timeout in seconds
OVERALL_FETCH_TIMEOUT = 600  # Overall fetch operation timeout


# ============================================================================
# SSL and Warning Configuration
# ============================================================================

ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


# ============================================================================
# HTTP Session Configuration
# ============================================================================

user_agent = UserAgent()

DEFAULT_HEADERS = {
    'User-Agent': user_agent.random,
    'Referer': 'https://www.google.com/',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
    'Keep-Alive': 'timeout=5, max=100',
    'Upgrade-Insecure-Requests': '1'
}

# Initialize session with connection pooling
session = requests.Session()
session.headers.update(DEFAULT_HEADERS)

adapter = requests.adapters.HTTPAdapter(
    pool_connections=100,
    pool_maxsize=100,
    max_retries=3,
    pool_block=False
)
session.mount('http://', adapter)
session.mount('https://', adapter)


# ============================================================================
# Text Processing Utilities
# ============================================================================

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


# ============================================================================
# Snippet Context Extraction
# ============================================================================

def extract_snippet_with_context(
    full_text: str, 
    snippet: Optional[str], 
    context_chars: int = 2500
) -> Tuple[bool, str]:
    """
    Extract the sentence that best matches the snippet and its context from the full text.

    Args:
        full_text: The full text extracted from the webpage.
        snippet: The snippet to match (can be None).
        context_chars: Number of characters to include before and after the snippet.

    Returns:
        Tuple of (success: bool, extracted_context: str)
    """
    try:
        # Handle None or empty full_text
        if not full_text or full_text.startswith("Error"):
            return False, full_text[:context_chars] if full_text else ""
        
        full_text = full_text[:50000]

        # Handle None or empty snippet
        if not snippet or not snippet.strip():
            return False, full_text[:context_chars]

        snippet = snippet.lower()
        snippet = remove_punctuation(snippet)
        snippet_words = set(snippet.split())
        
        # Handle empty snippet after processing
        if not snippet_words:
            return False, full_text[:context_chars]

        best_sentence = None
        best_f1 = 0.2

        sentences = sent_tokenize(full_text)

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
            return False, full_text[:context_chars]
            
    except Exception as e:
        return False, f"Failed to extract snippet context due to {str(e)}"


# ============================================================================
# URL Content Extraction
# ============================================================================

def extract_pdf_text(url: str) -> str:
    """
    Extract text from a PDF URL.

    Args:
        url: URL of the PDF file.

    Returns:
        Extracted text content or error message.
    """
    try:
        response = session.get(url, timeout=20)
        if response.status_code != 200:
            return f"Error: Unable to retrieve the PDF (status code {response.status_code})"
        
        with pdfplumber.open(BytesIO(response.content)) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text
        
        cleaned_text = ' '.join(full_text.split()[:600])
        return cleaned_text
        
    except requests.exceptions.Timeout:
        return "Error: Request timed out after 20 seconds"
    except Exception as e:
        return f"Error: {str(e)}"


def extract_text_from_url(
    url: str, 
    use_jina: bool = False, 
    jina_api_key: Optional[str] = None, 
    snippet: Optional[str] = None
) -> str:
    """
    Extract text from a URL. If a snippet is provided, extract the context related to it.

    Args:
        url: URL of a webpage or PDF.
        use_jina: Whether to use Jina for extraction.
        jina_api_key: Jina API key.
        snippet: The snippet to search for.

    Returns:
        Extracted text or context.
    """
    try:
        if use_jina:
            jina_headers = {
                'Authorization': f'Bearer {jina_api_key}',
                'X-Return-Format': 'markdown',
            }
            response = requests.get(
                f'https://r.jina.ai/{url}', 
                headers=jina_headers, 
                verify=False, 
                timeout=10
            ).text
            
            # Remove URLs from markdown
            pattern = r"\(https?:.*?\)|\[https?:.*?\]"
            text = re.sub(pattern, "", response)
            text = text.replace('---', '-').replace('===', '=')
            text = text.replace('   ', ' ').replace('   ', ' ')
        else:
            response = session.get(url, timeout=20, verify=False)
            response.raise_for_status()
            
            content_type = response.headers.get('Content-Type', '')
            if 'pdf' in content_type:
                return extract_pdf_text(url)
            
            # Limit response size to prevent hanging on huge pages
            response_text = response.text[:MAX_RESPONSE_SIZE]
            
            try:
                soup = BeautifulSoup(response_text, 'lxml')
            except Exception:
                soup = BeautifulSoup(response_text, 'html.parser')
            
            text = soup.get_text(separator=' ', strip=True)

        if snippet:
            success, context = extract_snippet_with_context(text, snippet)
            if success:
                return context
            else:
                return text
        else:
            # Return more context when no snippet provided
            return text[:DEFAULT_TEXT_RETURN_LENGTH]
            
    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}"
    except requests.exceptions.ConnectionError:
        return "Error: Connection error occurred"
    except requests.exceptions.Timeout:
        return "Error: Request timed out after 20 seconds"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


def fetch_page_content(
    urls: List[str], 
    max_workers: int = 4,
    use_jina: bool = False, 
    jina_api_key: Optional[str] = None, 
    snippets: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """
    Concurrently fetch content from multiple URLs.

    Args:
        urls: List of URLs to scrape.
        max_workers: Maximum number of concurrent threads.
        use_jina: Whether to use Jina for extraction.
        jina_api_key: Jina API key.
        snippets: A dictionary mapping URLs to their respective snippets.

    Returns:
        Dictionary mapping URLs to extracted content or context.
    """
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                extract_text_from_url, 
                url, 
                use_jina, 
                jina_api_key, 
                snippets.get(url) if snippets else None
            ): url
            for url in urls
        }
        try:
            for future in concurrent.futures.as_completed(futures, timeout=OVERALL_FETCH_TIMEOUT):
                url = futures[future]
                try:
                    data = future.result(timeout=URL_FETCH_TIMEOUT)
                    results[url] = data
                except concurrent.futures.TimeoutError:
                    results[url] = f"Error fetching {url}: Request timed out"
                    future.cancel()
                except Exception as exc:
                    results[url] = f"Error fetching {url}: {exc}"
        except concurrent.futures.TimeoutError:
            # Handle case where as_completed itself times out
            print("Overall fetch operation timed out after 600 seconds")
            for future, url in futures.items():
                if url not in results:
                    results[url] = "Error: Overall operation timed out"
                    future.cancel()
    return results


async def fetch_page_content_async(
    urls: List[str], 
    use_jina: bool = False, 
    jina_api_key: Optional[str] = None, 
    snippets: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """
    Concurrently fetch content from multiple URLs using asyncio.
    
    Alternative to fetch_page_content that uses asyncio instead of ThreadPoolExecutor.

    Args:
        urls: List of URLs to scrape.
        use_jina: Whether to use Jina for extraction.
        jina_api_key: Jina API key.
        snippets: A dictionary mapping URLs to their respective snippets.

    Returns:
        Dictionary mapping URLs to extracted content.
    """
    async def run_one(url, use_jina, jina_api_key, snippet):
        return await asyncio.to_thread(
            extract_text_from_url, url, use_jina, jina_api_key, snippet
        )
    
    tasks = [
        run_one(url, use_jina, jina_api_key, snippets.get(url) if snippets else None) 
        for url in urls
    ]
    results = await asyncio.gather(*tasks)
    return {url: result for url, result in zip(urls, results)}


# ============================================================================
# Rate Limit Handling
# ============================================================================

def sleep_until(timestamp_str: str) -> None:
    """
    Sleep until the specified UTC timestamp.

    Args:
        timestamp_str: ISO 8601 formatted timestamp string in UTC (e.g., "2025-08-09T06:14:07.036Z")
    """
    try:
        if timestamp_str.endswith('Z'):
            timestamp_str = timestamp_str[:-1] + '+00:00'

        future_time = datetime.fromisoformat(timestamp_str)

        if future_time.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware.")

        now_utc = datetime.now(timezone.utc)
        sleep_duration_seconds = (future_time - now_utc).total_seconds() + 1

        if sleep_duration_seconds > 0:
            time.sleep(sleep_duration_seconds)

    except (ValueError, TypeError):
        pass


# ============================================================================
# Jina Search API
# ============================================================================

def jina_web_search(
    query: str, 
    subscription_key: str, 
    depth: int = 0
) -> List[Dict[str, str]]:
    """
    Perform web search using Jina API.
    
    Args:
        query: Search query string.
        subscription_key: Jina API key.
        depth: Recursion depth for retries.
    
    Returns:
        List of search results with title, url, and snippet.
    """
    # URL encode the query for better search results
    search_url = f"https://s.jina.ai/{query.replace(' ', '+')}"
    search_headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {subscription_key}',
        'X-Respond-With': 'no-content'
    }
    
    try:
        response = requests.get(search_url, headers=search_headers, verify=False, timeout=10)
        response = json.loads(response.text)
        
        if response.get("data") is None:
            message = response.get("message", "")
            if 'No search results available' in message:
                return []
            elif 'rate limit exceeded' in message:
                if "retryAfterDate" in response:
                    sleep_until(response["retryAfterDate"])
                raise Exception(f"Rate limit exceeded")
            else:
                raise Exception(f"Jina search API error: {message}")
        
        results = []
        for e in response["data"]:
            results.append({
                "title": e.get('title', ''),
                "url": e.get('url', ''),
                "snippet": e.get('description', '')
            })
        return results
        
    except Exception:
        if depth < 128:
            time.sleep(0.1)
            return jina_web_search(query, subscription_key, depth + 1)
        else:
            return []


async def jina_web_search_all_queries(
    queries: List[str], 
    jina_api_key: Optional[str] = None
) -> Dict[str, List[Dict[str, str]]]:
    """
    Perform web search for multiple queries concurrently.
    
    Args:
        queries: List of search queries.
        jina_api_key: Jina API key.
    
    Returns:
        Dictionary mapping queries to their search results.
    """
    async def run_one(query):
        return await asyncio.to_thread(jina_web_search, query, jina_api_key, depth=0)
    
    tasks = [run_one(query) for query in queries]
    results = await asyncio.gather(*tasks)
    return {query: result for query, result in zip(queries, results)}


# ============================================================================
# Azure Bing Search API
# ============================================================================

def azure_bing_search(
    query: str, 
    subscription_key: str, 
    mkt: str, 
    top_k: int, 
    depth: int = 0
) -> List[Dict[str, str]]:
    """
    Perform web search using Azure Bing API.
    
    Args:
        query: Search query string.
        subscription_key: Azure API subscription key.
        mkt: Market code (e.g., "en-US").
        top_k: Number of results to return.
        depth: Recursion depth for retries.
    
    Returns:
        List of search results with title, link, and snippet.
    """
    params = {'q': query, 'mkt': mkt, 'count': top_k}
    headers = {'Ocp-Apim-Subscription-Key': subscription_key}

    try:
        response = requests.get(
            "https://api.bing.microsoft.com/v7.0/search", 
            headers=headers, 
            params=params
        )
        json_response = response.json()
        
        results = []
        for e in json_response['webPages']['value']:
            results.append({
                "title": e['name'],
                "link": e['url'],
                "snippet": e['description']
            })
        return results
        
    except Exception:
        if depth < 8:
            time.sleep(0.1)
            return azure_bing_search(query, subscription_key, mkt, top_k, depth + 1)
        return []


# ============================================================================
# Serper Google Search API
# ============================================================================

def serper_google_search(
    query: str, 
    serper_api_key: str,
    top_k: int,
    region: str = "us",
    lang: str = "en",
    depth: int = 0
) -> List[Dict[str, str]]:
    """
    Perform web search using Serper Google Search API.
    
    Args:
        query: Search query string.
        serper_api_key: Serper API key.
        top_k: Number of results to return.
        region: Region code (e.g., "us").
        lang: Language code (e.g., "en").
        depth: Recursion depth for retries.
    
    Returns:
        List of organic search results.
    """
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
        
        if not data:
            raise Exception("The google search API is temporarily unavailable.")

        if "organic" not in data:
            raise Exception(f"No results found for query: '{query}'.")
        
        return data["organic"]
        
    except Exception:
        if depth < 8:
            time.sleep(0.1)
            return serper_google_search(
                query, serper_api_key, top_k, region, lang, depth + 1
            )
        return []


# ============================================================================
# Search Result Processing
# ============================================================================

def extract_relevant_info(search_results: Dict) -> List[Dict[str, str]]:
    """
    Extract relevant information from Bing search results.

    Args:
        search_results: JSON response from the Bing Web Search API.

    Returns:
        List of dictionaries containing the extracted information with
        id, title, url, site_name, date, snippet, and context fields.
    """
    useful_info = []
    
    if 'webPages' in search_results and 'value' in search_results['webPages']:
        for id, result in enumerate(search_results['webPages']['value']):
            info = {
                'id': id + 1,
                'title': result.get('name', ''),
                'url': result.get('url', ''),
                'site_name': result.get('siteName', ''),
                'date': result.get('datePublished', '').split('T')[0] if result.get('datePublished') else '',
                'snippet': result.get('snippet', ''),
                'context': ''  # Reserved field to be filled later
            }
            useful_info.append(info)
    
    return useful_info


def filter_search_results_by_relevance(
    results: List[Dict[str, str]], 
    query: str,
    min_relevance_score: float = 0.1
) -> List[Dict[str, str]]:
    """
    Filter search results by relevance to the query using F1 score.
    
    Args:
        results: List of search results.
        query: Original search query.
        min_relevance_score: Minimum F1 score threshold.
    
    Returns:
        Filtered list of search results sorted by relevance.
    """
    query_words = set(remove_punctuation(query.lower()).split())
    
    scored_results = []
    for result in results:
        # Combine title and snippet for relevance scoring
        text = f"{result.get('title', '')} {result.get('snippet', '')}"
        text_words = set(remove_punctuation(text.lower()).split())
        
        score = f1_score(query_words, text_words)
        if score >= min_relevance_score:
            result['relevance_score'] = score
            scored_results.append(result)
    
    # Sort by relevance score descending
    scored_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
    
    return scored_results
