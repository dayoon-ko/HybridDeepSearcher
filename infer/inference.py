import os
import json
import asyncio
import logging
import argparse
import datasets

from utils import (
    Generator, 
    extract_answer, 
    extract_between, 
    get_response_from_llm, 
    prompt_for_webpage_to_reasonchain_instruction
)
from bing_search import (
    jina_web_search_all_queries, 
    fetch_page_content, 
    extract_snippet_with_context
)
from typing import List, Dict


# ============================================================================
# Constants
# ============================================================================

PROMPT = """**Task Instruction:**

You are a reasoning assistant equipped with web search capabilities to accurately answer the user's questions.

Follow these steps:
1. **Clearly identify** the specific information you need to answer the user's question.
2. **Perform a web search** for the required information by writing your queries as follows:
```
<|begin_search_queries|>
Your search queries here (multiple queries can be placed together seperated by ";\n")
<|end_search_queries|>
```
3. Review the provided search results.
4. If additional information is still required, repeat step 2 with new queries.
5. Once all relevant information has been gathered, use your reasoning abilities to synthesize a clear, concise, and accurate answer.

**Remember:**
* Clearly separate each search query.
* Combine multiple queries into a single search action when they can be run simultaneously.
"""

BEGIN_SEARCH_QUERY = "<|begin_search_queries|>"
END_SEARCH_QUERY = "<|end_search_queries|>"
BEGIN_SEARCH_RESULT = "<|begin_search_results|>"
END_SEARCH_RESULT = "<|end_search_results|>"

DATASET_PATHS = {
    "browse_comp": "datasets/BrowseComp/test.json",
    "med_browse_comp": "datasets/MedBrowseComp/test.json",
    "musique": "datasets/MuSiQue/dev.json",
    "fanoutqa": "datasets/FanOutQA/dev.json",
    "frames": "datasets/FRAMES/test.json",
}


# ============================================================================
# Argument Parsing
# ============================================================================

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", "-d", required=True, 
                        choices=["musique", "browse_comp", "hle", "gpqa", "fanoutqa", 
                                 "cwq", "hotpotqa", "med_browse_comp", "multihopqa", "frames"]) 
    parser.add_argument("--dataset_split", "-sp", default="train")
    parser.add_argument("--model_id_or_path", "-m", default="dayoon/HybridDeepSearcher-GRPO")
    parser.add_argument("--apply_chat", type=bool, default=True)
    parser.add_argument("--use_jina", action="store_true", 
                        help="Whether to use jina for extracting text from urls")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--max_doc_len", type=int, default=1024) 
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=10, help="Retrieve the top k documents")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--start_idx", "-s", type=int)
    parser.add_argument("--end_idx", "-e", type=int)
    parser.add_argument("--api_model", "-am", default="Qwen/Qwen3-32B")
    parser.add_argument("--api_model_max_tokens", type=int, default=2048)
    parser.add_argument("--api_model_temperature", type=float, default=0.7)
    parser.add_argument("--api_model_top_p", type=float, default=0.8)
    parser.add_argument("--api_model_base_url", default="http://localhost:9001/v1")
    parser.add_argument("--api_model_key", default=os.environ.get("OPENROUTER_API_KEY", ""))
    parser.add_argument("--jina_api_key", default=os.environ.get("JINA_API_KEY", ""))
    parser.add_argument("--max_iteration", type=int, default=10)
    parser.add_argument("--batch_size_for_processing", type=int, default=16)
    args = parser.parse_args()
    return args


# ============================================================================
# Logger Setup
# ============================================================================

def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


# ============================================================================
# Dataset Loading
# ============================================================================

def load_dataset(args):
    """Load dataset from file or HuggingFace."""
    if args.dataset_name in DATASET_PATHS:
        with open(DATASET_PATHS[args.dataset_name], "r") as f:
            dataset = json.load(f)
    else:
        dataset = datasets.load_dataset(args.dataset_name)[args.dataset_split]
        dataset = [item for item in dataset]
    
    # Select subset of dataset
    if args.start_idx is not None and args.end_idx is not None:
        dataset = dataset[args.start_idx:args.end_idx]
        
    # Split dataset into batches
    if args.batch_size_for_processing > 0:
        batches = [
            dataset[i:i+args.batch_size_for_processing] 
            for i in range(0, len(dataset), args.batch_size_for_processing)
        ]
        return batches
    else:
        return [dataset]


# ============================================================================
# Cache Management
# ============================================================================

def get_cache(output_dir):
    """Load search and URL caches from disk."""
    search_cache = {}
    url_cache = {}
    
    search_cache_path = f"{output_dir}/search_cache.json"
    url_cache_path = f"{output_dir}/url_cache.json"
    
    if os.path.exists(search_cache_path):
        with open(search_cache_path, "r") as f:
            search_cache = json.load(f)
    
    if os.path.exists(url_cache_path):
        with open(url_cache_path, "r") as f:
            url_cache = json.load(f)
    
    return search_cache, url_cache


def save_cache(output_dir, search_cache, url_cache):
    """Save search and URL caches to disk."""
    with open(f"{output_dir}/search_cache.json", "w") as f:
        json.dump(search_cache, f, indent=2)
    with open(f"{output_dir}/url_cache.json", "w") as f:
        json.dump(url_cache, f, indent=2)


# ============================================================================
# Item Initialization
# ============================================================================

def normalize_question_field(item):
    """Normalize question field name across different datasets."""
    if "question" not in item:
        if "Question" in item:
            item["question"] = item["Question"]
            del item["Question"]
        elif "query" in item:
            item["question"] = item["query"]
            del item["query"]
    return item


def initialize_item(item, get_prompt_fn):
    """Initialize item with required fields for processing."""
    item = normalize_question_field(item)
    item["finished"] = False
    item["prompt"] = get_prompt_fn(item)
    item["output"] = ""
    item["output_history"] = []
    item["search_count"] = []
    item["executed_search_queries"] = []
    item["related_info_analysis"] = []
    item["relevant_info"] = {}
    return item


# ============================================================================
# Response Processing
# ============================================================================

def process_model_response(item, response, search_cache, top_k):
    """
    Process model response to extract search queries and determine next action.
    
    Returns:
        tuple: (needs_search: bool, relevant_info: dict, queries_to_search: set)
    """
    # Fix missing </think> tag
    if "</think>" not in response:
        response = response.replace(
            f"\n\n{BEGIN_SEARCH_QUERY}", 
            f"\n</think>\n\n{BEGIN_SEARCH_QUERY}"
        )
    
    # Update item with response
    search_queries = extract_between(response, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)
    item["prompt"] += response
    item["output"] += response
    item["output_history"].append(response)

    # If no search queries, extract final answer and mark as finished
    if search_queries is None:    
        item["finished"] = True
        item["generated_answer"] = extract_answer(response)
        return False, {}, set()
    
    # Process search queries
    relevant_info = {}
    queries_to_search = set()
    
    for search_query in search_queries:
        # Skip already executed queries
        if search_query in set(item['executed_search_queries']):
            repeat_msg = (
                f"\n{BEGIN_SEARCH_RESULT}\nYou have searched this query. "
                "Please refer to previous results.\n"
                f"{END_SEARCH_RESULT}\n"
            )
            relevant_info[search_query] = repeat_msg
            continue
        
        # Use cached results if available
        if search_query in search_cache:
            results = search_cache[search_query]
            relevant_info[search_query] = results[:top_k]
        else:
            queries_to_search.add(search_query)
            relevant_info[search_query] = None
    
    return True, relevant_info, queries_to_search


def update_item_with_search_results(item, relevant_info, all_search_results, search_cache, top_k):
    """Update item with search results and return URLs to fetch."""
    # Update relevant_info with search results
    for query, info in relevant_info.items():
        if info is None:
            results = all_search_results[query]
            relevant_info[query] = results[:top_k]
            search_cache[query] = results
    
    item['relevant_info'].update(relevant_info)
    item['search_count'].append(len(relevant_info))
    item['executed_search_queries'].extend(relevant_info.keys())
    
    # Extract URLs and snippets
    urls_to_fetch = []
    snippets = {}
    
    for _, results in relevant_info.items():
        if isinstance(results, str):
            continue
        for info in results:
            if 'url' in info:
                urls_to_fetch.append(info['url'])
                if 'snippet' in info:
                    snippets[info['url']] = info['snippet']
    
    return urls_to_fetch, snippets


# ============================================================================
# Reasoning Truncation
# ============================================================================

def truncate_reasoning_steps(output):
    """Truncate previous reasoning steps for context window management."""
    all_reasoning_steps = output.replace('\n\n', '\n').split("\n")
    truncated_prev_reasoning = ""
    
    for i, step in enumerate(all_reasoning_steps):
        truncated_prev_reasoning += f"Step {i + 1}: {step}\n\n"
    
    prev_steps = truncated_prev_reasoning.split('\n\n')
    
    if len(prev_steps) <= 5:
        truncated_prev_reasoning = '\n\n'.join(prev_steps)
    else:
        truncated_prev_reasoning = ''
        for i, step in enumerate(prev_steps):
            should_include = (
                i == 0 or 
                i >= len(prev_steps) - 4 or 
                BEGIN_SEARCH_QUERY in step or 
                BEGIN_SEARCH_RESULT in step
            )
            if should_include:
                truncated_prev_reasoning += step + '\n\n'
            else:
                if truncated_prev_reasoning[-len('\n\n...\n\n'):] != '\n\n...\n\n':
                    truncated_prev_reasoning += '...\n\n'
    
    return truncated_prev_reasoning.strip('\n')


# ============================================================================
# Document Formatting
# ============================================================================

def format_documents_for_summarization(relevant_info, executed_queries, url_cache, max_doc_len):
    """Format documents from search results for batch summarization."""
    formatted_documents = {}
    
    for sub_query in executed_queries:
        sub_relevant_info = relevant_info[sub_query]
        
        # Skip if already a string (e.g., "already searched" message)
        if isinstance(sub_relevant_info, str):
            formatted_documents[sub_query] = sub_relevant_info
            continue
        
        doc_str = ""
        for i, doc_info in enumerate(sub_relevant_info):
            url = doc_info.get('url', "")
            raw_context = url_cache.get(url, "")
            
            # Clean snippet
            snippet = doc_info.get("snippet")
            if snippet:
                snippet = snippet.replace('<b>', '').replace('</b>', '')
            doc_info['snippet'] = snippet
            
            # Extract context around snippet
            success, filtered_context = extract_snippet_with_context(
                raw_context, snippet, context_chars=max_doc_len
            )
            context = filtered_context if success else raw_context[:max_doc_len]
            doc_info['context'] = context
            
            doc_str += f"**Web Page {i + 1}:**\n"
            doc_str += json.dumps(doc_info, ensure_ascii=False, indent=2) + "\n"
        
        formatted_documents[sub_query] = doc_str
    
    return formatted_documents


# ============================================================================
# Webpage Analysis (API Model)
# ============================================================================

async def generate_webpage_to_reasonchain_batch(
        args,
        logger,
        original_questions: List[str],
        prev_reasonings: List[str],
        search_queries: List[List[str]],
        documents: Dict[str, str],
        dataset_name: str,
        max_tokens: int = 2048,
    ) -> List[Dict[str, str]]:
    """Batch process webpages to extract relevant information using API model."""
    
    logger.info(f"API model: {args.api_model}")
    
    def get_reasonchain_prompt(prev_reasoning, query, document):
        prompt = prompt_for_webpage_to_reasonchain_instruction.format(
            prev_reasoning=prev_reasoning, 
            search_query=query, 
            document=document
        )
        return [
            {"role": "system", "content": "You are a helpful assistant."}, 
            {"role": "user", "content": prompt}
        ]
    
    # Prepare prompts for each query
    all_user_prompts = {}
    for prev_reasoning, seq_queries in zip(prev_reasonings, search_queries):
        for query in seq_queries:
            prompt = get_reasonchain_prompt(prev_reasoning, query, documents[query])
            all_user_prompts[query] = prompt

    async def run_one(idx, query, prompt):
        result = await asyncio.to_thread(
            get_response_from_llm,
            args,
            prompt,
            query_for_the_prompt=query,
            return_type="tuple",
            depth=0,
        )
        return result
    
    # Execute all requests concurrently
    tasks = [
        asyncio.create_task(run_one(idx, q, p)) 
        for idx, (q, p) in enumerate(all_user_prompts.items())
    ]
    outputs = await asyncio.gather(*tasks)
    
    # Extract answers from outputs
    extracted_infos = {
        query: extract_answer(output, mode='infogen') 
        for query, output in outputs
    }
    return extracted_infos


# ============================================================================
# Final Answer Generation
# ============================================================================

def prepare_item_for_final_answer(item):
    """Prepare item prompt/output for final answer generation."""
    prompt = item["prompt"].strip() 
    output = item["output"].strip()
    
    # Remove trailing search tokens
    tokens_to_strip = [END_SEARCH_QUERY, BEGIN_SEARCH_QUERY, END_SEARCH_RESULT, BEGIN_SEARCH_RESULT]
    
    while True:
        stripped = False
        for token in tokens_to_strip:
            if prompt.endswith(token):
                prompt = prompt[:-len(token)].strip()
                output = output[:-len(token)].strip()
                stripped = True
                break
        if not stripped:
            break
    
    prompt = prompt.strip()
    output = output.strip()
    
    if not prompt.endswith("</think>"):
        final_answer_prefix = "\n</think>\n\n**Final Answer:**\n\\boxed{"
        prompt += final_answer_prefix
        output += final_answer_prefix
        item["output_history"].append(final_answer_prefix)
    
    item["prompt"] = prompt 
    item["output"] = output
    return item


def generate_final_answers(generator, batch, args):
    """Generate final answers for items that haven't finished naturally."""
    items_to_generate_final_answer = [
        item for item in batch 
        if not item["finished"] or "generated_answer" not in item or item["generated_answer"] == ""
    ]
    
    if not items_to_generate_final_answer:
        return
    
    # Prepare items
    items_to_generate_final_answer = [
        prepare_item_for_final_answer(item) 
        for item in items_to_generate_final_answer
    ]
    
    # Generate responses
    prompts = [item["prompt"] for item in items_to_generate_final_answer]
    responses = generator.generate(
        prompts, 
        max_tokens=args.max_tokens, 
        apply_chat=False,
        enable_thinking=False,
    )
    
    # Update items with responses
    for item, response in zip(items_to_generate_final_answer, responses):
        item["prompt"] += response
        item["output"] += response
        item["output_history"][-1] += response
        item["finished"] = True
        item["generated_answer"] = response.strip("}")


# ============================================================================
# Main Processing Loop
# ============================================================================

def process_batch_iteration(
    args, logger, generator, batch, items_remained, 
    search_cache, url_cache
):
    """Process one iteration of the batch inference loop."""
    
    # Initialize batch variables
    batch_relevant_info = []
    batch_original_questions = []
    batch_prev_reasonings = []
    batch_search_queries = []
    batch_documents = {}
    batch_sequences = []
    all_urls_to_fetch = set()
    all_queries_to_search = set()
    url_snippets = {}
    
    # Generate model responses
    prompts = [item["prompt"] for item in items_remained]
    responses = generator.generate(
        prompts, 
        max_tokens=args.max_tokens, 
        stop=[END_SEARCH_QUERY], 
        apply_chat=False,
        enable_thinking=True,
    )
    
    # Process responses
    items_remained_ = []
    responses_ = []
    
    for item, response in zip(items_remained, responses):
        needs_search, relevant_info, queries_to_search = process_model_response(
            item, response, search_cache, args.top_k
        )
        
        if not needs_search:
            continue
        
        items_remained_.append(item)
        responses_.append(response)
        batch_relevant_info.append(relevant_info)
        all_queries_to_search.update(queries_to_search)
    
    items_remained = items_remained_
    responses = responses_
    
    # Execute search queries
    all_search_results = {}
    if all_queries_to_search:
        logger.info(f"Executing {len(all_queries_to_search)} queries...")
        all_search_results = asyncio.run(
            jina_web_search_all_queries(
                list(all_queries_to_search),
                jina_api_key=args.jina_api_key,
            )
        )
    
    # Process search results and collect URLs
    for item, response, relevant_info in zip(items_remained, responses, batch_relevant_info):
        urls_to_fetch, snippets = update_item_with_search_results(
            item, relevant_info, all_search_results, search_cache, args.top_k
        )
        
        # Filter URLs that are not cached
        urls_to_fetch_filtered = [u for u in urls_to_fetch if u not in url_cache]
        
        for url in urls_to_fetch_filtered:
            all_urls_to_fetch.add(url)
            url_snippets[url] = snippets.get(url, "")
        
        # Collect data for batch processing
        truncated_prev_reasoning = truncate_reasoning_steps(item['output'])
        batch_original_questions.append(item['question'])
        batch_prev_reasonings.append(truncated_prev_reasoning)
        batch_search_queries.append(relevant_info.keys())
        batch_sequences.append(item)
    
    # Fetch all URLs (synchronous with ThreadPoolExecutor for timeout control)
    if all_urls_to_fetch:
        logger.info(f"Fetching {len(all_urls_to_fetch)} URLs...")
        try:
            fetched_contents = fetch_page_content(
                list(all_urls_to_fetch),
                use_jina=args.use_jina,
                jina_api_key=args.jina_api_key,
                snippets=url_snippets
            )
            logger.info(f"Fetched {len(fetched_contents)} URLs successfully.")
        except Exception as e:
            logger.info(f"Error during batch URL fetching: {e}")
            fetched_contents = {url: f"Error fetching URL: {e}" for url in all_urls_to_fetch}
        
        for url, content in fetched_contents.items():
            url_cache[url] = content
    
    # Format documents for summarization
    for relevant_info, executed_queries in zip(batch_relevant_info, batch_search_queries):
        formatted_documents = format_documents_for_summarization(
            relevant_info, executed_queries, url_cache, args.max_doc_len
        )
        batch_documents.update(formatted_documents)
    
    # Batch summarization
    if batch_sequences:
        logger.info(f"Batch summarizing {len(batch_sequences)} sequences...")
        webpage_analyses = asyncio.run(
            generate_webpage_to_reasonchain_batch(
                args,
                logger,
                original_questions=batch_original_questions,
                prev_reasonings=batch_prev_reasonings,
                search_queries=batch_search_queries,
                documents=batch_documents,
                dataset_name=args.dataset_name,
                max_tokens=args.api_model_max_tokens
            )
        )
        logger.info("Batch summarization completed.")

        # Assign outputs to sequences
        for seq, queries in zip(batch_sequences, batch_search_queries):
            combined_analysis = ""
            for query in queries:
                analysis = webpage_analyses[query]
                combined_analysis += f"{query}: {analysis}\n"
                seq["related_info_analysis"].append(analysis)
            
            append_text = f"\n\n{BEGIN_SEARCH_RESULT}\n{combined_analysis.strip()}\n{END_SEARCH_RESULT}\n\n<think>\n"
            seq['prompt'] += append_text
            seq['output'] += append_text
    
    return items_remained


# ============================================================================
# Main Function
# ============================================================================

def main():
    args = get_args()
    logger = setup_logger()
    
    # Setup output paths
    output_dir = f"{args.output_dir}/{args.dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/output.json"
    
    logger.info(f"Output file path: {output_path}")
    
    # Load dataset and caches
    dataset = load_dataset(args)
    search_cache, url_cache = get_cache(output_dir)
    
    # Initialize generator
    generator = Generator(model_id_or_path=args.model_id_or_path)
    
    # Define prompt function
    def get_prompt(item):
        prompt = PROMPT
        prompt += "\nYou should provide your final answer in the format \\boxed{{YOUR_ANSWER}}."
        prompt += "\n\nPlease answer the question: " + item["question"]
        prompt = generator.apply_chat_template(prompt, enable_thinking=True)
        return prompt
    
    output_data = []
    
    # Process each batch
    for batch_idx, batch in enumerate(dataset):
        output_path_for_batch = output_path.replace(".json", f"_batch_{batch_idx}.json")
        
        # Skip if already processed
        if os.path.exists(output_path_for_batch):
            logger.info(f"Batch {batch_idx} already processed, skipping...")
            with open(output_path_for_batch, "r") as f:
                batch = json.load(f)
                output_data.extend(batch)
            continue
        
        logger.info(f"Processing batch {batch_idx}...")
        
        # Initialize items
        for item in batch:
            initialize_item(item, get_prompt)
        
        # Main inference loop
        try:
            num_iteration = 0
            while True:
                num_iteration += 1
                logger.info(f"---------- Iteration {num_iteration} ----------")
                
                # Select ongoing items
                items_remained = [item for item in batch if not item["finished"]]
                if len(items_remained) == 0:
                    break
                
                logger.info(f"Among {len(batch)} items, {len(items_remained)} items are left.")
                
                # Process iteration
                items_remained = process_batch_iteration(
                    args, logger, generator, batch, items_remained,
                    search_cache, url_cache
                )
                
                # Check termination conditions
                items_remained = [item for item in items_remained if not item['finished']]
                
                if len(items_remained) > 0:
                    if num_iteration < args.max_iteration:
                        continue
                    else:
                        # Generate final answer for remaining items
                        generate_final_answers(generator, batch, args)
                        break
                else:
                    break
                    
        except Exception as e:
            logger.error(f"Error: {e}")
            save_cache(output_dir, search_cache, url_cache)
            raise
    
        # Save batch results
        output_data.extend(batch)
        save_cache(output_dir, search_cache, url_cache)
        logger.info(f"Saving results... {output_path_for_batch}...")
        
        with open(output_path_for_batch, "w") as f:
            json.dump(batch, f, indent=2)
        
        logger.info(f"Batch {batch_idx} completed. Output: {output_path_for_batch}")
        
    # Save final results
    save_cache(output_dir, search_cache, url_cache)
    logger.info(f"Saving final results... {output_path}...")
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Output file: {output_path}")


if __name__ == "__main__":
    main()
