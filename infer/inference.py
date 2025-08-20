import re
import os
import time
import json 
import torch
import asyncio
import logging 
import argparse
import datasets
from openai import OpenAI
from utils import *
from bing_search import *
from typing import List, Dict, Any, Optional


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
* 
"""

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", "-d", required=True, choices=["musique", "browse_comp", "hle", "gpqa", "fanoutqa", "cwq", "hotpotqa", "med_browse_comp", "multihopqa", "frames"]) 
    parser.add_argument("--dataset_split", "-sp", default="train")
    parser.add_argument("--model_id_or_path", "-m", default="dayoon/Qwen3-8B-sft")
    parser.add_argument("--apply_chat", type=bool, default=True)
    parser.add_argument("--use_jina", action="store_true", help="Whether to use jina for extracting text from urls")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--max_doc_len", type=int, default=1024) 
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=10, help="Retrieve the top k documents")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--start_idx", "-s", type=int)
    parser.add_argument("--end_idx", "-e", type=int)
    parser.add_argument("--api_model", "-am", default="qwen/qwen3-235b-a22b-2507") #"qwen/qwen3-32b"
    parser.add_argument("--api_model_max_tokens", type=int, default=2048)
    parser.add_argument("--api_model_temperature", type=float, default=0.7)
    parser.add_argument("--api_model_top_p", type=float, default=0.8)
    parser.add_argument("--api_model_base_url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api_model_key", default="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")
    parser.add_argument("--jina_api_key", default=os.environ["JINA_API_KEY"])
    parser.add_argument("--max_iteration", type=int, default=10)
    parser.add_argument("--batch_size_for_processing", type=int, default=64)
    args = parser.parse_args()
    return args



async def generate_webpage_to_reasonchain_batch(
        args,
        logger,
        original_questions: List[str],
        prev_reasonings: List[str],
        search_queries: List[List[str]],    # list of queries per sequence
        documents: Dict[str, str],          # { search_query: formatted_doc_string }
        dataset_name: str,
        max_tokens: int = 2048,
    ) -> List[Dict[str, str]]:  # returns list of dict of outputs per query per sequence
    
    logger.info(f"API model: {args.api_model}")
    def get_reasonchain_prompt(prev_reasoning, query, document):
        prompt = prompt_for_webpage_to_reasonchain_instruction.format(
            prev_reasoning=prev_reasoning, 
            search_query=query, 
            document=document
        )
        prompt = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
        return prompt
    
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
        # logger.info(f"Query {idx} completed")
        return result
    
    # Kick off all requests concurrently
    tasks = [asyncio.create_task(run_one(idx, q, p)) for idx, (q, p) in enumerate(all_user_prompts.items())]
    outputs = await asyncio.gather(*tasks)
    
    # Extract answers from outputs
    extracted_infos = {query: extract_answer(output, mode='infogen') for query, output in outputs}
    return extracted_infos


def load_dataset(args):
    # Load dataset
    if args.dataset_name == "browse_comp":
        with open("datasets/BrowseComp/test.json", "r") as f:
            dataset = json.load(f)
    elif args.dataset_name == "med_browse_comp":
        with open("datasets/MedBrowseComp/test.json", "r") as f:
            dataset = json.load(f)
    elif args.dataset_name == "musique":
        with open("datasets/MuSiQue/dev.json", "r") as f:
            dataset = json.load(f)
    elif args.dataset_name == "fanoutqa":
        with open("datasets/FanOutQA/dev.json", "r") as f:
            dataset = json.load(f)
    elif args.dataset_name == "frames":
        with open("datasets/FRAMES/test.json", "r") as f:
            dataset = json.load(f)
    else:
        dataset = datasets.load_dataset(args.dataset_name)[args.dataset_split]
        dataset = [item for item in dataset]
    
    # Select subset of dataset
    if args.start_idx is not None and args.end_idx is not None:
        dataset = dataset[args.start_idx:args.end_idx]
        
    # Split dataset into batches
    if args.batch_size_for_processing > 0:
        batches = [dataset[i:i+args.batch_size_for_processing] for i in range(0, len(dataset), args.batch_size_for_processing)]
        return batches
    else:
        return [dataset]

def get_cache(output_dir):
    # Set cache for search
    if os.path.exists(f"{output_dir}/search_cache.json"):
        with open(f"{output_dir}/search_cache.json", "r") as f:
            search_cache = json.load(f)
    else:
        search_cache = {}
    if os.path.exists(f"{output_dir}/url_cache.json"):
        with open(f"{output_dir}/url_cache.json", "r") as f:
            url_cache = json.load(f)
    else:
        url_cache = {}
    return search_cache, url_cache

def save_cache(output_dir, search_cache, url_cache):
    with open(f"{output_dir}/search_cache.json", "w") as f:
        json.dump(search_cache, f, indent=2)
    with open(f"{output_dir}/url_cache.json", "w") as f:
        json.dump(url_cache, f, indent=2)

def main():
    args = get_args()
    
    BEGIN_SEARCH_QUERY = "<|begin_search_queries|>"
    END_SEARCH_QUERY = "<|end_search_queries|>"
    BEGIN_SEARCH_RESULT = "<|begin_search_results|>"
    END_SEARCH_RESULT = "<|end_search_results|>"
    
    dataset_name = args.dataset_name.split("/")[-1]
    model_name = args.model_id_or_path.split("/")[-1]
    output_dir = f"{args.output_dir}/{args.dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/output.json"
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    logger.info(f"Output file path: {output_path}")
    
    # Load dataset
    dataset = load_dataset(args)
    
    # Get cache
    search_cache, url_cache = get_cache(output_dir)
    
    # Use Generator for LLM and tokenizer
    generator = Generator(model_id_or_path=args.model_id_or_path)
    
    # Get prompt
    def get_prompt(item):
        prompt = PROMPT
        prompt += "\nYou should provide your final answer in the format \\boxed{{YOUR_ANSWER}}."
        prompt += "\n\nPlease answer the question: " + item["question"]
        prompt = generator.apply_chat_template(prompt, enable_thinking=True)
        return prompt
    
    output_data = []
    
    for batch_idx, batch in enumerate(dataset):
        # Check if batch is already processed
        output_path_for_batch = output_path.replace(".json", f"_batch_{batch_idx}.json")
        if os.path.exists(output_path_for_batch):
            logger.info(f"Batch {batch_idx} already processed, skipping...")
            with open(output_path_for_batch, "r") as f:
                batch = json.load(f)
                output_data.extend(batch)
            continue
        else:
            logger.info(f"Processing batch {batch_idx}...")
        
        # Initialize items in batch
        for item in batch:
            if "question" not in item:
                if "Question" in item:
                    item["question"] = item["Question"]
                    del item["Question"]
                elif "query" in item:
                    item["question"] = item["query"]
                    del item["query"]
            item["finished"] = False
            item["prompt"] = get_prompt(item)
            item["output"] = ""
            item["output_history"] = []
            item["search_count"] = []
            item["executed_search_queries"] = []
            item["related_info_analysis"] = []
            item["relevant_info"] = {}
        
        # Main loop
        try:
            num_iteration = 0
            while True:
                num_iteration += 1
                logger.info(f"---------- Iteration {num_iteration} ----------")
                
                # Select only ongoing items
                items_remained = [item for item in batch if not item["finished"]]
                if len(items_remained) == 0:
                    break
                logger.info(f"Among {len(batch)} items, {len(items_remained)} items are left.")
                
                # Initialize batch variables
                batch_relevant_info = []
                batch_original_questions = []
                batch_prev_reasonings = []
                batch_search_queries = []
                batch_documents = {}
                batch_sequences = []

                # Collect URLs to fetch across all sequences
                all_urls_to_fetch = set()
                all_queries_to_search = set()
                url_snippets = {}
                
                # Prepare prompts & Inference
                prompts = [item["prompt"] for item in items_remained]
                responses = generator.generate(
                                prompts, 
                                max_tokens=args.max_tokens, 
                                stop=[END_SEARCH_QUERY], 
                                apply_chat=False,
                                enable_thinking=True, # 'enable_thinking = True' doesn't do anything
                            )
                
                # Store raw model outputs for each item
                items_remained_ = []
                responses_ = []
                for item,  response in zip(items_remained, responses):
                    if "</think>" not in response:
                        response = response.replace(f"\n\n{BEGIN_SEARCH_QUERY}", f"\n</think>\n\n{BEGIN_SEARCH_QUERY}")
                    search_queries = extract_between(response, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)
                    item["prompt"] += response
                    item["output"] += response
                    item["output_history"].append(response)

                    # If no search queries are found, extract final answer
                    if search_queries is None:    
                        item["finished"] = True
                        item["generated_answer"] = extract_answer(response)
                        continue
                    
                    # Select search queries to execute
                    relevant_info = {}
                    for search_query in search_queries:
                        if search_query in set(item['executed_search_queries']):
                            repeat_msg = (
                                f"\n{BEGIN_SEARCH_RESULT}\nYou have searched this query. "
                                "Please refer to previous results.\n"
                                f"{END_SEARCH_RESULT}\n"
                            )
                            relevant_info[search_query] = repeat_msg
                            continue
                        if search_query in search_cache:
                            results = search_cache[search_query]
                            relevant_info[search_query] = results[:args.top_k]
                        else:
                            all_queries_to_search.add(search_query)
                            relevant_info[search_query] = None
                    
                    items_remained_.append(item)
                    responses_.append(response)
                    batch_relevant_info.append(relevant_info)
                
                items_remained = items_remained_
                responses = responses_
                
                # Execute search queries
                if all_queries_to_search:
                    logger.info(f"Executing {len(all_queries_to_search)} queries...")
                    all_search_results = asyncio.run(
                        jina_web_search_all_queries(
                            list(all_queries_to_search),
                            jina_api_key=args.jina_api_key,
                        )
                    )
                    
                # Postprocess search results
                for item, response, relevant_info in zip(items_remained, responses, batch_relevant_info):            
                    # Update relevant_info with search results
                    for query, info in relevant_info.items():
                        if info is None:
                            results = all_search_results[query]
                            relevant_info[query] = results[:args.top_k]
                            search_cache[query] = results
                    
                    item['relevant_info'].update(relevant_info)
                    item['search_count'].append(len(relevant_info))
                    item['executed_search_queries'].extend(relevant_info.keys())
                    
                    # Extract URLs and snippets
                    urls_to_fetch = []
                    snippets = {}
                    for _, results in relevant_info.items():
                        for info in results:
                            if 'url' in info:
                                urls_to_fetch.append(info['url'])
                                if 'snippet' in info:
                                    snippets[info['url']] = info['snippet']
                        
                    # Filter URLs that are not cached
                    urls_to_fetch_filtered = [u for u in urls_to_fetch if u not in url_cache]
                    cached_urls = [u for u in urls_to_fetch if u in url_cache]
                    
                    # Store info for all_urls_to_fetch and url_snippets
                    for url in urls_to_fetch_filtered:
                        all_urls_to_fetch.add(url)
                        url_snippets[url] = snippets.get(url, "")
                    
                    # Truncate previous reasoning steps
                    all_reasoning_steps = item['output']
                    all_reasoning_steps = all_reasoning_steps.replace('\n\n', '\n').split("\n")
                    truncated_prev_reasoning = ""
                    for i, step in enumerate(all_reasoning_steps):
                        truncated_prev_reasoning += f"Step {i + 1}: {step}\n\n"
                    prev_steps = truncated_prev_reasoning.split('\n\n')
                    if len(prev_steps) <= 5:
                        truncated_prev_reasoning = '\n\n'.join(prev_steps)
                    else:
                        truncated_prev_reasoning = ''
                        for i, step in enumerate(prev_steps):
                            if i == 0 or i >= len(prev_steps) - 4 or BEGIN_SEARCH_QUERY in step or BEGIN_SEARCH_RESULT in step:
                                truncated_prev_reasoning += step + '\n\n'
                            else:
                                if truncated_prev_reasoning[-len('\n\n...\n\n'):] != '\n\n...\n\n':
                                    truncated_prev_reasoning += '...\n\n'
                    truncated_prev_reasoning = truncated_prev_reasoning.strip('\n')

                    # Collect data for batch processing
                    batch_original_questions.append(item['question'])
                    batch_prev_reasonings.append(truncated_prev_reasoning)
                    batch_search_queries.append(relevant_info.keys())
                    batch_sequences.append(item)
                    
                # Batch fetch all URLs at once to optimize speed
                if all_urls_to_fetch:
                    logger.info(f"Fetching {len(all_urls_to_fetch)} URLs...")
                    try:
                        fetched_contents = asyncio.run(
                            fetch_page_content(
                                list(all_urls_to_fetch),
                                use_jina=args.use_jina,
                                jina_api_key=args.jina_api_key,
                                snippets=url_snippets  # Do not pass snippets when updating url_cache directly
                            )
                        )   
                        logger.info(f"Fetched {len(fetched_contents)} URLs successfully.")
                    except Exception as e:
                        logger.info(f"Error during batch URL fetching: {e}")
                        fetched_contents = {url: f"Error fetching URL: {e}" for url in all_urls_to_fetch}
                    # Update cache with fetched contents
                    for url, content in fetched_contents.items():
                        url_cache[url] = content

                # After fetching, prepare formatted documents for batch summarization
                for relevant_info, executed_queries in zip(batch_relevant_info, batch_search_queries): 
                    # relevant_info: [ [{query: doc_info}] for each search_query] for each item ] 
                    # executed_queries: [ search_queries ] for each item 
                    formatted_documents = {}
                    for sub_query in executed_queries:
                        doc_str = ""
                        sub_relevant_info = relevant_info[sub_query]
                        if type(sub_relevant_info) == str:
                            formatted_documents[sub_query] = sub_relevant_info # Already searched
                            continue
                        for i, doc_info in enumerate(sub_relevant_info):
                            url = doc_info.get('url', "")
                            raw_context = url_cache.get(url, "")
                            snippet = doc_info["snippet"].replace('<b>', '').replace('</b>', '') if doc_info["snippet"] else None
                            doc_info['snippet'] = snippet
                            success, filtered_context = extract_snippet_with_context(raw_context, snippet, context_chars=args.max_doc_len)
                            context = filtered_context if success else raw_context[:args.max_doc_len]
                            doc_info['context'] = context
                            doc_str += f"**Web Page {i + 1}:**\n"
                            doc_str += json.dumps(doc_info, ensure_ascii=False, indent=2) + "\n"
                        formatted_documents[sub_query] = doc_str
                    batch_documents.update(formatted_documents)

                # Batch summarization
                if batch_sequences:
                    logger.info(f"Batch summarizing {len(batch_sequences)} sequences with generate_webpage_to_reasonchain_batch...")
                    webpage_analyses = asyncio.run(
                        generate_webpage_to_reasonchain_batch(
                            args,
                            logger,
                            original_questions=batch_original_questions, # list of questions
                            prev_reasonings=batch_prev_reasonings, # list of previous reasoning steps
                            search_queries=batch_search_queries, # [[search_queries] for each question]
                            documents=batch_documents, # {search_query: formatted_doc_string} for all questions
                            dataset_name=args.dataset_name,
                            max_tokens=args.api_model_max_tokens
                        )
                    )
                    logger.info("Batch summarization completed, assigning outputs to sequences...")

                    for seq, queries in zip(batch_sequences, batch_search_queries):
                        combined_analysis = ""
                        for query in queries:
                            analysis = webpage_analyses[query]
                            combined_analysis += f"{query}: {analysis}\n"
                            seq["related_info_analysis"].append(analysis)
                        
                        append_text = f"\n\n{BEGIN_SEARCH_RESULT}\n{combined_analysis.strip()}\n{END_SEARCH_RESULT}\n\n<think>\n"
                        seq['prompt'] += append_text
                        seq['output'] += append_text            
                
                items_remained = [item for item in items_remained if not item['finished']]
                if len(items_remained) > 0:
                    # If the number of iterations exceeds the maximum number of iterations, continue
                    if num_iteration < args.max_iteration:
                        continue 
                    # Generate final answer for remaining items
                    else:
                        items_to_generate_final_answer = [
                            item for item in batch 
                            if not item["finished"] or "generated_answer" not in item 
                            or item["generated_answer"] == ""
                        ]
                        
                        # Preprocess prompts for final answer generation
                        def map_item(item):
                            prompt = item["prompt"].strip() 
                            output = item["output"].strip()
                            while True: 
                                if prompt.endswith(END_SEARCH_QUERY):
                                    prompt = prompt.strip(END_SEARCH_QUERY).strip()
                                    output = output.strip(END_SEARCH_QUERY).strip()
                                elif prompt.endswith(BEGIN_SEARCH_QUERY):
                                    prompt = prompt.strip(BEGIN_SEARCH_QUERY).strip()
                                    output = output.strip(BEGIN_SEARCH_QUERY).strip()
                                elif prompt.endswith(END_SEARCH_RESULT):
                                    prompt = prompt.strip(END_SEARCH_RESULT).strip()
                                    output = output.strip(END_SEARCH_RESULT).strip()
                                elif prompt.endswith(BEGIN_SEARCH_RESULT):
                                    prompt = prompt.strip(BEGIN_SEARCH_RESULT).strip()
                                    output = output.strip(BEGIN_SEARCH_RESULT).strip()
                                else:
                                    break
                            prompt = prompt.strip()
                            output = output.strip()
                            if not prompt.endswith("</think>"):
                                prompt += "\n</think>\n\n**Final Answer:**\n\\boxed{"
                                output += "\n</think>\n\n**Final Answer:**\n\\boxed{"
                            item["prompt"] = prompt 
                            item["output"] = output
                            item["output_history"].append("\n</think>\n\n**Final Answer:**\n\\boxed{")
                            return item
                        
                        # Generate final answer
                        items_to_generate_final_answer = list(map(map_item, items_to_generate_final_answer))
                        prompts = [item["prompt"] for item in items_to_generate_final_answer]
                        responses = generator.generate(
                                        prompts, 
                                        max_tokens=args.max_tokens, 
                                        apply_chat=False,
                                        enable_thinking=False,
                                    )
                        
                        # Postprocess outputs
                        for item, response in zip(items_to_generate_final_answer, responses):
                            item["prompt"] += response
                            item["output"] += response
                            item["output_history"][-1] += response
                            item["finished"] = True
                            item["generated_answer"] = response.strip("}")
                        break 
                else:
                    break
        except Exception as e:
            logger.info(f"Error: {e}")
            save_cache(output_dir, search_cache, url_cache)
            exit()
    
        # Save results
        output_data.extend(dataset)
        save_cache(output_dir, search_cache, url_cache)
        logger.info(f"Saving results... {output_path_for_batch}...")
        with open(output_path_for_batch, "w") as f:
            json.dump(batch, f, indent=2)
        logger.info(f"{'*'*30}\nOutput file: {output_path_for_batch}\n{'*'*30}")
        print("Done!")
        
    # Save final results
    save_cache(output_dir, search_cache, url_cache)
    print(f"Saving final results... {output_path}...")
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"{'*'*30}\nOutput file: {output_path}\n{'*'*30}")
    
if __name__ == "__main__":
    main()
    
    
