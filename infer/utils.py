"""
Utility Functions for LLM Inference

This module provides:
- LLM generation wrapper (vLLM-based Generator class)
- API-based LLM client for webpage summarization
- Text extraction utilities (answer extraction, query parsing)
"""

import re
import json
import time

import torch
import requests
from openai import OpenAI
from typing import List, Dict, Any, Optional
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


# ============================================================================
# Prompts
# ============================================================================

prompt_for_webpage_to_reasonchain_instruction = """**Task Instruction:**

You are tasked with reading and analyzing web pages based on the following inputs: **Previous Reasoning Steps**, **Current Search Query**, and **Searched Web Pages**. Your objective is to extract relevant and helpful information for **Current Search Query** from the **Searched Web Pages** and seamlessly integrate this information into the **Previous Reasoning Steps** to continue reasoning for the original question.

**Guidelines:**

1. **Analyze the Searched Web Pages:**
- Carefully review the content of each searched web page.
- Identify factual information that is relevant to the **Current Search Query** and can aid in the reasoning process for the original question.

2. **Extract Relevant Information:**
- Select the information from the Searched Web Pages that directly contributes to advancing the **Previous Reasoning Steps**.
- Ensure that the extracted information is accurate and relevant.

3. **Output Format:**
- **If the web pages provide helpful information for current search query:** Present the information beginning with `**Final Information**` as shown below.
**Final Information**

[Helpful information]

- **If the web pages do not provide any helpful information for current search query:** Output the following text.

**Final Information**

No helpful information found.

**Inputs:**
- **Previous Reasoning Steps:**  
{prev_reasoning}

- **Current Search Query:**  
{search_query}

- **Searched Web Pages:**  
{document}

Now you should analyze each web page and find helpful information based on the current search query "{search_query}" and previous reasoning steps.
"""


# ============================================================================
# API-based LLM Client
# ============================================================================

def get_response_from_llm(
    args,
    messages: List[Dict[str, Any]],
    depth: int = 0,
    query_for_the_prompt: Optional[str] = None,
    return_type: str = "str"
) -> Any:
    """
    Get response from an API-based LLM (OpenRouter or local vLLM server).
    
    Args:
        args: Arguments containing API configuration.
        messages: Chat messages in OpenAI format.
        depth: Recursion depth for retries.
        query_for_the_prompt: Optional query identifier for tuple return.
        return_type: "str" for string response, "tuple" for (query, response).
    
    Returns:
        Response string or tuple based on return_type.
    """
    client = OpenAI(
        base_url=args.api_model_base_url,
        api_key=args.api_model_key,
    )
    
    try:
        if "openrouter" in args.api_model_base_url:
            # OpenRouter API with reasoning exclusion
            url = f"{args.api_model_base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {args.api_model_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": args.api_model,
                "messages": messages,
                "reasoning": {
                    "max_tokens": 0,     
                    "exclude": True      
                }
            }
            response = requests.post(
                url, headers=headers, data=json.dumps(payload), verify=False
            )
            response = response.json()
            content = response['choices'][0]['message']['content'].strip()
        else:
            # Standard OpenAI-compatible API
            response = client.chat.completions.create(
                model=args.api_model,
                messages=messages,
                max_tokens=args.api_model_max_tokens,
                temperature=args.api_model_temperature,
                top_p=args.api_model_top_p,
                extra_body={
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": False}
                }
            )
            if hasattr(response.choices[0].message, 'content') and response.choices[0].message.content:
                content = response.choices[0].message.content
        
        if return_type == "str":
            return content.strip()
        elif return_type == "tuple":
            return query_for_the_prompt, content.strip()
        
    except Exception as e:
        if depth < 512:
            time.sleep(0.1)
            return get_response_from_llm(
                args=args,
                messages=messages, 
                depth=depth + 1, 
                query_for_the_prompt=query_for_the_prompt, 
                return_type=return_type
            )
        raise e


# ============================================================================
# Answer Extraction
# ============================================================================

def extract_answer(output: str, mode: str = 'gen') -> str:
    """
    Extract answer from model output.
    
    Args:
        output: Model output string.
        mode: 'gen' for boxed answer extraction, 'infogen' for Final Information extraction.
    
    Returns:
        Extracted answer string.
    """
    extracted_text = ''
    
    if mode == 'infogen':
        # Extract content after **Final Information**
        pattern_info = "**Final Information**"
        if pattern_info in output:
            extracted_text = output.split(pattern_info)[-1]
            extracted_text = extracted_text.replace("\n", "").strip("```").strip()
        else:
            extracted_text = "No helpful information found."
    else:
        # Extract content from \boxed{...}
        pattern = r'\\boxed\{(.*)\}'
        matches = re.findall(pattern, output)
        if matches:
            extracted_text = matches[-1]
            
            # Handle nested \text{...}
            inner_pattern = r'\\text\{(.*)\}'
            inner_matches = re.findall(inner_pattern, extracted_text)
            if inner_matches:
                extracted_text = inner_matches[-1]
            
            extracted_text = extracted_text.strip("()")
    
    return extracted_text


# ============================================================================
# Query Extraction
# ============================================================================

def extract_between(
    text: str, 
    begin_search_query_token: str, 
    end_search_query_token: str
) -> Optional[List[str]]:
    """
    Extract queries between two delimiter tokens.
    
    Args:
        text: Text containing search queries.
        begin_search_query_token: Start delimiter (e.g., "<|begin_search_queries|>").
        end_search_query_token: End delimiter (e.g., "<|end_search_queries|>").
    
    Returns:
        List of extracted queries or None if no valid queries found.
    """
    # Escape special characters for regex
    pattern_begin = begin_search_query_token.replace('|', '\\|')
    pattern = (
        begin_search_query_token.replace('|', '\\|') + 
        r"([\s\S]*?)" + 
        end_search_query_token.replace('|', '\\|')
    )
    
    # Return None if multiple begin tokens found (malformed)
    if len(re.findall(pattern_begin, text)) > 1:
        return None
    
    output = re.findall(pattern, text)
    if len(output) == 0:
        return None
    
    # Split queries by newline or semicolon
    output = output[0].strip().strip("\n")
    output = [i.strip().strip(";") for i in output.split("\n")]
    
    if len(output) == 1 and ";" in output[0]:
        output = output[0].split(";")
        output = [o.strip() for o in output]
    
    # Filter empty queries
    output = [o for o in output if len(o) > 0]
    return output if output else None


# ============================================================================
# vLLM-based Generator
# ============================================================================

class Generator:
    """
    vLLM-based text generator for local LLM inference.
    
    Supports Qwen3 models with thinking mode and custom sampling parameters.
    """
    
    def __init__(self, model_id_or_path: str = "Qwen/Qwen3-32B"):
        """
        Initialize the generator with a model.
        
        Args:
            model_id_or_path: HuggingFace model ID or local path.
        """
        self.llm = LLM(
            model=model_id_or_path, 
            enforce_eager=True,
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=0.9,
            max_num_seqs=32,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
        self.params = {}

    def apply_chat_template(
        self, 
        prompt: str, 
        enable_thinking: bool = True
    ) -> str:
        """
        Apply chat template to a prompt.
        
        Args:
            prompt: User prompt string.
            enable_thinking: Whether to enable thinking mode.
        
        Returns:
            Formatted prompt string with chat template applied.
        """
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
       
    def generate(
        self, 
        prompts: List[str], 
        max_tokens: int,
        sampling_params: Optional[Dict] = None,
        n: int = 1,
        stop: Optional[List[str]] = None,
        repetition_penalty: Optional[float] = None,
        apply_chat: bool = True,
        enable_thinking: bool = True
    ) -> List[str]:
        """
        Generate responses for a batch of prompts.
        
        Args:
            prompts: List of input prompts.
            max_tokens: Maximum tokens to generate.
            sampling_params: Custom sampling parameters dict.
            n: Number of responses per prompt.
            stop: List of stop sequences.
            repetition_penalty: Repetition penalty value.
            apply_chat: Whether to apply chat template.
            enable_thinking: Whether to enable thinking mode.
        
        Returns:
            List of generated response strings.
        """
        # Set sampling parameters
        if sampling_params:
            self.params = sampling_params
        else:
            self.params = {
                "temperature": 0.6 if enable_thinking else 0.7,
                "top_p": 0.95 if enable_thinking else 0.8,
                "top_k": 20,
                "min_p": 0.0,
                "n": n
            }
        
        self.params["max_tokens"] = max_tokens
        
        # Handle stop sequences
        if stop is not None:
            if isinstance(stop, str):
                self.params["stop"] = [stop]
            elif isinstance(stop, list):
                self.params["stop"] = stop
            self.params["include_stop_str_in_output"] = True
        
        if repetition_penalty is not None:
            self.params["repetition_penalty"] = repetition_penalty 
        
        # Apply chat template if needed
        if apply_chat:
            prompts = [self.apply_chat_template(p, enable_thinking) for p in prompts]
        
        # Generate responses
        responses = self.llm.generate(
            prompts, 
            SamplingParams(**self.params),
        )
        
        return [response.outputs[0].text for response in responses]
