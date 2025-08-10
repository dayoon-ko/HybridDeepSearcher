import os
import re
import json
import torch
from openai import OpenAI
from typing import List, Dict, Optional, Any
import requests
import time
from urllib.parse import urlparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


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


def get_response_from_llm(
    args,
    messages: List[Dict[str, Any]],
    depth: int = 0,
    query_for_the_prompt: str = None,
    return_type: str = "str"
):
    client = OpenAI(
      base_url=args.api_model_base_url,
      api_key=args.api_model_key,
    )
    
    try:
        if "openrouter" in args.api_model_base_url:
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
            response = requests.post(url, headers=headers, data=json.dumps(payload), verify=False)
            response = response.json()
            content = response['choices'][0]['message']['content'].strip()
        else:
            response = client.chat.completions.create(
                model=args.api_model,
                messages=messages,
                max_tokens=args.api_model_max_tokens,
                temperature=args.api_model_temperature,
                top_p=args.api_model_top_p,
                extra_body={"top_k": 20,
                            "chat_template_kwargs": {"enable_thinkng": False}}
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
                depth=depth+1, 
                query_for_the_prompt=query_for_the_prompt, 
                return_type=return_type
            )
        raise e


def extract_answer(output, mode='gen'):
    extracted_text = ''
    if mode == 'infogen':
        # Extract content after **Final Information** or **Modified Reasoning Steps**
        pattern_info = "**Final Information**"
        if pattern_info in output:
            extracted_text = output.split(pattern_info)[-1].replace("\n","").strip("```").strip()
        else:
            extracted_text = "No helpful information found."
    else:
        pattern = r'\\boxed\{(.*)\}'
        matches = re.findall(pattern, output)
        if matches:
            extracted_text = matches[-1]  # Take the last match
            inner_pattern = r'\\text\{(.*)\}'
            inner_matches = re.findall(inner_pattern, extracted_text)
            if inner_matches:
                extracted_text = inner_matches[-1]  # Take the last match
            extracted_text = extracted_text.strip("()")
    return extracted_text



def extract_between(text: str, begin_search_query_token: str, end_search_query_token: str) -> List[str]:
    '''
    Function to extract queries between two query tokens
    '''
    
    # Set pattern 
    pattern_begin = begin_search_query_token.replace('|', '\\|')
    pattern = begin_search_query_token.replace('|', '\\|') + "([\s\S]*?)" + end_search_query_token.replace('|', '\\|')
    
    # Return if wrong format
    if len(re.findall(pattern_begin, text)) > 1:
        return
    output = re.findall(pattern, text)
    if len(output) == 0:
        return
    
    # Split queries
    output = output[0].strip().strip("\n")
    output = [i.strip().strip(";") for i in output.split("\n")]
    if len(output) == 1 and ";" in output[0]:
        output = output[0].split(";")
        output = [o.strip() for o in output]
    
    # Return output when queries are extracted
    output = [o for o in output if len(o) > 0]
    return output


def replace_recent_steps(origin_str, replace_str):
    """
    Replaces specific steps in the original reasoning steps with new steps.
    If a replacement step contains "DELETE THIS STEP", that step is removed.

    Parameters:
    - origin_str (str): The original reasoning steps.
    - replace_str (str): The steps to replace or delete.

    Returns:
    - str: The updated reasoning steps after applying replacements.
    """

    def parse_steps(text):
        """
        Parses the reasoning steps from a given text.

        Parameters:
        - text (str): The text containing reasoning steps.

        Returns:
        - dict: A dictionary mapping step numbers to their content.
        """
        step_pattern = re.compile(r"Step\s+(\d+):\s*")
        steps = {}
        current_step_num = None
        current_content = []

        for line in text.splitlines():
            step_match = step_pattern.match(line)
            if step_match:
                # If there's an ongoing step, save its content
                if current_step_num is not None:
                    steps[current_step_num] = "\n".join(current_content).strip()
                current_step_num = int(step_match.group(1))
                content = line[step_match.end():].strip()
                current_content = [content] if content else []
            else:
                if current_step_num is not None:
                    current_content.append(line)
        
        # Save the last step if any
        if current_step_num is not None:
            steps[current_step_num] = "\n".join(current_content).strip()
        
        return steps

    # Parse the original and replacement steps
    origin_steps = parse_steps(origin_str)
    replace_steps = parse_steps(replace_str)

    # Apply replacements
    for step_num, content in replace_steps.items():
        if "DELETE THIS STEP" in content:
            # Remove the step if it exists
            if step_num in origin_steps:
                del origin_steps[step_num]
        else:
            # Replace or add the step
            origin_steps[step_num] = content

    # Sort the steps by step number
    sorted_steps = sorted(origin_steps.items())

    # Reconstruct the reasoning steps as a single string
    new_reasoning_steps = "\n\n".join([f"{content}" for num, content in sorted_steps])

    return new_reasoning_steps

class Generator:
    def __init__(self, model_id_or_path="Qwen/Qwen3-32B"):
        self.llm = LLM(
            model=model_id_or_path, 
            enforce_eager=True,
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=0.9,
            max_num_seqs=32, # Reduce this number if you have less GPU memory
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)

    def apply_chat_template(self, prompt, enable_thinking=True):
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
       
    def generate(self, prompts, max_tokens, sampling_params=None, n=1, stop=None, repetition_penalty=None, apply_chat=True, enable_thinking=True):
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
            
        # Set additional sampling params
        self.params["max_tokens"] = max_tokens
        if stop is not None:
            if isinstance(stop, str):
                self.params["stop"] = [stop]
            elif isinstance(stop, list):
                self.params["stop"] = stop
            self.params["include_stop_str_in_output"] = True
        if repetition_penalty is not None:
            self.params["repetition_penalty"] = repetition_penalty 
            
        # Set prompts
        if apply_chat:
            prompts = [self.apply_chat_template(p, enable_thinking) for p in prompts]
        
        # Generate responses
        responses = self.llm.generate(
            prompts, 
            SamplingParams(**self.params),
        )
        return [response.outputs[0].text for response in responses]

